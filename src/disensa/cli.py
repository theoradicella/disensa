"""Command line: `disensa run | sweep | app | ping`."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import typer

from disensa.checks import config_checks
from disensa.config import ExperimentConfig
from disensa.runs import load_meta, new_run_dir

app = typer.Typer(no_args_is_help=True, help="disensa: opinion dynamics with numeric or LLM agents")


@app.command()
def run(config: Path, steps: int = typer.Option(None, help="override `steps`")):
    """Run an experiment from a YAML config (numeric or LLM, set by `dynamics.type`)."""
    from disensa.engine import RunFailed
    from disensa.engine import run as run_experiment
    from disensa.report import format_report

    cfg = ExperimentConfig.from_yaml(config)
    if steps:
        cfg.steps = steps
    checks = config_checks(cfg)
    for c in checks:
        color = "red" if c.level == "error" else "yellow"
        typer.secho(f"{c.level}: {c.message}", fg=color, err=True)
        typer.secho(f"         see the docs: {c.docs_title} (/docs#{c.docs})", fg=color, err=True)
    if any(c.level == "error" for c in checks):
        raise typer.Exit(2)
    d = cfg.dynamics
    if d.type == "llm" and any(m.name.startswith("ollama") and not m.api_base for m in d.models):
        from disensa.llm import ollama

        if err := ollama.ensure_running():
            raise typer.BadParameter(err)
    run_dir = new_run_dir(cfg)
    what = f"numeric/{d.rule}" if d.type == "numeric" else ", ".join(m.name for m in d.models)
    typer.echo(
        f"{cfg.name}: {cfg.network.n} agents on {cfg.network.topology}, {cfg.steps} steps, {what}"
    )
    t0 = time.time()

    def progress(t: int, m: dict) -> None:
        typer.echo(
            f"  step {t:3d}  pol {m['polarization']:.3f}  dis {m['global_disagreement']:.3f}  "
            f"nci {m['neighbor_correlation']:+.3f}  errors {m['errors']}  [{time.time() - t0:.0f}s]"
        )

    try:
        asyncio.run(run_experiment(cfg, run_dir, progress))
    except RunFailed as e:
        typer.echo(format_report(e.report))
        typer.secho(f"run failed: {e.reason}", fg="red", err=True)
        typer.echo(f"finished steps saved to {run_dir}")
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        typer.secho(f"cancelled; finished steps saved to {run_dir}", fg="yellow", err=True)
        raise typer.Exit(130) from None
    if d.type == "llm":
        typer.echo(format_report(load_meta(run_dir).get("report")))
    typer.echo(f"saved to {run_dir}")


@app.command()
def sweep(
    file: Path,
    dry_run: bool = typer.Option(False, "--dry-run", help="list the runs, run nothing"),
    shard: str = typer.Option(None, help="run share i of N (1-based), e.g. 2/4"),
    cache_dir: str = typer.Option(None, help="override dynamics.cache_dir for every run"),
):
    """Run every combination of a sweep file, one after the other (see /docs#sweeps)."""
    from disensa import sweep as sw

    try:
        spec, base = sw.load_sweep(file)
        runs = sw.expand(spec, base, Path(cache_dir) if cache_dir else None)
        part = sw.parse_shard(shard)
    except sw.SweepError as e:
        typer.secho(f"error: {e}", fg="red", err=True)
        raise typer.Exit(2) from None
    todo = sw.shard(runs, part)
    sw.refresh(runs, spec.output_dir / spec.name)
    left = [r for r in todo if r.status != "finished"]
    llm = sum(r.config.dynamics.type == "llm" for r in todo)
    where = f", shard {part[0]}/{part[1]}" if part else ""
    typer.echo(
        f"{spec.name}: {len(runs)} runs{where}: {len(todo)} here ({llm} LLM), "
        f"{len(todo) - len(left)} already finished, {len(left)} to run"
    )
    typer.echo(
        f"estimated LLM calls: {sum(r.llm_calls for r in left):,} to make "
        f"({sum(r.llm_calls for r in todo):,} in all), n_agents × (steps + 1) per LLM run"
    )
    if dry_run:
        typer.echo(sw.table(todo))
        return
    d = [r.config.dynamics for r in left if r.config.dynamics.type == "llm"]
    if any(m.name.startswith("ollama") and not m.api_base for dyn in d for m in dyn.models):
        from disensa.llm import ollama

        if err := ollama.ensure_running():
            raise typer.BadParameter(err)
    try:
        done = asyncio.run(sw.run_sweep(spec, runs, todo, Path(file), echo=typer.echo))
    except KeyboardInterrupt:
        typer.secho("sweep stopped; finished runs are kept and skipped next time", fg="yellow")
        raise typer.Exit(130) from None
    typer.echo(sw.table(done))
    if any(r.status == "failed" for r in done):
        raise typer.Exit(1)


@app.command()
def score(
    path: Path,
    model: str = typer.Option(..., help="the scorer, a LiteLLM model name"),
    api_base: str = typer.Option(None, help="server address, e.g. http://localhost:8000/v1"),
    field: str = typer.Option("message", help="which text to rate: message or reasoning"),
    max_concurrency: int = typer.Option(8, help="parallel requests to the scorer"),
):
    """Rate every logged agent text of a run (or of every run of a sweep) with a scorer model.

    A measurement only: the scores are written to scores.parquet and never change the run.
    """
    from disensa import scorer
    from disensa.llm import LLMClient, ollama

    if field not in scorer.FIELDS:
        raise typer.BadParameter(f"--field must be one of {', '.join(scorer.FIELDS)}")
    dirs = [d for d in scorer.run_dirs(path) if scorer.agent_models(load_meta(d))]
    if not dirs:
        raise typer.BadParameter(f"no LLM run folder at {path}")
    if model.startswith("ollama") and not api_base and (err := ollama.ensure_running()):
        raise typer.BadParameter(err)
    client = LLMClient(max_concurrency=max_concurrency)
    for d in dirs:
        if model in scorer.agent_models(load_meta(d)):
            typer.secho(
                f"warning: {model} is also one of the agents' models in {d.name}: it would "
                "be rating its own texts (self-rating bias). Prefer a scorer that is not.",
                fg="yellow",
                err=True,
            )
        rows = asyncio.run(scorer.score_run(d, model, field, api_base, client))
        ok = rows["score"].notna().sum()
        unclear = (rows["score"].isna() & rows["error"].isna()).sum()
        typer.echo(
            f"{d.name}: {len(rows)} texts, {ok} scored, {unclear} unclear, "
            f"{rows['error'].notna().sum()} failed -> scores.parquet"
        )


validation_app = typer.Typer(
    no_args_is_help=True, help="Human validation of self-rating and scorer (see /docs#validation)"
)
app.add_typer(validation_app, name="validation")


@validation_app.command("export")
def validation_export(
    paths: list[Path],
    n: int = typer.Option(100, help="how many messages to sample"),
    seed: int = typer.Option(0, help="seed of the sample"),
    out: str = typer.Option(..., help="folder to write, e.g. validation/topics_core"),
):
    """Sample messages for blind human rating, stratified by self-rated level."""
    from disensa import validation

    out = Path(out)
    try:
        sample, key = validation.export(paths, out, n=n, seed=seed)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    levels = key["self_rating"].map(validation._level).value_counts().sort_index()
    typer.echo(
        f"{len(sample)} messages from {key['run'].nunique()} runs -> {out}/sample.csv "
        f"(by self-rated level: {', '.join(f'{k:+d}: {v}' for k, v in levels.items())})"
    )
    typer.echo(
        f"give raters sample.csv and INSTRUCTIONS.md, never key.csv; collect "
        f"ratings_<rater>.csv into {out}/"
    )


@validation_app.command("report")
def validation_report(folder: Path):
    """Agreement between raters, and of self-rating and each scorer with the raters."""
    from disensa import validation

    try:
        r = validation.report(folder)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    typer.echo(validation.to_markdown(r))
    typer.echo(f"written to {folder}/report.json and {folder}/report.md")


@app.command()
def analyze(
    paths: list[Path],
    out: str = typer.Option(..., help="folder for the tables and PDF figures"),
    measure: str = typer.Option("self", help="opinions for the response curve: self or a scorer"),
    delta: float = typer.Option(0.5, help="gap that separates two opinion clusters"),
):
    """Tables and thesis figures for a set of runs or sweep folders (needs matplotlib)."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        raise typer.BadParameter(
            "the figures need matplotlib: pip install 'disensa[analysis]'"
        ) from None
    from disensa.analysis.analyze import analyze as run_analysis

    try:
        run_analysis(paths, Path(out), measure, delta, echo=typer.echo)
    except (ValueError, FileNotFoundError) as e:
        raise typer.BadParameter(str(e)) from None


@app.command(name="app")
def app_cmd(
    port: int = typer.Option(8765, help="port on 127.0.0.1"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="open it in your browser"),
):
    """Open the web app: configure runs, start them, and watch them live."""
    from disensa.app.server import serve

    serve(port=port, open_browser=open_browser)


@app.command()
def ping(
    model: str,
    prompt: str = "In one sentence: is remote work good for society?",
    api_base: str = typer.Option(
        None, help="server address, e.g. http://localhost:8000/v1 for vLLM"
    ),
    think: bool = typer.Option(
        None, "--think/--no-think", help="switch thinking on or off (default: not sent)"
    ),
):
    """Send one prompt to a model, to check that it answers."""
    from disensa.llm import LLMClient, ollama

    if model.startswith("ollama") and not api_base and (err := ollama.ensure_running()):
        raise typer.BadParameter(err)

    t0 = time.time()
    resp = asyncio.run(
        LLMClient().complete(
            model,
            [{"role": "user", "content": prompt}],
            api_base=api_base,
            think=think,
            use_cache=False,
        )
    )
    typer.echo(f"[{resp.model}] {resp.text}")
    tokens = resp.usage.get("completion_tokens")
    typer.echo(
        f"({tokens if tokens is not None else '?'} reply tokens, {time.time() - t0:.1f}s, "
        + (f"thinking: {len(resp.thinking)} characters)" if resp.thinking else "no thinking)")
    )


if __name__ == "__main__":
    app()
