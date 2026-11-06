"""Command line: `disensa run <config>`, `disensa app`, `disensa ping <model>`."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import typer

from disensa.checks import config_checks
from disensa.config import ExperimentConfig
from disensa.runs import new_run_dir

app = typer.Typer(no_args_is_help=True, help="disensa: opinion dynamics with numeric or LLM agents")


@app.command()
def run(config: Path, steps: int = typer.Option(None, help="override `steps`")):
    """Run an experiment from a YAML config (numeric or LLM, set by `dynamics.type`)."""
    from disensa.engine import run as run_experiment

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

    asyncio.run(run_experiment(cfg, run_dir, progress))
    typer.echo(f"saved to {run_dir}")


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
