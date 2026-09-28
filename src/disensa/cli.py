"""Command line: `disensa run configs/quick.yaml`, `disensa baseline ...`, `disensa ping ...`."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import typer
from dotenv import load_dotenv

from disensa.baselines import run_baseline
from disensa.config import ExperimentConfig
from disensa.metrics import all_metrics
from disensa.networks import build_network, initial_opinions
from disensa.runs import RunWriter, new_run_dir, save_metrics, save_trajectory, update_meta

app = typer.Typer(no_args_is_help=True, help="disensa: LLM-agent opinion dynamics")
load_dotenv()  # API keys from .env, if present

METRIC_KEYS = ("polarization", "global_disagreement", "neighbor_correlation")


def _print_change(first: dict, last: dict) -> None:
    for k in METRIC_KEYS:
        typer.echo(f"  {k:22s} {first[k]:+.3f} -> {last[k]:+.3f}")


@app.command()
def run(config: Path, days: int = typer.Option(None, help="override simulation.days")):
    """Run an LLM-agent simulation from a YAML config."""
    from disensa.engine import Simulation

    cfg = ExperimentConfig.from_yaml(config)
    if days:
        cfg.simulation.days = days
    run_dir = new_run_dir(cfg)
    writer = RunWriter(run_dir)
    simulation = Simulation(cfg)
    models = sorted({a.model.name for a in simulation.agents})
    typer.echo(
        f"{cfg.name}: {len(simulation.agents)} agents on {cfg.network.topology}, "
        f"{cfg.simulation.days} days, models: {', '.join(models)}"
    )
    t0 = time.time()

    def progress(day: int, m: dict) -> None:
        typer.echo(
            f"  day {day:3d}  pol {m['polarization']:.3f}  dis {m['global_disagreement']:.3f}  "
            f"nci {m['neighbor_correlation']:+.3f}  errors {int(m['errors'])}  "
            f"[{time.time() - t0:.0f}s]"
        )

    try:
        history = asyncio.run(simulation.run(writer, progress))
    finally:
        writer.close()
    c = simulation.client
    update_meta(
        run_dir,
        finished=True,
        seconds=round(time.time() - t0, 1),
        llm_calls=c.calls,
        cache_hits=c.cache_hits,
    )
    _print_change(history[0], history[-1])
    typer.echo(f"saved to {run_dir}")


@app.command()
def baseline(config: Path, every: int = typer.Option(10, help="metrics every N steps")):
    """Run a numeric baseline model from a YAML config."""
    cfg = ExperimentConfig.from_yaml(config)
    if cfg.baseline is None:
        raise typer.BadParameter("config has no `baseline` section")
    g = build_network(cfg.network, seed=cfg.seed)
    x0 = initial_opinions(cfg.opinions, g.number_of_nodes(), seed=cfg.seed)
    traj = run_baseline(g, x0, cfg.baseline, seed=cfg.seed)

    rows = [{"step": t, **all_metrics(g, traj[t])} for t in range(0, len(traj), every)]
    if rows[-1]["step"] != len(traj) - 1:
        rows.append({"step": len(traj) - 1, **all_metrics(g, traj[-1])})

    run_dir = new_run_dir(cfg)
    save_trajectory(run_dir, traj)
    save_metrics(run_dir, rows)

    typer.echo(f"{cfg.baseline.model} on {cfg.network.topology} (N={cfg.network.n})")
    _print_change(rows[0], rows[-1])
    typer.echo(f"saved to {run_dir}")


@app.command()
def ping(model: str, prompt: str = "In one sentence: is remote work good for society?"):
    """Send one prompt to any model (checks keys / endpoints)."""
    from disensa.llm import LLMClient

    resp = asyncio.run(
        LLMClient().complete(model, [{"role": "user", "content": prompt}], use_cache=False)
    )
    typer.echo(f"[{resp.model}] {resp.text}")


@app.command()
def inspect(
    run_dir: Path,
    agent: int = typer.Option(None, help="show one agent's day-by-day story"),
    min_delta: float = typer.Option(0.5, help="smallest opinion change counted as a shift"),
    top: int = typer.Option(10, help="how many of the largest shifts to list"),
    csv: bool = typer.Option(False, help="also write shifts.csv / model_summary.csv"),
):
    """Inspect a finished run: per-model behaviour and the largest opinion shifts, or one agent."""
    import pandas as pd

    from disensa.analysis import (
        agent_timeline,
        change_points,
        load_run,
        model_summary,
        opinion_shifts,
    )

    pd.set_option("display.width", 160, "display.max_colwidth", 80)
    run = load_run(run_dir)

    if agent is not None:
        a = run.agents[run.agents["agent"] == agent].iloc[0]
        typer.echo(
            f"Agent {agent}: {a['age']}-year-old {a['gender']}, {a['education']}, "
            f"model {a['model']}"
        )
        tl = agent_timeline(run, agent)
        cps = change_points(tl["opinion"].to_numpy())
        for _, r in tl.iterrows():
            flag = "  <- change point" if r["day"] in cps else ""
            fm = (
                ""
                if pd.isna(r["feed_mean"])
                else f" read {r['read_from']} (mean {r['feed_mean']:+.2f})"
            )
            typer.echo(f"\nday {r['day']:3d}  opinion {r['opinion']:+.2f}{fm}{flag}")
            if r["reasoning"]:
                typer.echo(f"  thinks: {r['reasoning']}")
            typer.echo(f"  posts:  {r['post']}")
            if r["error"]:
                typer.echo(f"  ERROR:  {r['error']}")
        return

    m = run.metrics
    typer.echo(f"{run.meta['config']['name']}: {len(run.agents)} agents, {run.days} days")
    _print_change(m.iloc[0].to_dict(), m.iloc[-1].to_dict())
    summary = model_summary(run, min_delta)
    typer.echo("\nPer model:")
    typer.echo(summary.round(3).to_string(index=False))
    shifts = opinion_shifts(run, min_delta)
    typer.echo(f"\n{len(shifts)} shifts of at least {min_delta}; largest {top}:")
    if len(shifts):
        big = shifts.reindex(shifts["delta"].abs().sort_values(ascending=False).index).head(top)
        cols = ["day", "agent", "model", "before", "after", "direction", "n_read", "reasoning"]
        typer.echo(big[cols].round(2).to_string(index=False))
    if csv:
        shifts.to_csv(run.path / "shifts.csv", index=False)
        summary.to_csv(run.path / "model_summary.csv", index=False)
        typer.echo(f"\nwrote shifts.csv and model_summary.csv to {run.path}")


@app.command(name="app")
def app_cmd(
    port: int = typer.Option(8765, help="port on 127.0.0.1"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="open it in your browser"),
):
    """Open the web app: configure runs, start them, and watch them live."""
    from disensa.app import serve

    serve(port=port, open_browser=open_browser)


@app.command()
def viz(
    run_dir: Path,
    open_browser: bool = typer.Option(True, "--open/--no-open", help="open it in your browser"),
):
    """Write <run>/viewer.html (the d3 viewer, data embedded) and open it."""
    import webbrowser

    from disensa.export import write_viewer

    out = write_viewer(run_dir)
    typer.echo(f"wrote {out}")
    if open_browser:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    app()
