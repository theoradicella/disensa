"""Command line: `agorasim run configs/quick.yaml`, `agorasim baseline ...`, `agorasim ping ...`."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import typer
from dotenv import load_dotenv

from agorasim.baselines import run_baseline
from agorasim.config import ExperimentConfig
from agorasim.metrics import all_metrics
from agorasim.networks import build_network, initial_opinions
from agorasim.runs import RunWriter, new_run_dir, save_metrics, save_trajectory, update_meta

app = typer.Typer(no_args_is_help=True, help="agorasim: LLM-agent opinion dynamics")
load_dotenv()  # API keys from .env, if present

METRIC_KEYS = ("polarization", "global_disagreement", "neighbor_correlation")


def _print_change(first: dict, last: dict) -> None:
    for k in METRIC_KEYS:
        typer.echo(f"  {k:22s} {first[k]:+.3f} -> {last[k]:+.3f}")


@app.command()
def run(config: Path, days: int = typer.Option(None, help="override simulation.days")):
    """Run an LLM-agent simulation from a YAML config."""
    from agorasim.engine import Simulation

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
    from agorasim.llm import LLMClient

    resp = asyncio.run(
        LLMClient().complete(model, [{"role": "user", "content": prompt}], use_cache=False)
    )
    typer.echo(f"[{resp.model}] {resp.text}")


if __name__ == "__main__":
    app()
