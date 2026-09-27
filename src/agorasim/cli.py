"""Command line: `agorasim baseline configs/bc_small_world.yaml`."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from agorasim.baselines import run_baseline
from agorasim.config import ExperimentConfig
from agorasim.metrics import all_metrics
from agorasim.networks import build_network, initial_opinions
from agorasim.runs import new_run_dir, save_metrics, save_trajectory

app = typer.Typer(no_args_is_help=True, help="agorasim: LLM-agent opinion dynamics")


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

    first, last = rows[0], rows[-1]
    typer.echo(f"{cfg.baseline.model} on {cfg.network.topology} (N={cfg.network.n})")
    for k in ("polarization", "global_disagreement", "neighbor_correlation"):
        typer.echo(f"  {k:22s} {first[k]:+.3f} -> {last[k]:+.3f}")
    typer.echo(f"saved to {run_dir}")


@app.command()
def ping(model: str, prompt: str = "In one sentence: is remote work good for society?"):
    """Send one prompt to any model (checks keys / endpoints)."""
    from agorasim.llm import LLMClient

    resp = asyncio.run(LLMClient().complete(model, [{"role": "user", "content": prompt}]))
    typer.echo(f"[{resp.model}{' (cached)' if resp.cached else ''}] {resp.text}")


if __name__ == "__main__":
    app()
