"""Numeric baselines written in the same run format as LLM simulations.

That way DeGroot, bounded confidence, Deffuant and Friedkin–Johnsen runs open in the same
viewer and the same analysis tools as LLM runs: one "day" per model step, no posts.
"""

from __future__ import annotations

from pathlib import Path

from disensa.baselines import run_baseline
from disensa.config import ExperimentConfig
from disensa.metrics import all_metrics
from disensa.networks import build_network, initial_opinions
from disensa.personas import generate_personas
from disensa.runs import RunWriter, new_run_dir, update_meta


def run_numeric(cfg: ExperimentConfig, run_dir: Path | None = None, progress=None) -> Path:
    if cfg.baseline is None:
        raise ValueError("a numeric run needs a `baseline` section")
    g = build_network(cfg.network, seed=cfg.seed)
    n = g.number_of_nodes()
    x0 = initial_opinions(cfg.opinions, n, seed=cfg.seed)
    traj = run_baseline(g, x0, cfg.baseline, seed=cfg.seed)
    label = f"numeric/{cfg.baseline.model}"

    run_dir = run_dir or new_run_dir(cfg)
    w = RunWriter(run_dir)
    w.agents(
        [
            {**p.to_dict(), "model": label, "degree": g.degree(p.agent_id)}
            for p in generate_personas(x0, cfg.seed)
        ]
    )
    w.edges(0, sorted(g.edges()))
    for day, x in enumerate(traj):
        w.opinions(
            [
                {
                    "day": day,
                    "agent": i,
                    "model": label,
                    "belief": float(v),
                    "score": None,
                    "opinion": float(v),
                }
                for i, v in enumerate(x)
            ]
        )
        for i, v in enumerate(x):
            w.event(
                {
                    "day": day,
                    "type": "post",
                    "agent": i,
                    "model": label,
                    "feed": [],
                    "reasoning": "",
                    "belief": float(v),
                    "score": None,
                    "post": "",
                    "memory": "",
                    "prompt": "",
                    "raw": "",
                    "error": None,
                }
            )
        m = {"day": day, **all_metrics(g, x), "errors": 0, "rewired": 0}
        w.metrics(m)
        if progress:
            progress(day, m)
    w.close()
    update_meta(run_dir, finished=True, engine="numeric")
    return run_dir
