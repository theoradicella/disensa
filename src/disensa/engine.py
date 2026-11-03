"""The simulation loop. It is the same for numeric (ABM) and LLM agents:

    G     <- build the network                      (network.py)
    x(0)  <- draw initial opinions                  (network.py)
    for t = 1 .. steps:
        every agent i looks at its neighbours at t-1
        x_i(t) <- update(i)                         (dynamics/: a numeric rule or an LLM)
        record x(t) and the metrics                 (metrics.py, runs.py)

Updates are synchronous: every agent reads the state of step t-1 and all new opinions are
applied together, so the order in which agents are processed never matters.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from disensa import OPINION_MAX, OPINION_MIN
from disensa.config import ExperimentConfig
from disensa.dynamics import make_dynamics
from disensa.metrics import all_metrics
from disensa.network import build_network, initial_opinions
from disensa.runs import RunWriter, update_meta

Progress = Callable[[int, dict], None]
OnReply = Callable[[int, int, str | None], None]  # (step, agent, error)


def step_metrics(graph, x: np.ndarray, labels: list[str], events: list[dict]) -> dict:
    m = all_metrics(graph, x)
    m["errors"] = sum(1 for e in events if e.get("error"))
    models = sorted(set(labels))
    if len(models) > 1:  # mixed models: mean opinion of each group
        for name in models:
            mask = np.array([lab == name for lab in labels])
            m[f"mean_opinion[{name}]"] = float(x[mask].mean())
    return m


async def run(
    cfg: ExperimentConfig,
    run_dir: Path | None = None,
    progress: Progress | None = None,
    on_reply: OnReply | None = None,
) -> list[dict]:
    """Run one experiment. Writes to `run_dir` if given; returns the metrics of every step.

    `progress(t, metrics)` is called after every step; `on_reply(t, agent, error)` after every
    single LLM reply (LLM agents only), for live progress inside a step.
    """
    graph = build_network(cfg.network, seed=cfg.seed)
    x = initial_opinions(cfg.opinions, graph.number_of_nodes(), seed=cfg.seed)
    dynamics = make_dynamics(cfg, graph)
    if on_reply and hasattr(dynamics, "on_reply"):
        dynamics.on_reply = on_reply
    labels = dynamics.labels()
    writer = RunWriter(run_dir) if run_dir else None
    t0 = time.time()

    def record(t: int, x: np.ndarray, events: list[dict]) -> dict:
        m = step_metrics(graph, x, labels, events)
        if writer:
            writer.step(t, x, m, events)
        if progress:
            progress(t, m)
        return {"step": t, **m}

    try:
        if writer:
            writer.start(graph, labels, x)
        history = [record(0, x, await dynamics.start(x))]
        for t in range(1, cfg.steps + 1):
            x, events = await dynamics.step(x, t)
            x = np.clip(x, OPINION_MIN, OPINION_MAX)
            history.append(record(t, x, events))
    finally:
        if writer:
            writer.close()

    if run_dir:
        stats = {"finished": True, "seconds": round(time.time() - t0, 1)}
        client = getattr(dynamics, "client", None)
        if client:
            stats.update(llm_calls=client.calls, cache_hits=client.cache_hits)
        update_meta(run_dir, **stats)
    return history
