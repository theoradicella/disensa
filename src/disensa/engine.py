"""The simulation loop. It is the same for numeric (ABM) and LLM agents:

    G     <- build the network                      (network.py)
    x(0)  <- draw initial opinions                  (network.py)
    for t = 1 .. steps:
        every agent i looks at its neighbours at t-1
        x_i(t) <- update(i)                         (dynamics/: a numeric rule or an LLM)
        record x(t) and the metrics                 (metrics.py, runs.py)

Updates are synchronous: every agent reads the state of step t-1 and all new opinions are
applied together, so the order in which agents are processed never matters.

The run's `status` in meta.json is running | finished | cancelled | failed. An LLM run stops
as `failed` when more than `dynamics.max_error_rate` of the replies of one step fail (step 0
included): a failed reply keeps the agent's opinion, so a run of mostly failed replies would
look like a run of stubborn agents. All finished steps are kept.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from disensa import OPINION_MAX, OPINION_MIN
from disensa.config import ExperimentConfig
from disensa.dynamics import make_dynamics
from disensa.metrics import all_metrics
from disensa.network import build_network, initial_opinions
from disensa.report import compact, failure_reason, summarize
from disensa.runs import RunWriter, update_meta

Progress = Callable[[int, dict], None]
OnReply = Callable[[int, int, str | None], None]  # (step, agent, error)


class RunFailed(RuntimeError):
    """Too many replies of one step failed. `history` holds the steps that finished."""

    def __init__(self, reason: str, history: list[dict], report: dict):
        super().__init__(reason)
        self.reason, self.history, self.report = reason, history, report


def step_metrics(graph, x: np.ndarray, labels: list[str], events: list[dict]) -> dict:
    m = all_metrics(graph, x)
    m["errors"] = sum(1 for e in events if e.get("error"))
    m["out_of_range"] = sum(1 for e in events if e.get("clipped"))
    m["prompt_tokens"] = sum(e.get("prompt_tokens") or 0 for e in events)
    m["completion_tokens"] = sum(e.get("completion_tokens") or 0 for e in events)
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
    single LLM reply (LLM agents only), for live progress inside a step. Raises RunFailed
    when the failure guard stops the run.
    """
    graph = build_network(cfg.network, seed=cfg.seed)
    x = initial_opinions(cfg.opinions, graph.number_of_nodes(), seed=cfg.seed)
    dynamics = make_dynamics(cfg, graph)
    if on_reply and hasattr(dynamics, "on_reply"):
        dynamics.on_reply = on_reply
    labels = dynamics.labels()
    n = len(labels)
    max_error_rate = getattr(cfg.dynamics, "max_error_rate", None)
    writer = RunWriter(run_dir) if run_dir else None
    t0 = time.time()
    seen: list[dict] = []  # compact events, for the report
    history: list[dict] = []

    def stats() -> dict:
        out = {"seconds": round(time.time() - t0, 1), "report": summarize(seen, time.time() - t0)}
        client = getattr(dynamics, "client", None)
        if client:
            out.update(llm_calls=client.calls, cache_hits=client.cache_hits)
        return out

    def record(t: int, x: np.ndarray, events: list[dict], started: float) -> None:
        m = step_metrics(graph, x, labels, events)
        m["seconds"] = round(time.time() - started, 3)
        seen.extend(compact(e) for e in events)
        if writer:
            writer.step(t, x, m, events)
            update_meta(run_dir, status="running", **stats())
        history.append({"step": t, **m})
        if progress:
            progress(t, m)
        if max_error_rate is not None and m["errors"] > max_error_rate * n:
            reason = failure_reason(t, events, n)
            if run_dir:
                update_meta(run_dir, status="failed", reason=reason, **stats())
            raise RunFailed(reason, history, stats()["report"])

    try:
        if writer:
            writer.start(graph, labels, x)
        started = time.time()
        record(0, x, await dynamics.start(x), started)
        for t in range(1, cfg.steps + 1):
            started = time.time()
            x, events = await dynamics.step(x, t)
            x = np.clip(x, OPINION_MIN, OPINION_MAX)  # safety net: agents clip already
            record(t, x, events, started)
    except asyncio.CancelledError:
        if run_dir:
            update_meta(run_dir, status="cancelled", cancelled=True, **stats())
        raise
    except RunFailed:
        raise
    except Exception as e:
        if run_dir:
            update_meta(run_dir, status="failed", reason=f"{type(e).__name__}: {e}", **stats())
        raise
    finally:
        if writer:
            writer.close()

    if run_dir:
        update_meta(run_dir, status="finished", finished=True, **stats())
    return history
