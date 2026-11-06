"""Turn a run folder into the JSON the lab's viewer draws."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from disensa.report import summarize
from disensa.runs import Run


def _num(v) -> float | None:
    """JSON-safe number: NaN/inf become null."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, 4) if np.isfinite(f) else None


def layout(n: int, edges: list[list[int]], seed: int) -> list[list[float]]:
    """Seeded force layout of the network, normalised to [0, 1]²."""
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    pos = nx.spring_layout(g, seed=seed, iterations=200, k=1.6 / np.sqrt(max(n, 1)))
    xy = np.array([pos[i] for i in range(n)])
    xy = (xy - xy.min(axis=0)) / np.maximum(np.ptp(xy, axis=0), 1e-9)
    return [[round(float(x), 4), round(float(y), 4)] for x, y in xy]


def build_payload(run: Run) -> dict:
    cfg = run.meta["config"]
    agents = run.agents.sort_values("agent")
    n = len(agents)
    ops = run.opinion_matrix()
    steps = run.steps

    # what each agent did at each step: its interactions (author ids), reasoning and message
    agent_steps: list[list[dict | None]] = [[None] * n for _ in range(steps + 1)]
    for e in run.events:
        if e["step"] <= steps:  # a live run can log events before that step's opinions
            agent_steps[e["step"]][e["agent"]] = {
                "message": e.get("message") or "",
                "reasoning": e.get("reasoning") or "",
                "interactions": [int(it["author"]) for it in e.get("interactions") or []],
                "error": e.get("error") or None,
            }

    # scorer ratings (scorer.py), if the run was scored: per step and agent, "<scorer> · <field>"
    scores = run.scores
    if scores is not None and len(scores):
        for r in scores.itertuples(index=False):
            entry = agent_steps[r.step][r.agent] if r.step <= steps else None
            if entry is not None:
                label = f"{r.scorer_model} · {r.field}"
                value = None if pd.isna(r.score) else int(r.score)
                entry.setdefault("scores", {})[label] = (
                    value if value is not None else ("failed" if r.error else "unclear")
                )

    dyn = cfg["dynamics"]
    # runs from before 0.3 have no report: rebuild what their events allow (the rest is None)
    report = run.meta.get("report")
    if report is None and dyn["type"] == "llm":
        report = summarize(run.events, run.meta.get("seconds"))
    return {
        "name": cfg["name"],
        "config": cfg,
        "topic": dyn.get("statement", "") if dyn["type"] == "llm" else "",
        "topology": cfg["network"]["topology"],
        "seed": cfg["seed"],
        "models": sorted(agents["model"].unique().tolist()),
        "agents": [
            {"id": int(a), "model": m}
            for a, m in zip(agents["agent"], agents["model"], strict=True)
        ],
        "positions": layout(n, run.edges, seed=int(cfg["seed"])),
        "opinions": [[_num(v) for v in row] for row in ops],
        "agent_steps": agent_steps,
        "edges": run.edges,
        "metrics": [{k: _num(v) for k, v in row.items()} for row in run.metrics.to_dict("records")],
        "report": _clean(report),
        "reason": run.meta.get("reason"),
    }


def _clean(obj):
    """Make a report JSON-safe (no NaN)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, float):
        return _num(obj)
    return obj
