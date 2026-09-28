"""Turn a run into the data the viewer needs, and write a self-contained viewer file.

The app (static/app.html) fetches this payload from the local server. `disensa viz <run>`
instead writes <run>/viewer.html: the same page with d3 and the run's data embedded, so it
opens straight from disk, offline, with no server. `positions` is a seeded starting layout;
the page then runs a live force layout on top of it.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import networkx as nx
import numpy as np

from disensa.analysis import RunData, load_run

METRICS = ("polarization", "global_disagreement", "neighbor_correlation", "mean_opinion")


def _num(v) -> float | None:
    """JSON-safe number: NaN/inf become null."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, 4) if np.isfinite(f) else None


def layout(run: RunData, seed: int) -> list[list[float]]:
    """Force layout of the day-0 graph, normalised to [0, 1]²."""
    g = run.graph_at(0)
    # union of all edges ever present, so rewired partners end up near each other too
    for _, edges in run.edge_snapshots:
        g.add_edges_from(edges)
    pos = nx.spring_layout(g, seed=seed, iterations=200, k=1.6 / np.sqrt(max(len(g), 1)))
    xy = np.array([pos[i] for i in sorted(g.nodes)])
    xy = (xy - xy.min(axis=0)) / np.maximum(np.ptp(xy, axis=0), 1e-9)
    return [[round(float(x), 4), round(float(y), 4)] for x, y in xy]


def build_payload(run: RunData) -> dict:
    cfg = run.meta["config"]
    ops = run.opinion_matrix()
    agents = run.agents.sort_values("agent")
    trait_cols = [c for c in agents.columns if c.startswith("trait_")]
    models = sorted(agents["model"].unique().tolist())

    posts: list[list[dict | None]] = [[None] * len(agents) for _ in range(run.days + 1)]
    for r in run.posts.itertuples(index=False):
        if int(r.day) > run.days:  # a live run can log posts before that day's opinions
            continue
        posts[int(r.day)][int(r.agent)] = {
            "post": r.post or "",
            "reasoning": r.reasoning or "",
            "feed": [int(f["author"]) for f in (r.feed or [])],
            "belief": _num(r.belief),
            "score": _num(r.score),
            "error": r.error or None,
        }

    metric_rows = []
    for row in run.metrics.to_dict("records"):
        out = {"day": int(row["day"])}
        for k, v in row.items():
            if k in METRICS or k.startswith("mean_opinion[") or k in ("rewired", "errors"):
                out[k] = _num(v)
        metric_rows.append(out)

    return {
        "name": cfg["name"],
        "topic": ""
        if run.meta.get("engine") == "numeric"
        else cfg.get("topic", {}).get("statement", ""),
        "topology": cfg["network"]["topology"],
        "seed": cfg["seed"],
        "days": run.days,
        "opinion_source": cfg.get("simulation", {}).get("opinion_source", "self"),
        "models": models,
        "agents": [
            {
                "id": int(a["agent"]),
                "gender": a["gender"],
                "age": int(a["age"]),
                "education": a["education"],
                "model": a["model"],
                "traits": {c.removeprefix("trait_"): bool(a[c]) for c in trait_cols},
            }
            for a in agents.to_dict("records")
        ],
        "positions": layout(run, seed=int(cfg["seed"])),
        "opinions": [[_num(v) for v in day] for day in ops],
        "posts": posts,
        "edges": [{"day": d, "edges": [list(e) for e in edges]} for d, edges in run.edge_snapshots],
        "metrics": metric_rows,
    }


def _embed(payload: dict) -> str:
    """JSON that is safe inside a <script> tag."""
    s = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return s.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def write_viewer(run_dir: str | Path) -> Path:
    run = load_run(run_dir)
    static = resources.files("disensa").joinpath("static")
    template = static.joinpath("app.html").read_text(encoding="utf-8")
    d3_js = static.joinpath("d3.min.js").read_text(encoding="utf-8")
    html = template.replace('<script src="/d3.min.js"></script>', f"<script>{d3_js}</script>", 1)
    html = html.replace("/*__DISENSA_DATA__*/null", _embed(build_payload(run)), 1)
    out = Path(run_dir) / "viewer.html"
    out.write_text(html, encoding="utf-8")
    return out
