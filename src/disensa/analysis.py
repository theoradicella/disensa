"""Individual-level inspection of a finished run.

Everything an agent read, thought and wrote is in the run folder; this module turns it into
tables you can reason about:

  load_run(path)             everything in one object
  agent_timeline(run, i)     day-by-day story of one agent: what it read, its reasoning, post
  opinion_shifts(run)        every notable change of opinion, with what the agent had just read
                             and whether it moved toward it (assimilation) or away (repulsion)
  model_summary(run)         the same numbers aggregated per LLM
  change_points(series)      days where an agent's opinion trajectory changes regime
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class RunData:
    path: Path
    meta: dict
    agents: pd.DataFrame  # one row per agent
    opinions: pd.DataFrame  # (day, agent) -> belief, score, opinion
    metrics: pd.DataFrame  # one row per day
    posts: pd.DataFrame  # (day, agent) -> feed, reasoning, post, memory, error, ...
    rewires: pd.DataFrame  # (day, agent, dropped, added)
    edge_snapshots: list[tuple[int, list[tuple[int, int]]]]

    @property
    def days(self) -> int:
        return int(self.opinions["day"].max())

    def opinion_matrix(self) -> np.ndarray:
        """Array of shape (days + 1, n_agents)."""
        return self.opinions.pivot(index="day", columns="agent", values="opinion").to_numpy()

    def graph_at(self, day: int) -> nx.Graph:
        """The network as it was at the end of `day` (after that day's rewiring)."""
        edges = next(e for d, e in reversed(self.edge_snapshots) if d <= day)
        g = nx.Graph()
        g.add_nodes_from(self.agents["agent"])
        g.add_edges_from(edges)
        return g


def load_run(path: str | Path) -> RunData:
    path = Path(path)
    events = []
    for line in open(path / "events.jsonl", encoding="utf-8"):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:  # a run still in progress may end mid-line
            break
    posts = pd.DataFrame([e for e in events if e.get("type", "post") == "post"])
    rewires = pd.DataFrame(
        [e for e in events if e.get("type") == "rewire"],
        columns=["day", "type", "agent", "dropped", "added"],
    )
    snapshots = []
    for line in open(path / "edges.jsonl", encoding="utf-8"):
        try:
            s = json.loads(line)
        except json.JSONDecodeError:
            break
        snapshots.append((s["day"], [tuple(e) for e in s["edges"]]))
    return RunData(
        path=path,
        meta=json.loads((path / "meta.json").read_text()),
        agents=pd.read_parquet(path / "agents.parquet"),
        opinions=pd.read_parquet(path / "opinions.parquet"),
        metrics=pd.read_parquet(path / "metrics.parquet"),
        posts=posts,
        rewires=rewires,
        edge_snapshots=sorted(snapshots, key=lambda s: s[0]),
    )


def _feed_authors(feed) -> list[int]:
    return [int(f["author"]) for f in (feed or [])]


def agent_timeline(run: RunData, agent: int) -> pd.DataFrame:
    """One row per day for `agent`, with the mean opinion (on the previous day) of what it read."""
    ops = run.opinion_matrix()
    p = run.posts[run.posts["agent"] == agent].sort_values("day")
    rows = []
    for _, r in p.iterrows():
        d = int(r["day"])
        authors = _feed_authors(r["feed"])
        feed_mean = float(np.mean(ops[d - 1, authors])) if authors and d > 0 else np.nan
        rows.append(
            {
                "day": d,
                "opinion": ops[d, agent],
                "belief": r["belief"],
                "score": r["score"],
                "read_from": authors,
                "feed_mean": feed_mean,
                "reasoning": r["reasoning"],
                "post": r["post"],
                "error": r["error"],
            }
        )
    return pd.DataFrame(rows)


def opinion_shifts(run: RunData, min_delta: float = 0.5) -> pd.DataFrame:
    """Every day-to-day change of at least `min_delta`, with its context.

    direction:
      toward  moved in the direction of the average post it read (assimilation)
      away    moved against it (repulsion / backfire)
      none    read nothing, or what it read matched its old opinion
    """
    ops = run.opinion_matrix()
    model = dict(zip(run.agents["agent"], run.agents["model"], strict=True))
    rows = []
    for _, r in run.posts[run.posts["day"] > 0].iterrows():
        d, i = int(r["day"]), int(r["agent"])
        before, after = ops[d - 1, i], ops[d, i]
        delta = after - before
        if abs(delta) < min_delta:
            continue
        authors = _feed_authors(r["feed"])
        feed_mean = float(np.mean(ops[d - 1, authors])) if authors else np.nan
        pull = feed_mean - before if authors else 0.0
        if not authors or abs(pull) < 1e-9:
            direction = "none"
        else:
            direction = "toward" if np.sign(delta) == np.sign(pull) else "away"
        rows.append(
            {
                "day": d,
                "agent": i,
                "model": model[i],
                "before": before,
                "after": after,
                "delta": delta,
                "n_read": len(authors),
                "feed_mean": feed_mean,
                "direction": direction,
                "read_from": authors,
                "reasoning": r["reasoning"],
            }
        )
    cols = ["day", "agent", "model", "before", "after", "delta", "n_read", "feed_mean"]
    cols += ["direction", "read_from", "reasoning"]
    return pd.DataFrame(rows, columns=cols)


def model_summary(run: RunData, min_delta: float = 0.5) -> pd.DataFrame:
    """Per-model behaviour: how much agents move, in which direction, and how often they fail."""
    ops = run.opinion_matrix()
    steps = np.abs(np.diff(ops, axis=0))  # (days, agents)
    shifts = opinion_shifts(run, min_delta)
    errors = run.posts.assign(err=run.posts["error"].notna()).groupby("model")["err"].mean()
    rows = []
    for m, group in run.agents.groupby("model"):
        idx = group["agent"].to_numpy()
        s = shifts[shifts["model"] == m]
        rows.append(
            {
                "model": m,
                "agents": len(idx),
                "initial_mean": float(ops[0, idx].mean()),
                "final_mean": float(ops[-1, idx].mean()),
                "mean_abs_daily_change": float(steps[:, idx].mean()) if len(steps) else 0.0,
                "shifts": len(s),
                "toward_share": float((s["direction"] == "toward").mean()) if len(s) else np.nan,
                "away_share": float((s["direction"] == "away").mean()) if len(s) else np.nan,
                "error_rate": float(errors.get(m, 0.0)),
            }
        )
    return pd.DataFrame(rows)


def change_points(series: np.ndarray, penalty: float = 1.0) -> list[int]:
    """Days where the trajectory changes regime.

    Uses PELT from `ruptures` when installed (`uv sync --extra analysis`); otherwise falls
    back to days whose absolute change exceeds `penalty`.
    """
    x = np.asarray(series, dtype=float)
    if len(x) < 3:
        return []
    try:
        import ruptures as rpt
    except ImportError:
        return [int(d) for d in np.flatnonzero(np.abs(np.diff(x)) >= penalty) + 1]
    breaks = rpt.Pelt(model="l2", min_size=1, jump=1).fit(x.reshape(-1, 1)).predict(pen=penalty)
    return [int(b) for b in breaks if b < len(x)]
