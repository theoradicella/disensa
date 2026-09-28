"""Run directories: config snapshot, git commit, and everything a run produces, on disk.

Layout of runs/<timestamp>_<name>_s<seed>/
  meta.json            config + git commit + summary
  agents.parquet       one row per agent: persona, model, initial opinion
  opinions.parquet     per (day, agent): self-reported belief, classifier score, opinion used
  metrics.parquet      one row per day: polarization, disagreement, neighbour correlation, ...
  events.jsonl         one line per (day, agent): what it read, its reasoning, post, memory, errors
  edges.jsonl          graph edges on day 0 and on every day the graph changed (rewiring)
  trajectory.parquet   (numeric baselines only) one row per (step, agent)
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from disensa.config import ExperimentConfig


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def new_run_dir(cfg: ExperimentConfig, suffix: str = "") -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = cfg.output_dir / f"{stamp}_{cfg.name}{suffix}_s{cfg.seed}"
    run_dir, i = base, 1
    while run_dir.exists():  # two runs in the same second
        run_dir = base.with_name(f"{base.name}-{i}")
        i += 1
    run_dir.mkdir(parents=True)
    meta = {"config": cfg.model_dump(mode="json"), "git_commit": _git_commit(), "started": stamp}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return run_dir


def update_meta(run_dir: Path, **fields) -> None:
    path = run_dir / "meta.json"
    meta = json.loads(path.read_text())
    meta.update(fields)
    path.write_text(json.dumps(meta, indent=2))


def save_trajectory(run_dir: Path, traj: np.ndarray) -> Path:
    """Long format: one row per (step, agent). Easy to query with DuckDB/pandas."""
    steps, n = traj.shape
    df = pd.DataFrame(
        {
            "step": np.repeat(np.arange(steps), n),
            "agent": np.tile(np.arange(n), steps),
            "opinion": traj.ravel(),
        }
    )
    path = run_dir / "trajectory.parquet"
    df.to_parquet(path, index=False)
    return path


def save_metrics(run_dir: Path, rows: list[dict]) -> Path:
    path = run_dir / "metrics.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


class RunWriter:
    """Streams an LLM simulation to disk as it runs, so a crash never loses finished days."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self._events = open(run_dir / "events.jsonl", "a", encoding="utf-8")
        self._edges = open(run_dir / "edges.jsonl", "a", encoding="utf-8")
        self.opinion_rows: list[dict] = []
        self.metric_rows: list[dict] = []

    def agents(self, rows: list[dict]) -> None:
        df = pd.DataFrame(rows).rename(columns={"agent_id": "agent"})
        # nested dicts (Big Five traits) → one boolean column per trait
        if "traits" in df:
            traits = pd.json_normalize(df.pop("traits")).add_prefix("trait_")
            df = pd.concat([df, traits], axis=1)
        df.to_parquet(self.run_dir / "agents.parquet", index=False)

    def event(self, record: dict) -> None:
        self._events.write(json.dumps(record, ensure_ascii=False) + "\n")

    def edges(self, day: int, edges: list[tuple[int, int]]) -> None:
        self._edges.write(json.dumps({"day": day, "edges": [list(e) for e in edges]}) + "\n")

    def opinions(self, rows: list[dict]) -> None:
        self.opinion_rows.extend(rows)

    def metrics(self, row: dict) -> None:
        self.metric_rows.append(row)

    def flush(self) -> None:
        self._events.flush()
        self._edges.flush()
        if self.opinion_rows:
            pd.DataFrame(self.opinion_rows).to_parquet(
                self.run_dir / "opinions.parquet", index=False
            )
        if self.metric_rows:
            pd.DataFrame(self.metric_rows).to_parquet(self.run_dir / "metrics.parquet", index=False)

    def close(self) -> None:
        self.flush()
        self._events.close()
        self._edges.close()
