"""Storing runs on disk: one folder per run, written while it runs, read back by the lab.

runs/<timestamp>_<name>_s<seed>/
  meta.json          the full config, git commit, start time, status, timings
  agents.parquet     one row per agent: id, model (or numeric rule), degree, initial opinion
  edges.json         the network as a list of [u, v] pairs
  opinions.parquet   one row per (step, agent): the opinion
  metrics.parquet    one row per step: polarization, disagreement, neighbour correlation, ...
  events.jsonl       LLM runs only, one line per (step, agent): its interactions (author +
                     message), its reasoning, new message, memory, the exact prompt and raw
                     reply, and any error

Parquet for tables (fast, typed, opens in pandas/DuckDB), JSONL for text. Every step is
flushed, so a crash or a cancelled run keeps all finished steps.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import networkx as nx
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


def new_run_dir(cfg: ExperimentConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = cfg.output_dir / f"{stamp}_{cfg.name}_s{cfg.seed}"
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


class RunWriter:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self._events = None  # opened on the first event: numeric runs have none
        self._opinions: list[dict] = []
        self._metrics: list[dict] = []

    def start(self, graph: nx.Graph, labels: list[str], x0: np.ndarray) -> None:
        agents = pd.DataFrame(
            {
                "agent": range(len(labels)),
                "model": labels,
                "degree": [graph.degree(i) for i in range(len(labels))],
                "initial_opinion": x0,
            }
        )
        agents.to_parquet(self.run_dir / "agents.parquet", index=False)
        edges = sorted([min(u, v), max(u, v)] for u, v in graph.edges())
        (self.run_dir / "edges.json").write_text(json.dumps(edges))

    def step(self, t: int, x: np.ndarray, metrics: dict, events: list[dict]) -> None:
        self._opinions += [{"step": t, "agent": i, "opinion": float(v)} for i, v in enumerate(x)]
        self._metrics.append({"step": t, **metrics})
        if events and self._events is None:
            self._events = open(self.run_dir / "events.jsonl", "a", encoding="utf-8")
        for e in events:
            self._events.write(json.dumps(e, ensure_ascii=False) + "\n")
        self.flush()

    def flush(self) -> None:
        if self._events:
            self._events.flush()
        pd.DataFrame(self._opinions).to_parquet(self.run_dir / "opinions.parquet", index=False)
        pd.DataFrame(self._metrics).to_parquet(self.run_dir / "metrics.parquet", index=False)

    def close(self) -> None:
        if self._events:
            self._events.close()


@dataclass
class Run:
    path: Path
    meta: dict
    agents: pd.DataFrame
    opinions: pd.DataFrame
    metrics: pd.DataFrame
    events: list[dict]
    edges: list[list[int]]

    @property
    def steps(self) -> int:
        return int(self.opinions["step"].max())

    def opinion_matrix(self) -> np.ndarray:
        """Array of shape (steps + 1, n_agents)."""
        return self.opinions.pivot(index="step", columns="agent", values="opinion").to_numpy()


def load_run(path: str | Path) -> Run:
    """Read a run folder (also works on a run that is still in progress)."""
    path = Path(path)
    events = []
    events_file = path / "events.jsonl"
    if events_file.exists():
        for line in open(events_file, encoding="utf-8"):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:  # a run in progress may end mid-line
                break
    return Run(
        path=path,
        meta=json.loads((path / "meta.json").read_text()),
        agents=pd.read_parquet(path / "agents.parquet"),
        opinions=pd.read_parquet(path / "opinions.parquet"),
        metrics=pd.read_parquet(path / "metrics.parquet"),
        events=events,
        edges=json.loads((path / "edges.json").read_text()),
    )
