"""Storing runs on disk: one folder per run, written while it runs, read back by the lab.

runs/<timestamp>_<name>_s<seed>[_r<replicate>]/
  meta.json          the full config, git commit, start time, status (running | finished |
                     cancelled | failed, with a `reason` when failed), timings, and the
                     run `report` (report.py), updated after every step
  agents.parquet     one row per agent: id, model (or numeric rule), degree, initial opinion
  edges.json         the network as a list of [u, v] pairs
  opinions.parquet   one row per (step, agent): the opinion
  metrics.parquet    one row per step: polarization, disagreement, neighbour correlation, ...
  events.jsonl       LLM runs only, one line per (step, agent): its interactions (author +
                     message), its reasoning, new message, memory, the exact prompt and raw
                     reply, any error and its kind, the belief before clipping, and what
                     the call cost (latency, tokens, cached)

Parquet for tables (fast, typed, opens in pandas/DuckDB), JSONL for text. Every step is
flushed, so a crash or a cancelled run keeps all finished steps.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from disensa import __version__
from disensa.config import ExperimentConfig
from disensa.seeding import SEEDING


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


# settings that do not change what a run computes: left out of the config hash
_NOT_HASHED = [
    ("name",),
    ("output_dir",),
    ("dynamics", "cache_dir"),
    ("dynamics", "max_concurrency"),
]


def config_hash(cfg: ExperimentConfig) -> str:
    """Hash of what a run computes: its config without the name, folders and concurrency.

    `disensa sweep` uses it to recognise a run that has already been done.
    """
    raw = cfg.model_dump(mode="json")
    for path in _NOT_HASHED:
        obj = raw
        for k in path[:-1]:
            obj = obj.get(k, {})
        obj.pop(path[-1], None)
    return hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()[:16]


def new_run_dir(cfg: ExperimentConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # only the folder name is slugged; the name itself is kept as typed
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", cfg.name).strip("_") or "run"
    rep = f"_r{cfg.replicate}" if cfg.replicate else ""
    base = cfg.output_dir / f"{stamp}_{slug}_s{cfg.seed}{rep}"
    run_dir, i = base, 1
    while run_dir.exists():  # two runs in the same second
        run_dir = base.with_name(f"{base.name}-{i}")
        i += 1
    run_dir.mkdir(parents=True)
    meta = {
        "config": cfg.model_dump(mode="json"),
        "git_commit": _git_commit(),
        "started": stamp,
        "status": "running",  # running | finished | cancelled | failed (engine.py)
        "seeding": SEEDING,  # which random-stream scheme made this run (seeding.py)
        "config_hash": config_hash(cfg),
        "disensa_version": __version__,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return run_dir


def load_meta(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "meta.json").read_text())


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
    def status(self) -> str:
        """running | finished | cancelled | failed; older runs only had flags."""
        if self.meta.get("status"):
            return self.meta["status"]
        if self.meta.get("cancelled"):
            return "cancelled"
        return "finished" if self.meta.get("finished") else "incomplete"

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
