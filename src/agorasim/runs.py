"""Run directories: config snapshot, git commit, metrics and trajectories on disk."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from agorasim.config import ExperimentConfig


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def new_run_dir(cfg: ExperimentConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = cfg.output_dir / f"{stamp}_{cfg.name}_s{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    meta = {"config": cfg.model_dump(mode="json"), "git_commit": _git_commit(), "started": stamp}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return run_dir


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
