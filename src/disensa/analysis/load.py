"""Finding and loading run folders: single runs, sweep folders, or a whole runs/ folder."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def find_run_dirs(paths) -> list[Path]:
    """Every run folder under the given paths: a run folder itself, the runs of a sweep
    folder, or both levels of a `runs/` folder. Sorted, without duplicates."""
    if isinstance(paths, str | Path):
        paths = [paths]
    found: set[Path] = set()
    for p in map(Path, paths):
        if (p / "meta.json").exists():
            found.add(p)
            continue
        if not p.is_dir():
            raise FileNotFoundError(p)
        for d in p.iterdir():
            if (d / "meta.json").exists():
                found.add(d)
            elif d.is_dir():
                found.update(x for x in d.iterdir() if (x / "meta.json").exists())
    return sorted(found)


def _status(meta: dict) -> str:
    if meta.get("status"):
        return meta["status"]
    if meta.get("cancelled"):
        return "cancelled"
    return "finished" if meta.get("finished") else "incomplete"


def flatten(meta: dict, path: Path) -> dict:
    """One row describing a run: where it is, its status, and its config fields."""
    cfg = meta["config"]
    dyn, net = cfg["dynamics"], cfg["network"]
    llm = dyn["type"] == "llm"
    rule = None if llm else dyn.get("rule")
    report = meta.get("report") or {}
    return {
        "path": str(path),
        "name": cfg["name"],
        "sweep": meta.get("sweep"),
        "status": _status(meta),
        "engine": dyn["type"],
        "rule": rule,
        "epsilon": dyn.get("epsilon") if rule == "bounded_confidence" else None,
        "mu": dyn.get("mu") if rule == "bounded_confidence" else None,
        "stubbornness": dyn.get("stubbornness") if rule == "friedkin_johnsen" else None,
        "models": ", ".join(sorted({m["name"] for m in dyn["models"]})) if llm else None,
        "statement": dyn.get("statement") if llm else None,
        "seed": cfg.get("seed"),
        "replicate": cfg.get("replicate", 0),
        "topology": net["topology"],
        "n": net["n"],
        "steps": cfg.get("steps"),
        "distribution": (cfg.get("opinions") or {}).get("distribution"),
        "failed_pct": report.get("failed_pct"),
        "seeding": meta.get("seeding", 1),
    }


def load_runs(paths, statuses: tuple[str, ...] | None = ("finished",)) -> pd.DataFrame:
    """A table of runs, one row each (see `flatten`). `statuses=None` keeps every run.

    The opinions, metrics, events and scores of a run are one `disensa.runs.load_run(path)`
    away; `path` is in the table.
    """
    rows = []
    for d in find_run_dirs(paths):
        try:
            meta = json.loads((d / "meta.json").read_text())
            row = flatten(meta, d)
        except (OSError, json.JSONDecodeError, KeyError):
            continue
        if statuses is None or row["status"] in statuses:
            rows.append(row)
    return pd.DataFrame(rows)
