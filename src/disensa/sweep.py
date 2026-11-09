"""`disensa sweep`: many runs from one file, locally or as a SLURM job array.

A sweep file names a base config and one or more grids. Each grid is a cartesian product of
values for dotted config paths; the sweep is the union of the grids:

    name: topics
    base: base.yaml                    # a path (relative to the sweep file) or a mapping
    grids:
      - seed: [1, 2, 3]
        dynamics.statement: ["...", "..."]
      - seed: [1, 2, 3]                # paired numeric baselines on the same worlds
        dynamics:                      # a whole sub-object can be a value: it replaces
          - {type: numeric, rule: degroot}
          - {type: numeric, rule: bounded_confidence}
        dynamics.epsilon: [0.5, 1.0]   # deeper keys apply after it

Every combination is validated as a full ExperimentConfig before anything runs. Runs go to
`<output_dir>/<sweep name>/`, next to `sweep.json`: the resolved list of runs with their
config hash, status, folder and report summary.

Runs execute one after the other. A run whose config hash already has a finished folder is
skipped, so an interrupted sweep resumes where it stopped; anything else (crashed, cancelled,
failed) is run again from scratch, and the LLM disk cache replays the calls already made.
One failed run never stops the sweep. `--shard i/N` (1-based) runs every N-th run starting
at the i-th, for SLURM job arrays: the shards are disjoint and together cover the sweep.
"""

from __future__ import annotations

import copy
import itertools
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from disensa.config import ExperimentConfig
from disensa.runs import config_hash


class SweepFile(BaseModel):
    name: str = Field(min_length=1)
    base: str | dict
    grids: list[dict[str, list]] = Field(default_factory=lambda: [{}])
    output_dir: Path = Path("runs")

    @field_validator("name")
    @classmethod
    def _safe_name(cls, v: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", v) or v.startswith("."):
            raise ValueError("a sweep name may only use letters, digits, '_', '-' and '.'")
        return v

    @field_validator("grids")
    @classmethod
    def _non_empty_values(cls, grids: list[dict[str, list]]) -> list[dict[str, list]]:
        for i, g in enumerate(grids, 1):
            for key, values in g.items():
                if not values:
                    raise ValueError(f"grid {i}: '{key}' has no values")
        return grids


class SweepError(ValueError):
    """The sweep file, or one of its combinations, is not valid."""


@dataclass
class PlannedRun:
    index: int  # position in the resolved list, 0-based
    name: str
    grid: int  # 1-based
    values: dict[str, Any]  # the grid values of this combination, by dotted key
    config: ExperimentConfig
    hash: str
    llm_calls: int = 0  # estimated: n_agents × (steps + 1) for an LLM run
    status: str = "pending"
    folder: str | None = None
    report: dict | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "grid": self.grid,
            "values": self.values,
            "hash": self.hash,
            "status": self.status,
            "folder": self.folder,
            "report": self.report,
        }


# ---------------------------------------------------------------- expansion


def set_path(obj: dict, dotted: str, value) -> None:
    """Set obj[a][b][c] = value for "a.b.c", creating mappings on the way."""
    keys = dotted.split(".")
    for k in keys[:-1]:
        if not isinstance(obj.get(k), dict):
            obj[k] = {}
        obj = obj[k]
    obj[keys[-1]] = copy.deepcopy(value)


def combinations(grid: dict[str, list]) -> list[dict[str, Any]]:
    """The cartesian product of one grid, as {dotted key: value} dicts, in file order."""
    keys = list(grid)
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*grid.values())]


def apply(base: dict, values: dict[str, Any]) -> dict:
    """base with the values set; shallower keys first, so `dynamics` comes before `dynamics.x`."""
    raw = copy.deepcopy(base)
    for key in sorted(values, key=lambda k: k.count(".")):
        set_path(raw, key, values[key])
    return raw


def _label(value, position: int) -> str:
    """How a value appears in a run name: itself if short, else its 1-based position."""
    if isinstance(value, dict):
        parts = [str(v) for k, v in value.items() if k != "type" and not isinstance(v, dict | list)]
        text = "-".join(parts)
    else:
        text = str(value)
    if len(text) > 24 or not re.fullmatch(r"[A-Za-z0-9_.+-]+", text):
        return str(position)
    return text


def load_sweep(path: str | Path) -> tuple[SweepFile, dict]:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    try:
        sweep = SweepFile.model_validate(raw)
    except ValidationError as e:
        raise SweepError(f"{path}: {e}") from e
    if isinstance(sweep.base, str):
        base_path = (
            (path.parent / sweep.base) if not Path(sweep.base).is_absolute() else Path(sweep.base)
        )
        if not base_path.exists() and Path(sweep.base).exists():
            base_path = Path(sweep.base)  # relative to the working directory
        if not base_path.exists():
            raise SweepError(f"{path}: base config {sweep.base} not found")
        with open(base_path) as f:
            base = yaml.safe_load(f) or {}
    else:
        base = sweep.base
    return sweep, base


def expand(
    sweep: SweepFile, base: dict, cache_dir: Path | None = None, api_base: str | None = None
) -> list[PlannedRun]:
    """Every run of the sweep, validated. Raises SweepError listing every invalid combination.

    `cache_dir` and `api_base` override the file's values for every LLM run (`api_base` for
    every model): on a cluster each job has its own cache folder and its own server port.
    Neither is part of the config hash, so overriding them never makes a run look new.
    """
    out: list[PlannedRun] = []
    problems: list[str] = []
    seen: dict[str, str] = {}  # hash -> name: the same config twice is run once
    for g, grid in enumerate(sweep.grids, 1):
        varied = [k for k, v in grid.items() if len(v) > 1]
        for values in combinations(grid):
            label = "__".join(
                f"{k.rsplit('.', 1)[-1]}={_label(values[k], grid[k].index(values[k]) + 1)}"
                for k in varied
            )
            name = f"{sweep.name}__{label}" if label else sweep.name
            raw = apply(base, values)
            raw["name"] = name
            raw["output_dir"] = str(sweep.output_dir / sweep.name)
            dyn = raw.get("dynamics")
            if cache_dir is not None and isinstance(dyn, dict):
                dyn["cache_dir"] = str(cache_dir)
            if api_base is not None and isinstance(dyn, dict):
                for m in dyn.get("models") or []:
                    if isinstance(m, dict):
                        m["api_base"] = api_base
            try:
                cfg = ExperimentConfig.model_validate(raw)
            except ValidationError as e:
                shown = ", ".join(f"{k}={values[k]!r}"[:80] for k in values) or "(no values)"
                errs = "; ".join(
                    f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()
                )
                problems.append(f"grid {g}, {shown}: {errs}")
                continue
            h = config_hash(cfg)
            if h in seen:
                continue
            seen[h] = name
            calls = cfg.network.n * (cfg.steps + 1) if cfg.dynamics.type == "llm" else 0
            out.append(PlannedRun(len(out), name, g, values, cfg, h, calls))
    if problems:
        raise SweepError(
            f"{len(problems)} invalid combination(s), nothing was run:\n  " + "\n  ".join(problems)
        )
    names = [r.name for r in out]
    for r in out:  # two grids gave the same name to different configs: tell them apart
        if names.count(r.name) > 1:
            r.name = f"{r.name}__g{r.grid}"
            r.config.name = r.name
    return out


def parse_shard(text: str | None) -> tuple[int, int] | None:
    """ "2/4" -> (2, 4). Shards are 1-based: 1/N … N/N."""
    if not text:
        return None
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", text)
    if not m or not 1 <= int(m.group(1)) <= int(m.group(2)):
        raise SweepError(f"--shard must be i/N with 1 ≤ i ≤ N, got {text!r}")
    return int(m.group(1)), int(m.group(2))


def shard(runs: list[PlannedRun], part: tuple[int, int] | None) -> list[PlannedRun]:
    """Runs i-1, i-1+N, i-1+2N, … of the given list: disjoint shards that cover it."""
    if part is None:
        return runs
    i, n = part
    return [r for k, r in enumerate(runs) if k % n == i - 1]


def select(runs: list[PlannedRun], statements: list[str] | None) -> list[PlannedRun]:
    """Only the LLM runs on these statements (all runs if none are given).

    For running a sweep in parts, e.g. one topic type at a time: the runs keep their names
    and hashes, so a later run of the whole sweep skips the ones already finished.
    """
    if not statements:
        return runs
    unknown = set(statements) - {
        r.config.dynamics.statement for r in runs if r.config.dynamics.type == "llm"
    }
    if unknown:
        raise SweepError(f"no run of the sweep has the statement(s): {sorted(unknown)}")
    return [
        r
        for r in runs
        if r.config.dynamics.type == "llm" and r.config.dynamics.statement in statements
    ]


# ---------------------------------------------------------------- state on disk


def _meta_status(meta: dict) -> str:
    if meta.get("status"):
        return meta["status"]
    if meta.get("cancelled"):
        return "cancelled"
    return "finished" if meta.get("finished") else "incomplete"


def existing(root: Path) -> dict[str, tuple[str, str, dict]]:
    """hash -> (folder, status, report) of the newest run folder of each config in root.

    A finished folder always wins over a newer unfinished one.
    """
    out: dict[str, tuple[str, str, dict]] = {}
    if not root.exists():
        return out
    for d in sorted(p for p in root.iterdir() if (p / "meta.json").exists()):
        try:
            meta = json.loads((d / "meta.json").read_text())
            h = meta.get("config_hash") or config_hash(
                ExperimentConfig.model_validate(meta["config"])
            )
        except (OSError, json.JSONDecodeError, KeyError, ValidationError):
            continue
        status = _meta_status(meta)
        if out.get(h, ("", "", {}))[1] == "finished" and status != "finished":
            continue
        out[h] = (d.name, status, meta.get("report") or {})
    return out


def refresh(runs: list[PlannedRun], root: Path) -> None:
    """Fill each run's status, folder and report from the run folders on disk."""
    found = existing(root)
    for r in runs:
        if r.hash in found:
            r.folder, r.status, rep = found[r.hash]
            r.report = summary(rep)
        else:
            r.folder, r.status, r.report = None, "pending", None


def summary(report: dict | None) -> dict | None:
    if not report:
        return None
    keys = ("calls", "failed", "failed_pct", "seconds", "prompt_tokens", "completion_tokens")
    return {k: report.get(k) for k in keys}


def write_sweep_json(sweep: SweepFile, source: Path | None, runs: list[PlannedRun]) -> Path:
    """Rewrite sweep.json from the folders on disk (atomic: shards may write it in turn).

    The folders are the truth: every shard rewrites the whole list from them, so the last
    write always sees every run that has finished anywhere.
    """
    root = sweep.output_dir / sweep.name
    root.mkdir(parents=True, exist_ok=True)
    found = existing(root)
    entries = []
    for r in runs:
        folder, status, rep = found.get(r.hash, (None, "pending", None))
        entries.append({**r.to_dict(), "folder": folder, "status": status, "report": summary(rep)})
    doc = {"name": sweep.name, "file": str(source) if source else None, "runs": entries}
    path = root / "sweep.json"
    tmp = root / f".sweep.json.{os.getpid()}"
    tmp.write_text(json.dumps(doc, indent=2, allow_nan=False, default=str))
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------- running


async def run_sweep(
    sweep: SweepFile,
    runs: list[PlannedRun],
    todo: list[PlannedRun],
    source: Path | None = None,
    echo=print,
) -> list[PlannedRun]:
    """Run `todo` (a subset of `runs`, e.g. one shard) one after the other."""
    from disensa.checks import config_checks
    from disensa.engine import RunFailed, run
    from disensa.report import format_report
    from disensa.runs import new_run_dir, update_meta

    refresh(runs, sweep.output_dir / sweep.name)
    write_sweep_json(sweep, source, runs)
    for k, r in enumerate(todo, 1):
        head = f"[{k}/{len(todo)}] {r.name}"
        if r.status == "finished":
            echo(f"{head}: already finished ({r.folder}), skipped")
            r.status = "skipped"
            continue
        errors = [c for c in config_checks(r.config) if c.level == "error"]
        if errors:
            r.status, r.report = "failed", None
            echo(f"{head}: not run: {errors[0].message}")
            continue
        run_dir = new_run_dir(r.config)
        update_meta(run_dir, sweep=sweep.name)
        echo(f"{head} -> {run_dir.name}")
        try:
            await run(r.config, run_dir)
            r.status = "finished"
        except RunFailed as e:
            r.status = "failed"
            echo(f"  failed: {e.reason}")
        except Exception as e:  # noqa: BLE001 - one broken run never stops the sweep
            r.status = "failed"
            echo(f"  failed: {type(e).__name__}: {e}")
        meta = json.loads((run_dir / "meta.json").read_text())
        r.folder, r.report = run_dir.name, summary(meta.get("report"))
        if r.config.dynamics.type == "llm":
            echo("  " + format_report(meta.get("report")).replace("\n", "\n  "))
        write_sweep_json(sweep, source, runs)
    return todo


def table(runs: list[PlannedRun]) -> str:
    """run, status, failure %, seconds: one line each."""
    width = max([len(r.name) for r in runs] + [3])
    lines = [f"{'run':<{width}}  {'status':<10}  {'failed':>7}  {'seconds':>8}"]
    for r in runs:
        rep = r.report or {}
        pct = rep.get("failed_pct")
        sec = rep.get("seconds")
        lines.append(
            f"{r.name:<{width}}  {r.status:<10}  "
            f"{'—' if pct is None else f'{pct:g}%':>7}  {'—' if sec is None else f'{sec:g}':>8}"
        )
    return "\n".join(lines)
