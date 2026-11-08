"""Where sweeps stand, in one screen: the summary to send back after a job.

    python scripts/indaco/status.py runs/<sweep> [runs/<sweep> ...] [--flag-above 5]

Per sweep: runs finished / failed / not done, the share of failed replies over all calls
and by kind, the worst run, every run above --flag-above % failed replies (kept, and
checked in the analysis with and without them), why failed runs stopped, and the time per
finished run, with an estimate of what is left. Reads only sweep.json and each run's meta.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def sweep_status(root: Path, flag_above: float, parallel: int) -> str:
    doc = load(root / "sweep.json")
    if not doc:
        return f"{root}: no sweep.json (not a sweep folder, or nothing ran yet)\n"
    runs = doc["runs"]
    status = Counter(r["status"] for r in runs)
    calls = failed = 0
    kinds: Counter = Counter()
    seconds, worst, flagged, reasons = [], (0.0, ""), [], []
    for r in runs:
        if not r.get("folder"):
            continue
        meta = load(root / r["folder"] / "meta.json")
        rep = meta.get("report") or {}
        calls += rep.get("calls") or 0
        failed += rep.get("failed") or 0
        kinds.update(rep.get("failed_by_kind") or {})
        pct = rep.get("failed_pct") or 0.0
        if pct > worst[0]:
            worst = (pct, r["name"])
        if r["status"] == "finished":
            if rep.get("seconds"):
                seconds.append(rep["seconds"])
            if pct > flag_above:
                flagged.append(f"{r['name']} ({pct:g}%)")
        elif r["status"] == "failed":
            reasons.append(f"{r['name']}: {meta.get('reason', 'no reason recorded')}")
    total = len(runs)
    done = status.get("finished", 0)
    lines = [
        f"== {doc['name']}  ({root})",
        f"runs: {done}/{total} finished, {status.get('failed', 0)} failed, "
        f"{status.get('running', 0)} running or interrupted, {status.get('pending', 0)} pending",
    ]
    if calls:
        by_kind = ", ".join(f"{k} {v}" for k, v in kinds.most_common()) or "none"
        lines.append(
            f"replies: {calls:,}, failed {failed:,} ({100 * failed / calls:.2f}%): {by_kind}"
        )
        lines.append(f"worst run: {worst[1] or '—'} ({worst[0]:g}% failed)")
    lines.append(
        f"runs above {flag_above:g}% failed replies (kept; flagged for the analysis): "
        + (", ".join(flagged) if flagged else "none")
    )
    lines += [f"stopped: {x}" for x in reasons]
    if seconds:
        mean = sum(seconds) / len(seconds)
        left = (total - done) * mean / max(parallel, 1) / 3600
        lines.append(
            f"time: {mean / 60:.1f} min per finished run; about {left:.1f} h left "
            f"with {parallel} runs side by side"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("folders", nargs="+", type=Path)
    p.add_argument("--flag-above", type=float, default=5.0)
    p.add_argument("--parallel", type=int, default=4, help="runs side by side, for the estimate")
    args = p.parse_args()
    for f in args.folders:
        print(sweep_status(f, args.flag_above, args.parallel))


if __name__ == "__main__":
    main()
