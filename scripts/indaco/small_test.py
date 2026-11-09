"""Is the small test sane? Reads runs/small_test/ (written by scripts/indaco/small_test.sh).

    python scripts/indaco/small_test.py runs/small_test

Per run: failed replies, beliefs outside [-2, 2], mean and spread of the opinions at the
start and the end, how many agents changed, one agent's last message and, if the scorer
ran, how its ratings compare with the agents' own. Then:
  - the drift d = mean(end) - mean(start) of the two climate framings, split as in
    thesis/EXPERIMENT_PLAN.md: agreeing with whatever is stated a = (d_a + d_b) / 2, and
    leaning to the human-cause side c = (d_a - d_b) / 2. With and without interactions.
    These are information, not checks: how the opinions move is a result.
  - the checks below. Any FAIL means: send Theo this output and the logs/ folder.
"""

import json
import sys
from pathlib import Path

import pandas as pd

MAX_FAILED_PCT = 5  # replies that could not be used
MAX_CLIPPED_PCT = 5  # beliefs outside [-2, 2]
MAX_SCORER_FAILED_PCT = 5  # scorer replies that were not one of -2 … 2 or "unclear"

root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/small_test")
sweep = json.loads((root / "sweep.json").read_text())
checks = []
drift = {}  # (statement, interactions?) -> d


def short(text: str) -> str:
    return text if len(text) <= 40 else text[:37] + "..."


for r in sweep["runs"]:
    if not r.get("folder"):
        print(f"\n== {r['name']}: not run")
        checks.append((False, f"{r['name']} did not run"))
        continue
    run = root / r["folder"]
    meta = json.loads((run / "meta.json").read_text())
    dyn = meta["config"]["dynamics"]
    statement = dyn["statement"]
    talks = dyn["interactions"]["max"] > 0
    name = short(statement) + ("" if talks else " [no interactions]")
    print(f"\n== {statement}" + ("" if talks else "\n   NO INTERACTIONS (control)"))
    rep = meta.get("report") or {}
    op = pd.read_parquet(run / "opinions.parquet").pivot(index="step", columns="agent")["opinion"]
    first, last = op.iloc[0], op.iloc[-1]
    drift[(statement, talks)] = last.mean() - first.mean()
    moved = ((last - first).abs() > 0.25).mean()
    calls = rep.get("calls") or 0
    clipped_pct = 100 * (rep.get("out_of_range") or 0) / calls if calls else 0.0
    print(
        f"   status {meta['status']}, {len(op) - 1} steps, "
        f"failed replies {rep.get('failed_pct', '?')}% {rep.get('failed_by_kind') or ''}"
    )
    print(f"   beliefs outside [-2, 2]: {rep.get('out_of_range', '?')} ({clipped_pct:.1f}%)")
    print(
        f"   opinion mean {first.mean():+.2f} -> {last.mean():+.2f}, "
        f"sd {first.std():.2f} -> {last.std():.2f}, {moved:.0%} of agents moved"
    )
    events = [json.loads(x) for x in (run / "events.jsonl").read_text().splitlines()]
    msg = next((e["message"] for e in reversed(events) if e.get("message")), None)
    print(f"   a last message: {msg!r}")
    checks.append((meta["status"] == "finished", f"{name}: finished ({meta['status']})"))
    checks.append(((rep.get("failed_pct") or 0) <= MAX_FAILED_PCT, f"{name}: failed replies <= 5%"))
    checks.append((clipped_pct <= MAX_CLIPPED_PCT, f"{name}: beliefs outside [-2, 2] <= 5%"))

    scores_file = run / "scores.parquet"
    if scores_file.exists():
        sc = pd.read_parquet(scores_file)
        sc = sc[sc["field"] == "message"]
        bad = sc["error"].notna()
        unclear = sc["score"].isna() & ~bad
        own = pd.DataFrame(
            [
                (e["step"], e["agent"], e["belief_raw"])
                for e in events
                if e.get("belief_raw") is not None
            ],
            columns=["step", "agent", "belief"],
        )
        both = sc.dropna(subset=["score"]).merge(own, on=["step", "agent"])
        both["own"] = both["belief"].clip(-2, 2).round()
        exact = (both["score"] == both["own"]).mean() if len(both) else float("nan")
        within1 = ((both["score"] - both["own"]).abs() <= 1).mean() if len(both) else float("nan")
        print(
            f"   scorer: {len(sc)} messages, failed {bad.mean():.1%}, "
            f"unclear {unclear.mean():.1%}; same as the agent's own rounded belief "
            f"{exact:.0%}, within 1 point {within1:.0%}"
        )
        checks.append((100 * bad.mean() <= MAX_SCORER_FAILED_PCT, f"{name}: scorer failed <= 5%"))
    else:
        checks.append((False, f"{name}: no scores.parquet (the scorer did not run)"))

print("\n== climate, the two framings (information, not a check)")


def drift_of(start: str, talks: bool) -> float | None:
    return next((d for (s, t), d in drift.items() if t == talks and s.startswith(start)), None)


for talks, label in ((True, "with interactions"), (False, "no interactions")):
    d_a, d_b = drift_of("Human activity", talks), drift_of("The global warming", talks)
    if d_a is None or d_b is None:
        continue
    print(
        f"   {label}: drift {d_a:+.2f} (human cause) and {d_b:+.2f} (natural factors): "
        f"agreeing a = {(d_a + d_b) / 2:+.2f}, "
        f"toward the human-cause side c = {(d_a - d_b) / 2:+.2f}"
    )
for (s, t), d in drift.items():
    if t and (s, False) in drift:
        print(f"   {short(s)}: drift {d:+.2f} with interactions, {drift[(s, False)]:+.2f} without")

print()
for ok, text in checks:
    print(("ok    " if ok else "FAIL  ") + text)
print(
    "\nall good: send Theo this output; he decides when the experiment starts"
    if all(ok for ok, _ in checks)
    else "\nsomething is off: send Theo this output and logs/"
)
