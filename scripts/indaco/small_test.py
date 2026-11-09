"""Is the small test sane? Reads runs/small_test/ (written by scripts/indaco/small_test.sh).

    python scripts/indaco/small_test.py runs/small_test

Per run: failed replies, mean and spread of the opinions at the start and the end, how many
agents changed, and one agent's last message. Then a few checks; any FAIL means: send Theo
this output and the logs/ folder.
"""

import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/small_test")
sweep = json.loads((root / "sweep.json").read_text())
checks = []
end_mean = {}

for r in sweep["runs"]:
    if not r.get("folder"):
        print(f"\n== {r['name']}: not run")
        checks.append((False, f"{r['name']} did not run"))
        continue
    run = root / r["folder"]
    meta = json.loads((run / "meta.json").read_text())
    statement = meta["config"]["dynamics"]["statement"]
    name = statement if len(statement) <= 40 else statement[:37] + "..."
    print(f"\n== {statement}")
    rep = meta.get("report") or {}
    op = pd.read_parquet(run / "opinions.parquet").pivot(index="step", columns="agent")["opinion"]
    first, last = op.iloc[0], op.iloc[-1]
    end_mean[statement] = last.mean()
    moved = ((last - first).abs() > 0.25).mean()
    print(
        f"   status {meta['status']}, {len(op) - 1} steps, "
        f"failed replies {rep.get('failed_pct', '?')}% {rep.get('failed_by_kind') or ''}"
    )
    print(
        f"   opinion mean {first.mean():+.2f} -> {last.mean():+.2f}, "
        f"sd {first.std():.2f} -> {last.std():.2f}, {moved:.0%} of agents moved"
    )
    events = [json.loads(x) for x in (run / "events.jsonl").read_text().splitlines()]
    msg = next((e["message"] for e in reversed(events) if e.get("message")), None)
    print(f"   a last message: {msg!r}")
    checks.append((meta["status"] == "finished", f"{name}: finished ({meta['status']})"))
    checks.append(((rep.get("failed_pct") or 0) <= 5, f"{name}: failed replies <= 5%"))
    checks.append((moved > 0.1, f"{name}: opinions changed"))

pro = next((m for s, m in end_mean.items() if s.startswith("Human activity")), None)
con = next((m for s, m in end_mean.items() if s.startswith("The global warming")), None)
if pro is not None and con is not None:
    checks.append(
        (
            pro > 0 > con,
            f"climate: the two framings end on opposite sides ({pro:+.2f} vs {con:+.2f})",
        )
    )

print()
for ok, text in checks:
    print(("ok    " if ok else "FAIL  ") + text)
print(
    "\nall good: the full experiment can start (scripts/indaco/run_all.sh)"
    if all(ok for ok, _ in checks)
    else "\nsomething is off: send Theo this output and logs/"
)
