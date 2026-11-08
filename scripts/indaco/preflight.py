"""Checks a sweep file before (and right after) a cluster job starts its vLLM server.

    python scripts/indaco/preflight.py check  <sweep file>
        Validates the sweep and prints `MODEL=<Hugging Face id>` for the job script. Refuses
        a sweep that needs more than one model (one vLLM server serves one model), a model
        that is not `hosted_vllm/...`, or a statement that is still a placeholder.

    python scripts/indaco/preflight.py models <sweep file>...
        The Hugging Face ids the sweeps need, one per line (setup.sh downloads them).

    python scripts/indaco/preflight.py state  <sweep file> [--scorer <Hugging Face id>]
        What is on disk: `NAME=… TOTAL=… FINISHED=… FAILED=… SCORED=…` (run_all.sh uses it
        to submit only what is left). SCORED counts finished runs fully rated by the scorer.

    python scripts/indaco/preflight.py smoke  <sweep file> --api-base <url> --out <folder>
        Once the server is up: asks it which models it serves, then runs the sweep's first
        LLM run for 2 steps only (150 calls for 50 agents) into <folder>. Fails if the model
        is not served, if the run stops, or if more than --max-failed % of the replies could
        not be used. Prints one agent's reply so the job log shows what the model writes.
        A model whose chat template rejects the system prompt, a JSON mode that loops, a
        wrong model name: all of these fail here, in a minute, not after hours.

Exit status: 0 fine, 1 the check failed, 2 the sweep file is invalid.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.request
from pathlib import Path

from disensa import sweep as sw

PLACEHOLDERS = ("PLACEHOLDER", "SET BY THE SWEEP", "TODO")


def fail(message: str, code: int = 1):
    print(f"preflight: {message}", file=sys.stderr)
    sys.exit(code)


def llm_runs(path: str):
    try:
        spec, base = sw.load_sweep(path)
        runs = sw.expand(spec, base)
    except sw.SweepError as e:
        fail(str(e), 2)
    llm = [r for r in runs if r.config.dynamics.type == "llm"]
    if not llm:
        fail(f"{path} has no LLM runs: run it on the Mac (`uv run disensa sweep {path}`)")
    return spec, runs, llm


def model_of(path: str) -> str:
    """The one Hugging Face id a sweep needs, after every check on its LLM runs."""
    _, _, llm = llm_runs(path)
    names = sorted({m.name for r in llm for m in r.config.dynamics.models})
    if len(names) != 1:
        fail(f"{path} needs {len(names)} models ({', '.join(names)}); a job serves one")
    if not names[0].startswith("hosted_vllm/"):
        fail(f"{names[0]}: on the cluster a model must be hosted_vllm/<Hugging Face id>")
    for r in llm:
        statement = r.config.dynamics.statement
        if any(p in statement for p in PLACEHOLDERS):
            fail(f"{r.name}: the statement is a placeholder: {statement!r}")
    return names[0].removeprefix("hosted_vllm/")


def cmd_check(args) -> None:
    model = model_of(args.sweep)
    spec, runs, llm = llm_runs(args.sweep)
    calls = sum(r.llm_calls for r in llm)
    print(
        f"{spec.name}: {len(runs)} runs ({len(llm)} LLM, {calls:,} calls), model {model}",
        file=sys.stderr,
    )
    print(f"MODEL={model}")
    print(f"SWEEP_NAME={spec.name}")


def cmd_models(args) -> None:
    for m in sorted({model_of(p) for p in args.sweeps}):
        print(m)


def cmd_state(args) -> None:
    from disensa.scorer import is_scored

    try:
        spec, base = sw.load_sweep(args.sweep)
        runs = sw.expand(spec, base)
    except sw.SweepError as e:
        fail(str(e), 2)
    root = spec.output_dir / spec.name
    sw.refresh(runs, root)
    finished = [r for r in runs if r.status == "finished"]
    scored = 0
    if args.scorer:
        name = f"hosted_vllm/{args.scorer}"
        scored = sum(is_scored(root / r.folder, name) for r in finished)
    print(
        f"NAME={spec.name} TOTAL={len(runs)} FINISHED={len(finished)} "
        f"FAILED={sum(r.status == 'failed' for r in runs)} SCORED={scored}"
    )


def served(api_base: str) -> list[str]:
    with urllib.request.urlopen(f"{api_base.rstrip('/')}/models", timeout=10) as r:
        return [m.get("id") for m in json.loads(r.read()).get("data", [])]


def cmd_smoke(args) -> None:
    from disensa.engine import RunFailed, run
    from disensa.report import format_report
    from disensa.runs import load_run, new_run_dir

    model = model_of(args.sweep)
    try:
        ids = served(args.api_base)
    except OSError as e:
        fail(f"no answer from {args.api_base}/models: {e}")
    if model not in ids:
        fail(f"the server at {args.api_base} serves {ids}, not {model}")

    spec, base = sw.load_sweep(args.sweep)
    first = next(
        r
        for r in sw.expand(spec, base, Path(args.cache_dir), args.api_base)
        if r.config.dynamics.type == "llm"
    )
    cfg = first.config.model_copy(deep=True)
    cfg.steps = 2
    cfg.name = f"smoke_{spec.name}"
    cfg.output_dir = Path(args.out)
    run_dir = new_run_dir(cfg)
    print(f"smoke run: {first.name}, 2 steps -> {run_dir}", file=sys.stderr)
    try:
        asyncio.run(run(cfg, run_dir))
    except RunFailed as e:
        fail(f"the smoke run stopped: {e.reason}")
    r = load_run(run_dir)
    report = r.meta.get("report") or {}
    print(format_report(report), file=sys.stderr)
    sample = next((e for e in r.events if e["step"] == 1 and not e.get("error")), None)
    if sample:
        print(
            f"one reply (agent {sample['agent']}, step 1): belief {sample['belief_raw']}\n"
            f"  reasoning: {sample['reasoning'][:300]}\n  message:   {sample['message'][:300]}",
            file=sys.stderr,
        )
    errors = [e for e in r.events if e.get("error")]
    for e in errors[:3]:
        print(f"  failed reply ({e['error_kind']}): {e['error'][:200]}", file=sys.stderr)
    pct = report.get("failed_pct") or 0.0
    if pct > args.max_failed:
        fail(f"{pct:g}% of the smoke run's replies failed (limit {args.max_failed:g}%)")
    print(f"preflight: smoke run fine ({pct:g}% failed)", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("sweep")
    m = sub.add_parser("models")
    m.add_argument("sweeps", nargs="+")
    st = sub.add_parser("state")
    st.add_argument("sweep")
    st.add_argument("--scorer")
    s = sub.add_parser("smoke")
    s.add_argument("sweep")
    s.add_argument("--api-base", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--cache-dir", default=".cache/smoke")
    s.add_argument("--max-failed", type=float, default=5.0)
    args = p.parse_args()
    commands = {"check": cmd_check, "models": cmd_models, "state": cmd_state, "smoke": cmd_smoke}
    commands[args.cmd](args)


if __name__ == "__main__":
    main()
