"""The run report: how far a run can be trusted, and what it cost.

A failed reply leaves the agent's opinion unchanged, so many failures look like stubborn
agents rather than a crash. Every run therefore records, per call (in `events.jsonl`), per
step (in `metrics.parquet`) and per run (the `report` object in `meta.json`):

  failures     how many replies could not be used, by kind and by model
  cost         tokens in and out, wall-clock time, seconds per model call
  out of range beliefs outside [-2, 2] (clipped, but counted: that is a finding)
  rounding     whole / half / other numbers among valid self-ratings (a finding too)

`summarize` works on the events of any run folder: older runs, which lack the newer fields,
get `None` ("not recorded") for what they did not log.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable

# Why a reply could not be used (the event's `error_kind`):
#   provider        the model server returned an error or could not be reached
#   invalid_json    no JSON object in the reply, even after asking the model to fix it
#   missing_belief  valid JSON without a `belief`
#   invalid_belief  a `belief` that is not a finite number (NaN, inf, true, "a lot", ...)
ERROR_KINDS = ("provider", "invalid_json", "missing_belief", "invalid_belief")

# the small fields of an event the report needs (the engine keeps only these in memory)
EVENT_FIELDS = (
    "step",
    "agent",
    "model",
    "error",
    "error_kind",
    "cached",
    "latency_s",
    "prompt_tokens",
    "completion_tokens",
    "belief_raw",
    "clipped",
)


def compact(event: dict) -> dict:
    return {k: event[k] for k in EVENT_FIELDS if k in event}


def rounding(v: float) -> str:
    """ "whole" (1, -2), "half" (0.5, -1.5) or "other" (0.7, 1.25)."""
    if abs(v - round(v)) < 1e-9:
        return "whole"
    if abs(2 * v - round(2 * v)) < 1e-9:
        return "half"
    return "other"


def _pct(a: int, b: int) -> float:
    return round(100 * a / b, 2) if b else 0.0


def summarize(events: Iterable[dict], seconds: float | None = None) -> dict:
    """The `report` object of a run, from its events (any subset of `EVENT_FIELDS`)."""
    events = list(events)
    calls = len(events)
    failed = [e for e in events if e.get("error")]
    by_kind = Counter(e.get("error_kind") or "not recorded" for e in failed)
    models: dict[str, dict] = {}
    for e in events:
        m = models.setdefault(e.get("model") or "?", {"calls": 0, "failed": 0})
        m["calls"] += 1
        m["failed"] += bool(e.get("error"))
    for m in models.values():
        m["pct"] = _pct(m["failed"], m["calls"])

    def total(key: str) -> int | None:
        vals = [e[key] for e in events if e.get(key) is not None]
        recorded = any(key in e for e in events)
        return int(sum(vals)) if recorded else None

    prompt, completion = total("prompt_tokens"), total("completion_tokens")
    recorded = "cached" in (events[0] if events else {})
    # seconds per call: calls that reached the model this time and got an answer
    lat = [
        e["latency_s"]
        for e in events
        if e.get("latency_s") is not None
        and not e.get("cached")
        and e.get("error_kind") != "provider"
    ]
    rated = [
        e["belief_raw"]
        for e in events
        if e.get("step", 0) >= 1 and not e.get("error") and e.get("belief_raw") is not None
    ]
    grains = Counter(rounding(v) for v in rated)
    has_beliefs = any("belief_raw" in e for e in events)
    return {
        "calls": calls,
        "cache_hits": sum(bool(e.get("cached")) for e in events) if recorded else None,
        "failed": len(failed),
        "failed_pct": _pct(len(failed), calls),
        "failed_by_kind": dict(sorted(by_kind.items())),
        "failed_by_model": dict(sorted(models.items())),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "prompt_tokens_mean": round(prompt / calls, 1) if prompt is not None and calls else None,
        "completion_tokens_mean": (
            round(completion / calls, 1) if completion is not None and calls else None
        ),
        "seconds": None if seconds is None else round(seconds, 1),
        "seconds_per_call": round(sum(lat) / len(lat), 2) if lat else None,
        "out_of_range": sum(bool(e.get("clipped")) for e in events) if has_beliefs else None,
        "rounding": (
            {k: grains.get(k, 0) for k in ("whole", "half", "other")} if has_beliefs else None
        ),
    }


def failure_reason(step: int, events: list[dict], n: int) -> str:
    """e.g. "step 3: 47/50 replies failed (provider: connection refused)"."""
    failed = [e for e in events if e.get("error")]
    kinds = Counter(e.get("error_kind") or "unknown" for e in failed)
    kind, _ = kinds.most_common(1)[0]
    example = next(e["error"] for e in failed if (e.get("error_kind") or "unknown") == kind)
    return f"step {step}: {len(failed)}/{n} replies failed ({kind}: {example[:160]})"


def _fmt(v, unit: str = "") -> str:
    if v is None:
        return "not recorded"
    if isinstance(v, float) and not math.isfinite(v):
        return "—"
    return f"{v:,}{unit}" if isinstance(v, int) else f"{v:g}{unit}"


def _per_call(r: dict) -> str:
    if r.get("seconds_per_call") is not None:
        return f"{r['seconds_per_call']:g} s per model call"
    if r.get("cache_hits") is not None and r["cache_hits"] == r.get("calls"):
        return "no model calls this time (all from the cache)"
    if r.get("cache_hits") is None:
        return "per model call not recorded"
    return "no model call answered"


def format_report(r: dict | None) -> str:
    """A few lines for the terminal (`disensa run`, `disensa sweep`)."""
    if not r:
        return "run report: not recorded"
    if not r.get("calls"):
        return f"run report: {_fmt(r.get('seconds'), ' s')}, no model calls"
    lines = [
        f"replies   {r['calls']:,} ({_fmt(r.get('cache_hits'))} from the cache)",
        f"failed    {r['failed']:,} ({r['failed_pct']:g}%)"
        + (
            "  " + ", ".join(f"{k} {v}" for k, v in r["failed_by_kind"].items())
            if r["failed"]
            else ""
        ),
    ]
    if r["failed"] and len(r["failed_by_model"]) > 1:
        lines += [
            f"          {m}: {s['failed']}/{s['calls']} ({s['pct']:g}%)"
            for m, s in r["failed_by_model"].items()
        ]
    lines += [
        f"tokens    {_fmt(r.get('prompt_tokens'))} in, {_fmt(r.get('completion_tokens'))} out"
        + (
            f"  ({r['prompt_tokens_mean']:g} / {r['completion_tokens_mean']:g} per reply)"
            if r.get("prompt_tokens_mean") is not None
            else ""
        ),
        f"time      {_fmt(r.get('seconds'), ' s')}, " + _per_call(r),
        f"range     {_fmt(r.get('out_of_range'))} beliefs outside [-2, 2] (clipped)",
    ]
    g = r.get("rounding", r.get("granularity"))  # "granularity": runs before the rename
    if g is not None and not sum(g.values()):
        lines.append("rounding  no valid self-ratings")
    elif g:
        n = sum(g.values()) or 1
        lines.append(
            "rounding  "
            + ", ".join(f"{k} {v} ({100 * v / n:.0f}%)" for k, v in g.items())
            + " (steps ≥ 1)"
        )
    return "run report\n" + "\n".join("  " + line for line in lines)
