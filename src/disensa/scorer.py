"""A separate opinion scorer: rates the logged agent texts after a run (measurement only).

In a run, each LLM agent rates its own opinion (`belief` in its reply), and that self-rating
is the simulation state. Chuang et al. (2024, App. I and M) instead rate each agent's text
with a separate classifier, validated against blind human ratings. `disensa score` does the
same after a run: it rates every logged text (the `message`, or the `reasoning`) of every
agent at every step on the 5-point scale, with any model, and writes `scores.parquet` in the
run folder. The scores never feed back into the simulation; which of the two measures the
analysis uses is decided by the human validation (validation.py).

scores.parquet, one row per (step, agent, field, scorer_model):
  step, agent    which text
  field          "message" or "reasoning"
  scorer_model   the LiteLLM name of the scorer
  score          -2 … 2, or null: "unclear" (error null) or failed (error set)
  raw            the scorer's reply, unparsed
  error          why the reply could not be used, or null
Several scorers and fields can coexist; scoring again with the same scorer and field
replaces its rows.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd

from disensa.llm import LLMCallError, LLMClient
from disensa.llm.prompts import SCORER_SYSTEM, scorer_prompt

FIELDS = ("message", "reasoning")
SCALE = (-2, -1, 0, 1, 2)
COLUMNS = ["step", "agent", "field", "scorer_model", "score", "raw", "error"]


def run_dirs(path: str | Path) -> list[Path]:
    """A run folder, or every run folder inside a sweep folder."""
    path = Path(path)
    if (path / "meta.json").exists():
        return [path]
    return sorted(d for d in path.iterdir() if (d / "meta.json").exists())


def parse_score(obj: dict) -> tuple[int | None, str | None]:
    """(score, error). "unclear" gives (None, None)."""
    if "score" not in obj:
        return None, "no 'score' in reply"
    v = obj["score"]
    if isinstance(v, str):
        if v.strip().lower() == "unclear":
            return None, None
        try:
            v = float(v.strip())
        except ValueError:
            return None, f"score is not a number: {v!r}"[:200]
    if isinstance(v, bool) or not isinstance(v, int | float) or v not in SCALE:
        return None, f"score is not one of -2, -1, 0, 1, 2: {v!r}"[:200]
    return int(v), None


def agent_models(meta: dict) -> list[str]:
    dyn = meta["config"]["dynamics"]
    return [m["name"] for m in dyn.get("models", [])] if dyn["type"] == "llm" else []


async def score_run(
    run_dir: Path,
    model: str,
    field: str = "message",
    api_base: str | None = None,
    client: LLMClient | None = None,
) -> pd.DataFrame:
    """Score every text of one run and write scores.parquet. Returns this scorer's rows."""
    if field not in FIELDS:
        raise ValueError(f"field must be one of {FIELDS}")
    meta = json.loads((run_dir / "meta.json").read_text())
    statement = meta["config"]["dynamics"]["statement"]
    events = []
    for line in open(run_dir / "events.jsonl", encoding="utf-8"):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            break  # a run in progress may end mid-line
    client = client or LLMClient()

    async def one(e: dict) -> dict:
        row = {"step": e["step"], "agent": e["agent"], "field": field, "scorer_model": model}
        text = (e.get(field) or "").strip()
        if not text:
            return {**row, "score": None, "raw": "", "error": f"no {field}"}
        messages = [
            {"role": "system", "content": SCORER_SYSTEM},
            {"role": "user", "content": scorer_prompt(statement, text)},
        ]
        try:
            obj, raw, _ = await client.complete_json(
                model,
                messages,
                temperature=0.0,
                max_tokens=60,
                seed=0,
                think=False,
                api_base=api_base,
            )
        except LLMCallError as err:
            return {**row, "score": None, "raw": err.raw, "error": str(err)[:300]}
        score, error = parse_score(obj)
        return {**row, "score": score, "raw": raw, "error": error}

    rows = await asyncio.gather(*(one(e) for e in events))
    new = pd.DataFrame(rows, columns=COLUMNS).astype({"score": "Float64"})
    path = run_dir / "scores.parquet"
    if path.exists():
        old = pd.read_parquet(path)
        old = old[~((old["scorer_model"] == model) & (old["field"] == field))]
        new_all = pd.concat([old, new], ignore_index=True) if len(old) else new
    else:
        new_all = new
    new_all.sort_values(["scorer_model", "field", "step", "agent"]).to_parquet(path, index=False)
    return new


def load_scores(run_dir: str | Path) -> pd.DataFrame | None:
    path = Path(run_dir) / "scores.parquet"
    return pd.read_parquet(path) if path.exists() else None
