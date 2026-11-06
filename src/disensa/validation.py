"""Human validation: does the self-rating or the scorer measure opinions better?

Several people rate a sample of agent messages blind, and both measures are compared with
them (the protocol of Chuang et al. 2024, App. M, applied to both measures).

  export   samples messages stratified by self-rated level (-2 … 2, as equal as possible)
           across the given runs, shuffles them and writes
             sample.csv        item_id, statement, message, rating (empty): for the raters
             key.csv           item_id -> run, step, agent, model, self_rating, scorer
                               ratings: never given to the raters
             INSTRUCTIONS.md   what the raters do
  report   reads every ratings_<rater>.csv (a copy of sample.csv with `rating` filled in)
           and writes report.json and report.md: inter-rater agreement (Krippendorff's
           alpha, ordinal), the majority rating of each item, and for the self-rating and
           each scorer: exact agreement with the majority, agreement within ±1, mean
           absolute difference and a confusion matrix.

Only messages from step 1 on whose reply was used are sampled: at step 0 the opinion is
given to the agent, not rated by it, and a failed reply repeats an older message.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = (-2, -1, 0, 1, 2)
LABELS = {
    -2: "strongly oppose",
    -1: "somewhat oppose",
    0: "neutral",
    1: "somewhat support",
    2: "strongly support",
}

INSTRUCTIONS = """# Rating instructions

Thank you for helping. You will rate {n} short messages. Each message was written by a
member of a community discussing one statement, shown next to it.

For each message, answer one question:

> **What is the author's position on this statement?**

Write one of these numbers in the `rating` column:

| rating | the author… |
|---:|---|
| -2 | strongly opposes the statement |
| -1 | somewhat opposes it |
| 0 | is neutral |
| 1 | somewhat supports it |
| 2 | strongly supports it |

If the message really does not show any position on the statement, write `unclear`.

- Rate only the text in front of you, each message on its own. Do not try to guess who
  wrote it or what they said elsewhere.
- Do not discuss the messages with the other raters until everyone has finished.
- There are no right answers and no time limit; about 20–30 seconds per message is typical.

Save the file as `ratings_<your name>.csv` (keep the other columns as they are) and send it
back.
"""


def _level(v: float) -> int:
    """The stance level of a self-rating, rounded as the agents' stance labels are."""
    return int(min(max(round(float(v)), -2), 2))


# ---------------------------------------------------------------- export


def collect(run_dirs: list[Path]) -> pd.DataFrame:
    """Every usable message of the given runs (or sweep folders), with its self-rating."""
    from disensa.runs import load_run
    from disensa.scorer import run_dirs as expand

    rows = []
    for d in [x for p in run_dirs for x in expand(p)]:
        r = load_run(d)
        if r.meta["config"]["dynamics"]["type"] != "llm":
            continue
        ops = r.opinions.set_index(["step", "agent"])["opinion"]
        scores = {}
        if r.scores is not None:
            s = r.scores[r.scores["field"] == "message"]
            for x in s.itertuples(index=False):
                scores[(x.step, x.agent, x.scorer_model)] = (
                    None if pd.isna(x.score) else int(x.score)
                )
        scorers = sorted({k[2] for k in scores})
        statement = r.meta["config"]["dynamics"]["statement"]
        for e in r.events:
            if e["step"] < 1 or e.get("error") or not (e.get("message") or "").strip():
                continue
            key = (e["step"], e["agent"])
            if key not in ops.index:
                continue
            row = {
                "run": str(d),
                "step": e["step"],
                "agent": e["agent"],
                "model": e["model"],
                "statement": statement,
                "message": e["message"].strip(),
                "self_rating": float(ops[key]),
            }
            for sc in scorers:
                row[f"score[{sc}]"] = scores.get((*key, sc))
            rows.append(row)
    return pd.DataFrame(rows)


def stratified_counts(available: dict[int, int], n: int) -> dict[int, int]:
    """How many items to take from each level: as equal as possible, never more than exist."""
    take = {lvl: 0 for lvl in available}
    left = min(n, sum(available.values()))
    while left:
        open_ = [lvl for lvl in sorted(available) if take[lvl] < available[lvl]]
        share = max(left // len(open_), 1)
        for lvl in open_:
            add = min(share, available[lvl] - take[lvl], left)
            take[lvl] += add
            left -= add
            if not left:
                break
    return take


def export(
    run_dirs: list[Path], out: Path, n: int = 100, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write sample.csv, key.csv and INSTRUCTIONS.md to `out`. Returns (sample, key)."""
    pool = collect(run_dirs)
    if pool.empty:
        raise ValueError("no usable LLM messages in the given runs")
    pool["level"] = pool["self_rating"].map(_level)
    rng = np.random.default_rng(seed)
    counts = stratified_counts(pool["level"].value_counts().to_dict(), n)
    parts = []
    for lvl, k in sorted(counts.items()):
        group = pool[pool["level"] == lvl]
        parts.append(group.iloc[rng.choice(len(group), size=k, replace=False)])
    picked = pd.concat(parts).iloc[rng.permutation(sum(counts.values()))].reset_index(drop=True)
    picked.insert(0, "item_id", [f"m{i:03d}" for i in range(1, len(picked) + 1)])
    sample = picked[["item_id", "statement", "message"]].assign(rating="")
    key = picked.drop(columns=["statement", "message", "level"])
    out.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out / "sample.csv", index=False)
    key.to_csv(out / "key.csv", index=False)
    (out / "INSTRUCTIONS.md").write_text(INSTRUCTIONS.format(n=len(sample)))
    return sample, key


# ---------------------------------------------------------------- agreement


def krippendorff_alpha_ordinal(data: list[list[float | None]]) -> float:
    """Krippendorff's alpha with the ordinal metric (Krippendorff 2011, "Computing
    Krippendorff's alpha-reliability", §C). `data[rater][unit]`, None/NaN for missing.

    Units with fewer than two ratings are not pairable and are left out.
    """
    units = list(zip(*data, strict=True))
    values = sorted({v for u in units for v in u if _ok(v)})
    if len(values) < 2:
        return float("nan")
    idx = {v: i for i, v in enumerate(values)}
    k = len(values)
    o = np.zeros((k, k))  # coincidence matrix
    for u in units:
        vals = [idx[v] for v in u if _ok(v)]
        m = len(vals)
        if m < 2:
            continue
        for a in range(m):
            for b in range(m):
                if a != b:
                    o[vals[a], vals[b]] += 1 / (m - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    delta = np.zeros((k, k))
    for c in range(k):
        for d in range(c + 1, k):
            delta[c, d] = delta[d, c] = (n_c[c : d + 1].sum() - (n_c[c] + n_c[d]) / 2) ** 2
    expected = (np.outer(n_c, n_c) * delta).sum()
    if expected == 0:
        return float("nan")
    return float(1 - (n - 1) * (o * delta).sum() / expected)


def _ok(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def parse_rating(v) -> float | None:
    """A rater's cell: -2 … 2, or None for empty / "unclear" / anything off the scale."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        f = float(str(v).strip())
    except ValueError:
        return None
    return f if f in LEVELS else None


def majority(ratings: list[float]) -> float | None:
    """The value more than half of the raters gave; else the median, halves toward 0."""
    if not ratings:
        return None
    value, count = Counter(ratings).most_common(1)[0]
    if count > len(ratings) / 2:
        return value
    med = float(np.median(ratings))
    return float(math.trunc(med)) if med != int(med) else med


def agreement(measure: list[float | None], human: list[float]) -> dict:
    """Exact and ±1 agreement, mean absolute difference and confusion matrix vs the majority.

    `measure` may hold continuous self-ratings: they are rounded to a level for agreement and
    the confusion matrix, and used as they are for the mean absolute difference. Items the
    measure has no value for count as disagreements and appear as "unclear" in the matrix.
    """
    n = len(human)
    exact = within = 0
    diffs = []
    confusion = {str(h): {str(m): 0 for m in [*LEVELS, "unclear"]} for h in LEVELS}
    for m, h in zip(measure, human, strict=True):
        if m is None or (isinstance(m, float) and math.isnan(m)):
            confusion[str(int(h))]["unclear"] += 1
            continue
        lvl = _level(m)
        exact += lvl == h
        within += abs(lvl - h) <= 1
        diffs.append(abs(float(m) - h))
        confusion[str(int(h))][str(lvl)] += 1
    return {
        "items": n,
        "rated": len(diffs),
        "exact_pct": round(100 * exact / n, 1) if n else None,
        "within_1_pct": round(100 * within / n, 1) if n else None,
        "mean_abs_diff": round(float(np.mean(diffs)), 3) if diffs else None,
        "confusion": confusion,  # rows: the raters' majority; columns: this measure
    }


def report(folder: Path) -> dict:
    """Read sample/key and every ratings_<rater>.csv in `folder`; write report.json/.md."""
    key = pd.read_csv(folder / "key.csv")
    files = sorted(folder.glob("ratings_*.csv"))
    if not files:
        raise ValueError(f"no ratings_<rater>.csv in {folder}")
    raters = {}
    for f in files:
        df = pd.read_csv(f, dtype={"rating": str})
        raters[f.stem.removeprefix("ratings_")] = {
            str(i): parse_rating(r) for i, r in zip(df["item_id"], df["rating"], strict=True)
        }
    items = list(key["item_id"].astype(str))
    data = [[raters[r].get(i) for i in items] for r in raters]
    alpha = krippendorff_alpha_ordinal(data)
    votes = {
        i: majority([raters[r][i] for r in raters if raters[r].get(i) is not None]) for i in items
    }
    used = [i for i in items if votes[i] is not None]
    human = [votes[i] for i in used]
    k = key.set_index(key["item_id"].astype(str))
    measures = {"self_rating": [float(k.loc[i, "self_rating"]) for i in used]}
    for col in [c for c in key.columns if c.startswith("score[")]:
        measures[col[6:-1]] = [
            None if pd.isna(k.loc[i, col]) else float(k.loc[i, col]) for i in used
        ]
    out = {
        "raters": list(raters),
        "items": len(items),
        "items_with_a_majority": len(used),
        "ratings_missing": sum(v is None for r in raters for v in raters[r].values()),
        "krippendorff_alpha_ordinal": None if math.isnan(alpha) else round(alpha, 3),
        "majority": votes,
        "measures": {name: agreement(vals, human) for name, vals in measures.items()},
    }
    (folder / "report.json").write_text(json.dumps(out, indent=2))
    (folder / "report.md").write_text(to_markdown(out))
    return out


def to_markdown(r: dict) -> str:
    a = r["krippendorff_alpha_ordinal"]
    lines = [
        "# Human validation report",
        "",
        f"- raters: {len(r['raters'])} ({', '.join(r['raters'])})",
        f"- items: {r['items']} ({r['items_with_a_majority']} with a majority rating; "
        f"{r['ratings_missing']} ratings empty or unclear)",
        f"- inter-rater agreement, Krippendorff's α (ordinal): {'—' if a is None else a}",
        "",
        "Agreement with the raters' majority (the median, halves toward 0, when there is no "
        "majority):",
        "",
        "| measure | exact | within ±1 | mean abs. difference | rated |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, m in r["measures"].items():
        lines.append(
            f"| {name} | {m['exact_pct']}% | {m['within_1_pct']}% | {m['mean_abs_diff']} | "
            f"{m['rated']}/{m['items']} |"
        )
    for name, m in r["measures"].items():
        cols = [*LEVELS, "unclear"]
        lines += [
            "",
            f"## Confusion matrix: {name}",
            "",
            "rows: raters' majority · columns: " + name,
            "",
            "| | " + " | ".join(str(c) for c in cols) + " |",
            "|---" * (len(cols) + 1) + "|",
        ]
        for h in LEVELS:
            row = m["confusion"][str(h)]
            lines.append(f"| **{h}** | " + " | ".join(str(row[str(c)]) for c in cols) + " |")
    return "\n".join(lines) + "\n"
