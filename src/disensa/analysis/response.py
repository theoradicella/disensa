"""Micro response curve: how much an agent moves against how far the opinions it heard were.

For every LLM agent i at every step t ≥ 1 with at least one interaction and a reply that was
used, with the authors j it heard at t:

    d = mean_j x_j(t−1) − x_i(t−1)      how far the opinions it heard were from its own
    Δ = x_i(t) − x_i(t−1)               how much it moved

Binned by d, the curve is the mean Δ and the share of agents that moved, P(Δ ≠ 0). Each
numeric rule predicts a shape for the same inputs (the same agent, the same authors, the
same opinions), computed here as if the agent applied the rule to the authors it heard:

    DeGroot              Δ = k/(k+1) · d                         a straight line
    bounded confidence   Δ = μ · mean_{j ∈ heard ∪ {i}, |x_j − x_i| ≤ ε} (x_j − x_i)
                         a straight line that drops to zero beyond ε
    Friedkin–Johnsen     Δ = (λ s_i + Σ_j x_j) / (λ + k) − x_i   also pulled back to s_i = x_i(0)

`measure` chooses the opinions: "self" (the self-rating, the simulation state) or a scorer
model's name, to use its ratings of the messages (scorer.py). With a scorer, x_j(t−1) is
the rating of the very message i heard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from disensa.config import NumericDynamics

BINS = np.arange(-4.0, 4.01, 0.5)


def _opinions(run, measure: str) -> dict[tuple[int, int], float]:
    if measure == "self":
        ops = run.opinions
        return {
            (s, a): float(v)
            for s, a, v in zip(ops["step"], ops["agent"], ops["opinion"], strict=True)
        }
    s = run.scores
    if s is None:
        raise ValueError(f"{run.path} has no scores.parquet: run `disensa score` first")
    s = s[(s["scorer_model"] == measure) & (s["field"] == "message") & s["score"].notna()]
    if s.empty:
        raise ValueError(f"{run.path} has no message scores by {measure}")
    return {
        (int(a), int(b)): float(c)
        for a, b, c in zip(s["step"], s["agent"], s["score"], strict=True)
    }


def response_observations(run, measure: str = "self") -> pd.DataFrame:
    """One row per (agent, step): d, Δ, and what it heard (for the numeric predictions)."""
    x = _opinions(run, measure)
    x0 = run.opinion_matrix()[0]
    rows = []
    for e in run.events:
        t, i = e["step"], e["agent"]
        if t < 1 or e.get("error") or not e.get("interactions"):
            continue
        authors = [int(it["author"]) for it in e["interactions"]]
        own_before, own_after = x.get((t - 1, i)), x.get((t, i))
        heard = [x.get((t - 1, j)) for j in authors]
        if own_before is None or own_after is None or any(h is None for h in heard):
            continue  # a scorer may have found a text unclear
        rows.append(
            {
                "step": t,
                "agent": i,
                "model": e.get("model"),
                "x_before": own_before,
                "heard": heard,
                "s": float(x0[i]),
                "d": float(np.mean(heard) - own_before),
                "delta": own_after - own_before,
            }
        )
    return pd.DataFrame(rows)


def predict(obs: pd.DataFrame, dyn: NumericDynamics) -> np.ndarray:
    """Δ a numeric rule predicts for each observation, applied to the authors heard."""
    out = []
    for xi, heard, s in zip(obs["x_before"], obs["heard"], obs["s"], strict=True):
        h = np.asarray(heard, dtype=float)
        k = len(h)
        if dyn.rule == "degroot":
            new = (xi + h.sum()) / (k + 1)
        elif dyn.rule == "friedkin_johnsen":
            new = (dyn.stubbornness * s + h.sum()) / (dyn.stubbornness + k)
        else:
            close = np.append(h[np.abs(h - xi) <= dyn.epsilon], xi)
            new = xi + dyn.mu * (close - xi).mean()
        out.append(float(new - xi))
    return np.array(out)


def binned(d: np.ndarray, delta: np.ndarray, bins=BINS) -> pd.DataFrame:
    """Per bin of d: n, mean Δ, its standard error, and P(Δ ≠ 0)."""
    idx = np.digitize(d, bins) - 1
    rows = []
    for b in range(len(bins) - 1):
        m = idx == b
        k = int(m.sum())
        vals = delta[m]
        rows.append(
            {
                "d_low": float(bins[b]),
                "d_high": float(bins[b + 1]),
                "d_mid": float((bins[b] + bins[b + 1]) / 2),
                "n": k,
                "mean_delta": float(vals.mean()) if k else np.nan,
                "se_delta": float(vals.std(ddof=1) / np.sqrt(k)) if k > 1 else np.nan,
                "p_move": float((np.abs(vals) > 1e-9).mean()) if k else np.nan,
            }
        )
    return pd.DataFrame(rows)


def response_curve(
    runs, measure: str = "self", rules: list[NumericDynamics] | None = None, bins=BINS
) -> pd.DataFrame:
    """The binned curve over one or more runs, with each rule's prediction as extra columns
    (`pred[<rule>]` mean Δ and `pmove[<rule>]`)."""
    runs = runs if isinstance(runs, list | tuple) else [runs]
    obs = pd.concat([response_observations(r, measure) for r in runs], ignore_index=True)
    if obs.empty:
        return binned(np.array([]), np.array([]), bins)
    d, delta = obs["d"].to_numpy(), obs["delta"].to_numpy()
    out = binned(d, delta, bins)
    for dyn in rules or []:
        name = dyn.rule
        pred = binned(d, predict(obs, dyn), bins)
        out[f"pred[{name}]"] = pred["mean_delta"]
        out[f"pmove[{name}]"] = pred["p_move"]
    return out
