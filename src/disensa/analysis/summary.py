"""Per-run summaries. Every function takes plain arrays, so it works on any run or simulation.

clusters          sort the final opinions and split wherever two neighbours are more than
                  δ apart. Default δ = 0.5: LLM beliefs move in steps of 0.5 (whole and
                  half numbers), so two clusters must be more than one half-step apart to
                  count as two. Numeric clusters are point masses: pass a small δ (e.g.
                  0.05) to count them exactly, as the bounded-confidence figure does.
convergence_step  the first step after which no opinion changes by `tol` or more
drift             final mean − initial mean
moved             at each step, the share of agents whose opinion changed
"""

from __future__ import annotations

import numpy as np

from disensa.metrics import polarization

DELTA = 0.5
TOL = 1e-3


def clusters(x, delta: float = DELTA) -> int:
    """Number of groups of opinions separated by gaps larger than `delta`."""
    x = np.sort(np.asarray(x, dtype=float))
    if len(x) == 0:
        return 0
    return int(1 + np.sum(np.diff(x) > delta))


def convergence_step(X, tol: float = TOL) -> int | None:
    """First step t such that max_i |x_i(s) − x_i(s−1)| < tol for every s > t.

    `X` has shape (steps + 1, agents). 0 means nobody ever moved by `tol`; None means the
    last step still moved (not converged within the run).
    """
    X = np.asarray(X, dtype=float)
    change = np.abs(np.diff(X, axis=0)).max(axis=1) if len(X) > 1 else np.array([])
    big = np.nonzero(change >= tol)[0]
    if len(big) == 0:
        return 0
    last = int(big[-1]) + 1  # change[k] is the move from step k to step k + 1
    return None if last == len(X) - 1 else last


def moved_fraction(X, tol: float = 1e-9) -> np.ndarray:
    """For steps 1..T, the share of agents whose opinion changed."""
    X = np.asarray(X, dtype=float)
    return (np.abs(np.diff(X, axis=0)) > tol).mean(axis=1)


def wasserstein(a, b) -> float:
    """1-D Wasserstein distance (earth mover's) between two samples: ∫ |F_a − F_b|."""
    a, b = np.sort(np.asarray(a, dtype=float)), np.sort(np.asarray(b, dtype=float))
    grid = np.sort(np.concatenate([a, b]))
    if len(grid) < 2:
        return 0.0
    fa = np.searchsorted(a, grid[:-1], side="right") / len(a)
    fb = np.searchsorted(b, grid[:-1], side="right") / len(b)
    return float(np.sum(np.abs(fa - fb) * np.diff(grid)))


def opinion_matrix(run, measure: str = "self") -> np.ndarray:
    """The run's opinions, shape (steps + 1, agents), as measured by `measure`.

    "self" is the self-rating (the simulation state). A scorer model's name uses its ratings
    of the agents' messages (scores.parquet, field "message"). A text the scorer found unclear
    or failed on keeps that agent's previous rating; one missing from the start takes the
    first rating it gets (a scorer reads what agents wrote, so this only fills gaps).
    Numeric runs and LLM runs that were not scored only have "self".
    """
    if measure == "self":
        return run.opinion_matrix()
    s = run.scores
    if s is None:
        raise ValueError(f"{run.path} has no scores.parquet: run `disensa score` first")
    s = s[(s["scorer_model"] == measure) & (s["field"] == "message")]
    if s.empty:
        raise ValueError(f"{run.path} has no message scores by {measure}")
    X = (
        s.pivot(index="step", columns="agent", values="score")
        .reindex(index=range(run.steps + 1), columns=range(len(run.agents)))
        .astype(float)
        .ffill()
        .bfill()
    )
    return X.to_numpy()


def granularity(values) -> dict:
    """Counts of whole / half / other numbers (see report.py)."""
    from disensa.report import granularity as kind

    out = {"whole": 0, "half": 0, "other": 0}
    for v in values:
        out[kind(float(v))] += 1
    return out


def polarization_curve(X) -> np.ndarray:
    return np.array([polarization(row) for row in np.asarray(X, dtype=float)])


def summarize_x(X, delta: float = DELTA, tol: float = TOL) -> dict:
    """The summary of an opinion matrix of shape (steps + 1, agents)."""
    X = np.asarray(X, dtype=float)
    final = X[-1]
    moved = moved_fraction(X)
    return {
        "final_mean": float(final.mean()),
        "final_std": float(final.std()),
        "final_polarization": polarization(final),
        "clusters": clusters(final, delta),
        "convergence_step": convergence_step(X, tol),
        "drift": float(final.mean() - X[0].mean()),
        "moved_mean": float(moved.mean()) if len(moved) else 0.0,
        "moved_by_step": [round(float(v), 4) for v in moved],
    }


def summarize_run(run, delta: float = DELTA, tol: float = TOL, measure: str = "self") -> dict:
    """`summarize_x` of a loaded run (disensa.runs.load_run), with its opinions measured by
    `measure` (see opinion_matrix), plus the granularity of its self-ratings: for LLM runs
    over the valid self-ratings from step 1 on, as in the run report."""
    out = summarize_x(opinion_matrix(run, measure), delta, tol)
    beliefs = [
        e["belief_raw"]
        for e in run.events
        if e.get("step", 0) >= 1 and not e.get("error") and e.get("belief_raw") is not None
    ]
    if not beliefs and run.events:  # runs from before 0.3: the opinions, failures left out
        ops = run.opinions.set_index(["step", "agent"])["opinion"]
        beliefs = [
            ops[(e["step"], e["agent"])]
            for e in run.events
            if e["step"] >= 1 and not e.get("error") and (e["step"], e["agent"]) in ops.index
        ]
    out["granularity"] = granularity(beliefs) if beliefs else None
    return out
