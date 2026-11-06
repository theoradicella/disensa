"""Macro fit: which numeric rule, with which parameters, does an LLM run resemble?

For one run, every numeric rule is simulated on the run's own world (its network and its
starting opinions, read from the run folder) over a grid of parameters, for as many steps as
the run had. Each simulation is compared with the run by several distances:

  wasserstein   between the final opinion distributions (the main one)
  clusters      |Δ number of clusters| of the final opinions (summary.clusters, δ = 0.5)
  drift         |Δ (final mean − initial mean)|
  polarization  root mean square difference of the polarization curves over the steps

and, per rule and distance, the best parameters are reported. A "nobody moves" baseline,
x(t) = x(0), is scored the same way: a rule only explains a run if it beats it.

The simulations use the same NumericAgents as the engine (dynamics/numeric.py), started from
the recorded x(0) and edges rather than from the seed, so they also fit runs made under the
earlier seeding scheme. They are instant, so nothing is written to disk.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable

import networkx as nx
import numpy as np
import pandas as pd

from disensa import OPINION_MAX, OPINION_MIN
from disensa.analysis.summary import (
    DELTA,
    clusters,
    opinion_matrix,
    polarization_curve,
    wasserstein,
)
from disensa.config import NumericDynamics
from disensa.dynamics.numeric import NumericAgents

MEASURES = ("wasserstein", "clusters", "drift", "polarization")

# default parameter grids (bounded confidence: ε × μ; Friedkin–Johnsen: λ)
EPSILONS = tuple(round(v, 2) for v in np.arange(0.1, 4.01, 0.1))
MUS = (0.1, 0.25, 0.5, 0.75, 1.0)
STUBBORNNESS = tuple(round(float(v), 4) for v in np.logspace(-2, 2, 25))


def grid(
    epsilons: Iterable[float] = EPSILONS,
    mus: Iterable[float] = MUS,
    stubbornness: Iterable[float] = STUBBORNNESS,
) -> list[NumericDynamics]:
    """Every numeric dynamics to try: DeGroot once, then the parameter grids."""
    out = [NumericDynamics(rule="degroot")]
    out += [
        NumericDynamics(rule="bounded_confidence", epsilon=e, mu=m) for e in epsilons for m in mus
    ]
    out += [NumericDynamics(rule="friedkin_johnsen", stubbornness=s) for s in stubbornness]
    return out


def world(run) -> tuple[nx.Graph, np.ndarray]:
    """The run's network and starting opinions, as recorded in its folder."""
    n = len(run.agents)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(map(tuple, run.edges))
    return g, run.opinion_matrix()[0]


def simulate_numeric(graph: nx.Graph, x0: np.ndarray, dyn: NumericDynamics, steps: int):
    """Opinion matrix (steps + 1, agents) of a numeric rule, exactly as the engine runs it."""
    agents = NumericAgents(dyn, graph)

    async def go():
        x = np.asarray(x0, dtype=float).copy()
        out = [x]
        await agents.start(x)
        for t in range(1, steps + 1):
            x, _ = await agents.step(x, t)
            x = np.clip(x, OPINION_MIN, OPINION_MAX)
            out.append(x)
        return np.array(out)

    return asyncio.run(go())


def distances(target: np.ndarray, sim: np.ndarray, delta: float = DELTA) -> dict:
    """The four distances between a run's opinion matrix and a simulation's."""
    return {
        "wasserstein": wasserstein(target[-1], sim[-1]),
        "clusters": abs(clusters(target[-1], delta) - clusters(sim[-1], delta)),
        "drift": abs((target[-1].mean() - target[0].mean()) - (sim[-1].mean() - sim[0].mean())),
        "polarization": float(
            np.sqrt(np.mean((polarization_curve(target) - polarization_curve(sim)) ** 2))
        ),
    }


def label(dyn: NumericDynamics) -> dict:
    if dyn.rule == "bounded_confidence":
        return {"rule": dyn.rule, "epsilon": dyn.epsilon, "mu": dyn.mu, "stubbornness": None}
    if dyn.rule == "friedkin_johnsen":
        return {"rule": dyn.rule, "epsilon": None, "mu": None, "stubbornness": dyn.stubbornness}
    return {"rule": dyn.rule, "epsilon": None, "mu": None, "stubbornness": None}


def scan(
    run,
    candidates: list[NumericDynamics] | None = None,
    delta: float = DELTA,
    measure: str = "self",
) -> pd.DataFrame:
    """Every candidate (and the "nobody moves" baseline) with its four distances to the run.

    The simulations always start from the run's true x(0); the run itself is read through
    `measure` (the self-ratings, or a scorer's ratings: see summary.opinion_matrix).
    """
    graph, x0 = world(run)
    target = opinion_matrix(run, measure)
    steps = len(target) - 1
    rows = [{"rule": "nobody_moves", **distances(target, np.repeat(x0[None], steps + 1, 0), delta)}]
    for dyn in candidates or grid():
        sim = simulate_numeric(graph, x0, dyn, steps)
        rows.append({**label(dyn), **distances(target, sim, delta)})
    return pd.DataFrame(rows)


def best(scanned: pd.DataFrame) -> pd.DataFrame:
    """Per rule and distance: the best parameters and the distance there.

    When several parameter values tie (common: a range of ε gives the same final state),
    the reported value is the median of the tied ones, and `tied` says how many there were.
    """
    rows = []
    for rule, g in scanned.groupby("rule", sort=False):
        for m in MEASURES:
            low = g[m].min()
            tied = g[np.isclose(g[m], low, rtol=0, atol=1e-9)]
            row = {"rule": rule, "measure": m, "distance": float(low), "tied": len(tied)}
            for p in ("epsilon", "mu", "stubbornness"):
                vals = tied[p].dropna() if p in tied else []
                row[p] = float(np.median(vals)) if len(vals) else None
            rows.append(row)
    return pd.DataFrame(rows)


def fit_run(
    run,
    candidates: list[NumericDynamics] | None = None,
    delta: float = DELTA,
    measure: str = "self",
):
    """(best table, full scan) for one run. See the module docstring."""
    scanned = scan(run, candidates, delta, measure)
    return best(scanned), scanned


def best_dynamics(table: pd.DataFrame, rule: str, measure: str = "wasserstein"):
    """The NumericDynamics of the best fit of one rule, e.g. to simulate it for a figure."""
    row = table[(table["rule"] == rule) & (table["measure"] == measure)].iloc[0]
    if rule == "bounded_confidence":
        return NumericDynamics(rule=rule, epsilon=row["epsilon"], mu=row["mu"])
    if rule == "friedkin_johnsen":
        return NumericDynamics(rule=rule, stubbornness=row["stubbornness"])
    return NumericDynamics(rule=rule)
