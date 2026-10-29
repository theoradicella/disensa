"""Classic numeric opinion dynamics: the agent-based-model baseline.

Each rule maps the opinions at t-1 to the opinions at t for every agent at once, using the
adjacency matrix A (A[i, j] = 1 if i and j are neighbours). Opinions live on [-2, 2].
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from disensa.config import NumericDynamics


def degroot(a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """DeGroot (1974): everyone moves to the average of themselves and their neighbours.

    x_i(t) = mean_{j ∈ N(i) ∪ {i}} x_j(t-1)
    """
    w = a + np.eye(len(x))
    return (w @ x) / w.sum(axis=1)


def bounded_confidence(a: np.ndarray, x: np.ndarray, epsilon: float, mu: float) -> np.ndarray:
    """Inertial Hegselmann–Krause on a graph (Chazelle & Wang, 2017).

    Agent i averages the opinions within ε of its own among its neighbours and itself, and
    moves a fraction μ of the way towards that average (μ = 1 is plain Hegselmann–Krause):
    x_i(t) = x_i(t-1) + μ · (mean_{j ∈ N(i) ∪ {i}, |x_i - x_j| ≤ ε} x_j(t-1) - x_i(t-1))
    """
    diff = x[None, :] - x[:, None]  # diff[i, j] = x_j - x_i
    confident = ((a > 0) | np.eye(len(x), dtype=bool)) & (np.abs(diff) <= epsilon)
    pull = (diff * confident).sum(axis=1) / confident.sum(axis=1)  # i itself is always in
    return x + mu * pull


def friedkin_johnsen(a: np.ndarray, x: np.ndarray, s: np.ndarray, lam: float) -> np.ndarray:
    """Friedkin–Johnsen (1990), in the form of Chitra & Musco (2020) cited by Wang et al. §2.

    Every agent keeps its initial opinion s_i as an anchor, weighted by the stubbornness λ,
    and averages it with its neighbours' current opinions:
    x_i(t) = (λ s_i + Σ_j A_ij x_j(t-1)) / (λ + k_i)
    With λ = 1 this is exactly Chitra & Musco's update.
    """
    return (lam * s + a @ x) / (lam + a.sum(axis=1))


class NumericAgents:
    """Numeric rules behind the same interface as LLM agents (see dynamics/__init__.py)."""

    def __init__(self, cfg: NumericDynamics, graph: nx.Graph):
        self.cfg = cfg
        self.n = graph.number_of_nodes()
        self.a = nx.to_numpy_array(graph, nodelist=range(self.n))
        self.x0: np.ndarray | None = None  # initial opinions, the anchor of Friedkin–Johnsen

    def labels(self) -> list[str]:
        return [f"numeric/{self.cfg.rule}"] * self.n

    async def start(self, x0: np.ndarray) -> list[dict]:
        self.x0 = x0.copy()
        return []

    async def step(self, x: np.ndarray, t: int) -> tuple[np.ndarray, list[dict]]:
        if self.cfg.rule == "degroot":
            return degroot(self.a, x), []
        if self.cfg.rule == "friedkin_johnsen":
            return friedkin_johnsen(self.a, x, self.x0, self.cfg.stubbornness), []
        return bounded_confidence(self.a, x, self.cfg.epsilon, self.cfg.mu), []
