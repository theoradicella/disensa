"""Classic numeric opinion-dynamics models on a graph.

Each model is a function (g, x0, params, rng) -> trajectory array of shape (steps + 1, N).
Opinions live on [OPINION_MIN, OPINION_MAX].
"""

from __future__ import annotations

from collections.abc import Callable

import networkx as nx
import numpy as np

from agorasim import OPINION_MAX, OPINION_MIN
from agorasim.config import BaselineConfig


def _adjacency(g: nx.Graph) -> np.ndarray:
    return nx.to_numpy_array(g, nodelist=range(g.number_of_nodes()))


def degroot(g: nx.Graph, x0: np.ndarray, cfg: BaselineConfig, rng) -> np.ndarray:
    """DeGroot (1974): everyone moves to the average of self + neighbours."""
    a = _adjacency(g) + np.eye(len(x0))
    w = a / a.sum(axis=1, keepdims=True)
    traj = [x0.copy()]
    for _ in range(cfg.steps):
        traj.append(w @ traj[-1])
    return np.array(traj)


def bounded_confidence(g: nx.Graph, x0: np.ndarray, cfg: BaselineConfig, rng) -> np.ndarray:
    """Synchronous bounded confidence on a graph, as in Wang et al. §3.4.

    x_i(t) = x_i(t-1) + μ · mean_{j∈N(i), |x_i-x_j|≤ε} (x_j(t-1) - x_i(t-1))
    (Hegselmann–Krause style averaging over all confident neighbours.)
    """
    a = _adjacency(g).astype(bool)
    traj = [x0.copy()]
    for _ in range(cfg.steps):
        x = traj[-1]
        diff = x[None, :] - x[:, None]  # diff[i, j] = x_j - x_i
        confident = a & (np.abs(diff) <= cfg.epsilon)
        counts = confident.sum(axis=1)
        pull = np.where(counts > 0, (diff * confident).sum(axis=1) / np.maximum(counts, 1), 0.0)
        traj.append(x + cfg.mu * pull)
    return np.array(traj)


def deffuant(g: nx.Graph, x0: np.ndarray, cfg: BaselineConfig, rng) -> np.ndarray:
    """Deffuant et al. (2000): random pairwise encounters along edges.

    One "step" = |E| random edge encounters, so step counts are comparable across graphs.
    """
    edges = np.array(g.edges())
    x = x0.copy()
    traj = [x.copy()]
    for _ in range(cfg.steps):
        for e in rng.integers(0, len(edges), len(edges)):
            i, j = edges[e]
            d = x[j] - x[i]
            if abs(d) <= cfg.epsilon:
                x[i] += cfg.mu * d
                x[j] -= cfg.mu * d
        traj.append(x.copy())
    return np.array(traj)


def friedkin_johnsen(g: nx.Graph, x0: np.ndarray, cfg: BaselineConfig, rng) -> np.ndarray:
    """Friedkin–Johnsen (1990) as in Chitra & Musco (2020).

    Innate opinions s = x0 stay fixed; expressed opinions iterate
    z_i(t+1) = (λ s_i + Σ_j A_ij z_j(t)) / (λ + d_i).
    """
    a = _adjacency(g)
    lam = cfg.stubbornness
    denom = lam + a.sum(axis=1)
    s = x0.copy()
    traj = [s.copy()]
    for _ in range(cfg.steps):
        traj.append((lam * s + a @ traj[-1]) / denom)
    return np.array(traj)


MODELS: dict[str, Callable] = {
    "degroot": degroot,
    "bounded_confidence": bounded_confidence,
    "deffuant": deffuant,
    "friedkin_johnsen": friedkin_johnsen,
}


def run_baseline(g: nx.Graph, x0: np.ndarray, cfg: BaselineConfig, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    traj = MODELS[cfg.model](g, np.asarray(x0, dtype=float), cfg, rng)
    return np.clip(traj, OPINION_MIN, OPINION_MAX)
