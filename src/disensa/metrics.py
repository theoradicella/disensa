"""Echo-chamber and polarization metrics, following Wang et al. (COLING 2025, §4.2).

All functions take a graph and an opinion vector `x` indexed by node id (0..N-1).
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def polarization(x: np.ndarray) -> float:
    """Pz = (1/N) Σ_i (x_i - mean(x))²   (Chitra & Musco 2020)."""
    x = np.asarray(x, dtype=float)
    return float(np.mean((x - x.mean()) ** 2))


def global_disagreement(g: nx.Graph, x: np.ndarray) -> float:
    """DG = (1 / 2N) Σ_i DG_i, with DG_i = mean over neighbours j of (x_i - x_j)²."""
    total = 0.0
    for i in g.nodes:
        nbrs = list(g.neighbors(i))
        if nbrs:
            total += np.mean((x[i] - x[nbrs]) ** 2)
    return float(total / (2 * g.number_of_nodes()))


def neighbor_correlation(g: nx.Graph, x: np.ndarray) -> float:
    """Pearson correlation between x_i and the mean opinion of i's neighbours (Cinus et al. 2021).

    Wang et al. call this "Normalized Clustering Index (NCI)" in §4.2 but "Neighbors
    Correlation" in Table 2. Returns NaN when opinions have no variance (full consensus).
    """
    own, around = [], []
    for i in g.nodes:
        nbrs = list(g.neighbors(i))
        if nbrs:
            own.append(x[i])
            around.append(x[nbrs].mean())
    a, b = np.array(own), np.array(around)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def all_metrics(g: nx.Graph, x: np.ndarray) -> dict[str, float]:
    return {
        "polarization": polarization(x),
        "global_disagreement": global_disagreement(g, x),
        "neighbor_correlation": neighbor_correlation(g, x),
        "mean_opinion": float(np.mean(x)),
    }
