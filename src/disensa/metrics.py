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


def local_disagreement(g: nx.Graph, x: np.ndarray) -> np.ndarray:
    """DG_i = Σ_{j∈N(i)} (x_i - x_j)² / |N(i)|. Isolated nodes get 0."""
    out = np.zeros(g.number_of_nodes())
    for i in g.nodes:
        nbrs = list(g.neighbors(i))
        if nbrs:
            out[i] = np.mean((x[i] - x[nbrs]) ** 2)
    return out


def global_disagreement(g: nx.Graph, x: np.ndarray) -> float:
    """DG = (1 / 2N) Σ_i DG_i   (Wang et al. eq. in §4.2)."""
    return float(local_disagreement(g, x).sum() / (2 * g.number_of_nodes()))


def neighbor_mean(g: nx.Graph, x: np.ndarray) -> np.ndarray:
    """Average opinion of each node's neighbours (NaN for isolated nodes)."""
    out = np.full(g.number_of_nodes(), np.nan)
    for i in g.nodes:
        nbrs = list(g.neighbors(i))
        if nbrs:
            out[i] = x[nbrs].mean()
    return out


def neighbor_correlation(g: nx.Graph, x: np.ndarray) -> float:
    """Pearson correlation between x_i and the mean of i's neighbours (Cinus et al. 2021).

    Note: Wang et al. call this "Normalized Clustering Index (NCI)" in §4.2 but
    "Neighbors Correlation" in Table 2; this is the neighbour-correlation definition.
    Returns NaN when opinions have no variance (e.g. full consensus).
    """
    nm = neighbor_mean(g, x)
    mask = ~np.isnan(nm)
    a, b = x[mask], nm[mask]
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def opinion_assortativity(g: nx.Graph, x: np.ndarray) -> float:
    """Newman's numeric assortativity of opinions across edges."""
    nx.set_node_attributes(g, {i: float(x[i]) for i in g.nodes}, "opinion")
    return float(nx.numeric_assortativity_coefficient(g, "opinion"))


def all_metrics(g: nx.Graph, x: np.ndarray) -> dict[str, float]:
    return {
        "polarization": polarization(x),
        "global_disagreement": global_disagreement(g, x),
        "neighbor_correlation": neighbor_correlation(g, x),
        "mean_opinion": float(np.mean(x)),
    }
