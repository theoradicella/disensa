"""The network and the starting opinions. Both are seeded, so a run is reproducible."""

from __future__ import annotations

import networkx as nx
import numpy as np

from disensa import OPINION_MAX, OPINION_MIN
from disensa.config import InitialOpinions, NetworkConfig


def build_network(cfg: NetworkConfig, seed: int = 0, max_tries: int = 100) -> nx.Graph:
    """Build the configured topology. Nodes are 0..n-1. Resampled until connected.

    The first draw uses `seed` itself; redraws use seeds derived from (seed, attempt), so the
    redraws of one run never reuse another run's network (as `seed + attempt` did).
    """
    for attempt in range(max_tries):
        s = seed if attempt == 0 else _redraw_seed(seed, attempt)
        if cfg.topology == "erdos_renyi":
            g = nx.erdos_renyi_graph(cfg.n, cfg.p, seed=s)
        elif cfg.topology == "watts_strogatz":
            g = nx.connected_watts_strogatz_graph(cfg.n, cfg.k, cfg.beta, seed=s)
        else:
            g = nx.barabasi_albert_graph(cfg.n, cfg.m, seed=s)
        if nx.is_connected(g):
            return g
    raise RuntimeError(f"Could not build a connected {cfg.topology} graph in {max_tries} tries")


def _redraw_seed(seed: int, attempt: int) -> int:
    return int(np.random.SeedSequence([abs(seed), attempt]).generate_state(1)[0])


def initial_opinions(cfg: InitialOpinions, n: int, seed: int = 0) -> np.ndarray:
    """Draw n initial opinions on [OPINION_MIN, OPINION_MAX]."""
    rng = np.random.default_rng(seed)
    if cfg.distribution == "uniform":
        x = rng.uniform(OPINION_MIN, OPINION_MAX, n)
    elif cfg.distribution == "normal":
        x = rng.normal(cfg.mean, cfg.std, n)
    else:  # bimodal: two camps at ±mean
        signs = rng.choice([-1.0, 1.0], n)
        x = rng.normal(signs * abs(cfg.mean), cfg.std)
    return np.clip(x, OPINION_MIN, OPINION_MAX)
