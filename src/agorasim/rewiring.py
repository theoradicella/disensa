"""Opinion-driven network rewiring: agents unfollow people they disagree with and follow others.

A simple version of the coevolving-network models of Holme & Newman (2006) and the
unfollowing mechanism of Sasahara et al. (2021). At the end of each day, each agent
independently, with probability `probability`:
  1. looks at its neighbours whose opinion differs from its own by more than
     `unfollow_threshold`; if there are none, nothing happens;
  2. drops the edge to the most distant one (ties broken at random), unless that would leave
     either endpoint with no edges;
  3. adds an edge to a random non-neighbour: one within `follow_threshold` of its opinion
     when `follow` is "similar" (falls back to anyone if none qualifies), anyone when "random".
The number of edges is conserved. Uses its own RNG stream (seed + 3_000).
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from agorasim.config import RewiringConfig


@dataclass(frozen=True)
class Rewire:
    agent: int
    dropped: int
    added: int


def rewire_step(
    graph: nx.Graph, opinions: np.ndarray, cfg: RewiringConfig, rng: np.random.Generator
) -> list[Rewire]:
    """Mutates `graph` in place; returns what changed. Agents act in random order."""
    changes: list[Rewire] = []
    n = graph.number_of_nodes()
    for i in rng.permutation(n).tolist():
        if rng.random() >= cfg.probability:
            continue
        nbrs = list(graph.neighbors(i))
        dist = {j: abs(opinions[i] - opinions[j]) for j in nbrs}
        far = [j for j in nbrs if dist[j] > cfg.unfollow_threshold]
        if not far or graph.degree(i) <= 1:
            continue
        worst = max(dist[j] for j in far)
        candidates = [j for j in far if dist[j] == worst and graph.degree(j) > 1]
        if not candidates:
            continue
        drop = int(rng.choice(candidates))

        others = [k for k in range(n) if k != i and not graph.has_edge(i, k)]
        if cfg.follow == "similar":
            close = [k for k in others if abs(opinions[i] - opinions[k]) <= cfg.follow_threshold]
            pool = close or others
        else:
            pool = others
        if not pool:
            continue
        add = int(rng.choice(pool))

        graph.remove_edge(i, drop)
        graph.add_edge(i, add)
        changes.append(Rewire(agent=i, dropped=drop, added=add))
    return changes
