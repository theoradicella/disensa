"""Model assignment: which LLM each agent runs on.

Strategies (config `assignment`):
  balanced   exact shares from the weights (largest-remainder rounding), randomly placed
  random     each agent draws a model independently with probability ∝ weight
  clustered  exact shares, but whole network communities (Louvain) get the same model,
             so model identity is correlated with network position

All strategies use their own RNG stream (seed + 2_000) and never touch personas, so the same
population can be compared under different model mixes.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from agorasim.config import ModelSpec


def target_counts(models: list[ModelSpec], n: int) -> list[int]:
    """Largest-remainder split of n agents by model weight (counts sum exactly to n)."""
    w = np.array([m.weight for m in models], dtype=float)
    raw = w / w.sum() * n
    counts = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - counts), kind="stable")[: n - counts.sum()]:
        counts[i] += 1
    return counts.tolist()


def assign_random(models: list[ModelSpec], n: int, rng: np.random.Generator) -> list[ModelSpec]:
    w = np.array([m.weight for m in models], dtype=float)
    return [models[i] for i in rng.choice(len(models), size=n, p=w / w.sum())]


def assign_balanced(models: list[ModelSpec], n: int, rng: np.random.Generator) -> list[ModelSpec]:
    pool = [m for m, c in zip(models, target_counts(models, n), strict=True) for _ in range(c)]
    return [pool[i] for i in rng.permutation(n)]


def assign_clustered(
    models: list[ModelSpec], graph: nx.Graph, rng: np.random.Generator
) -> list[ModelSpec]:
    """Order agents community by community (largest first), then hand out contiguous blocks."""
    n = graph.number_of_nodes()
    seed = int(rng.integers(0, 2**31 - 1))
    communities = nx.community.louvain_communities(graph, seed=seed)
    communities = sorted(communities, key=lambda c: (-len(c), min(c)))
    order = [node for c in communities for node in sorted(c)]
    out: list[ModelSpec | None] = [None] * n
    pos = 0
    for m, c in zip(models, target_counts(models, n), strict=True):
        for node in order[pos : pos + c]:
            out[node] = m
        pos += c
    return out  # type: ignore[return-value]


def assign_models(
    models: list[ModelSpec], graph: nx.Graph, strategy: str, seed: int
) -> list[ModelSpec]:
    if not models:
        raise ValueError("config needs at least one entry under `models`")
    rng = np.random.default_rng(seed + 2_000)
    n = graph.number_of_nodes()
    if strategy == "random":
        return assign_random(models, n, rng)
    if strategy == "balanced":
        return assign_balanced(models, n, rng)
    if strategy == "clustered":
        return assign_clustered(models, graph, rng)
    raise ValueError(f"unknown assignment strategy: {strategy}")


def model_homophily(graph: nx.Graph, models: list[ModelSpec]) -> float:
    """Share of edges joining two agents on the same model (1 = fully segregated)."""
    edges = list(graph.edges())
    if not edges:
        return float("nan")
    return sum(models[u].name == models[v].name for u, v in edges) / len(edges)
