from collections import Counter

import networkx as nx
import pytest

from disensa.config import ModelSpec
from disensa.ensemble import assign_models, model_homophily, target_counts

MODELS = [ModelSpec(name="a", weight=2), ModelSpec(name="b"), ModelSpec(name="c")]


def test_target_counts_sum_exactly():
    for n in (1, 7, 10, 33, 50):
        counts = target_counts(MODELS, n)
        assert sum(counts) == n
        assert counts[0] >= counts[1] >= counts[2] - 1  # 'a' has twice the weight


@pytest.mark.parametrize("strategy", ["balanced", "clustered"])
def test_exact_shares(strategy):
    g = nx.connected_watts_strogatz_graph(40, 4, 0.1, seed=1)
    counts = Counter(m.name for m in assign_models(MODELS, g, strategy, seed=0))
    assert counts == {"a": 20, "b": 10, "c": 10}


def test_assignment_is_seeded():
    g = nx.barabasi_albert_graph(30, 2, seed=1)
    a = [m.name for m in assign_models(MODELS, g, "balanced", seed=4)]
    b = [m.name for m in assign_models(MODELS, g, "balanced", seed=4)]
    c = [m.name for m in assign_models(MODELS, g, "balanced", seed=5)]
    assert a == b and a != c


def test_clustered_is_more_segregated_than_balanced():
    g = nx.connected_watts_strogatz_graph(60, 6, 0.05, seed=2)
    clustered = model_homophily(g, assign_models(MODELS, g, "clustered", seed=0))
    balanced = model_homophily(g, assign_models(MODELS, g, "balanced", seed=0))
    assert clustered > balanced + 0.2


def test_unknown_strategy():
    with pytest.raises(ValueError):
        assign_models(MODELS, nx.path_graph(3), "nope", seed=0)
