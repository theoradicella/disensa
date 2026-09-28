import networkx as nx
import numpy as np

from disensa.config import RewiringConfig
from disensa.rewiring import rewire_step


def two_camps(n=20):
    g = nx.connected_watts_strogatz_graph(n, 4, 0.3, seed=1)
    x = np.array([-2.0 if i % 2 else 2.0 for i in range(n)])
    return g, x


def test_edges_conserved_and_never_isolated():
    g, x = two_camps()
    m = g.number_of_edges()
    rng = np.random.default_rng(0)
    cfg = RewiringConfig(enabled=True, probability=1.0, unfollow_threshold=1.0)
    for _ in range(20):
        rewire_step(g, x, cfg, rng)
        assert g.number_of_edges() == m
        assert min(d for _, d in g.degree()) >= 1


def test_similar_following_sorts_camps():
    g, x = two_camps()
    rng = np.random.default_rng(0)
    cfg = RewiringConfig(enabled=True, probability=1.0, unfollow_threshold=1.0)

    def cross(g):
        return sum(x[u] != x[v] for u, v in g.edges())

    before = cross(g)
    for _ in range(30):
        rewire_step(g, x, cfg, rng)
    assert cross(g) < before / 2


def test_changes_are_reported_and_match_graph():
    g, x = two_camps()
    rng = np.random.default_rng(3)
    cfg = RewiringConfig(enabled=True, probability=1.0, unfollow_threshold=1.0)
    for c in rewire_step(g, x, cfg, rng):
        assert g.has_edge(c.agent, c.added)


def test_zero_probability_does_nothing():
    g, x = two_camps()
    edges = set(g.edges())
    cfg = RewiringConfig(enabled=True, probability=0.0)
    assert rewire_step(g, x, cfg, np.random.default_rng(0)) == []
    assert set(g.edges()) == edges
