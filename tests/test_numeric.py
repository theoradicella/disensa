import networkx as nx
import numpy as np
import pytest

from disensa.dynamics.numeric import bounded_confidence, degroot, friedkin_johnsen


def adj(g):
    return nx.to_numpy_array(g, nodelist=range(g.number_of_nodes()))


def test_degroot_averages_self_and_neighbours():
    a = adj(nx.path_graph(3))
    x = np.array([-2.0, 0.0, 2.0])
    assert degroot(a, x) == pytest.approx([-1.0, 0.0, 1.0])


def test_degroot_reaches_consensus_on_a_connected_graph():
    a = adj(nx.watts_strogatz_graph(30, 4, 0.2, seed=1))
    x = np.random.default_rng(0).uniform(-2, 2, 30)
    for _ in range(300):
        x = degroot(a, x)
    assert x.std() < 1e-3


def test_bounded_confidence_ignores_distant_neighbours():
    a = adj(nx.path_graph(3))
    x = np.array([-2.0, -1.5, 2.0])
    new = bounded_confidence(a, x, epsilon=1.0, mu=0.5)
    assert new[2] == 2.0  # its only neighbour is 3.5 away: it only listens to itself
    assert new[0] == pytest.approx(-2.0 + 0.5 * (-1.75 + 2.0))  # mean of itself and node 1
    assert new[1] == pytest.approx(-1.5 + 0.5 * (-1.75 + 1.5))  # node 0 and itself, not node 2


def test_bounded_confidence_with_mu_one_is_hegselmann_krause():
    a = adj(nx.path_graph(2))
    x = np.array([-0.5, 0.5])
    # both in each other's set: they meet in the middle instead of swapping
    assert bounded_confidence(a, x, epsilon=2.0, mu=1.0) == pytest.approx([0.0, 0.0])


def test_bounded_confidence_with_large_epsilon_and_mu_one_is_degroot():
    a = adj(nx.watts_strogatz_graph(20, 4, 0.2, seed=1))
    x = np.random.default_rng(0).uniform(-2, 2, 20)
    assert bounded_confidence(a, x, epsilon=4.0, mu=1.0) == pytest.approx(degroot(a, x))


def test_friedkin_johnsen_mixes_anchor_and_neighbours():
    a = adj(nx.path_graph(3))
    s = np.array([-2.0, 0.0, 2.0])
    x = np.array([-1.0, 1.0, 1.0])
    new = friedkin_johnsen(a, x, s, lam=1.0)
    assert new == pytest.approx([(-2.0 + 1.0) / 2, (0.0 - 1.0 + 1.0) / 3, (2.0 + 1.0) / 2])


def test_friedkin_johnsen_settles_short_of_consensus():
    g = nx.watts_strogatz_graph(30, 4, 0.2, seed=1)
    a = adj(g)
    s = np.random.default_rng(0).uniform(-2, 2, 30)
    x = s.copy()
    for _ in range(500):
        x = friedkin_johnsen(a, x, s, lam=1.0)
    laplacian = np.diag(a.sum(axis=1)) - a
    assert x == pytest.approx(np.linalg.solve(np.eye(30) + laplacian, s))  # (I + L)^-1 s
    assert x.std() > 0.05  # disagreement persists
