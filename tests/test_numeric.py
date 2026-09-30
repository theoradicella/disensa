import networkx as nx
import numpy as np
import pytest

from disensa.dynamics.numeric import bounded_confidence, degroot


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
    assert new[2] == 2.0  # its only neighbour is 3.5 away: nobody to listen to
    assert new[0] == pytest.approx(-2.0 + 0.5 * 0.5)
    assert new[1] == pytest.approx(-1.5 + 0.5 * (-0.5))  # listens to node 0 only
