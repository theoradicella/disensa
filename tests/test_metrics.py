import math

import networkx as nx
import numpy as np

from agorasim.metrics import global_disagreement, neighbor_correlation, polarization


def test_polarization_is_variance():
    x = np.array([-2.0, 2.0, -2.0, 2.0])
    assert polarization(x) == 4.0
    assert polarization(np.zeros(5)) == 0.0


def test_global_disagreement_path():
    # path 0-1-2 with opinions -1, 0, 1:
    # DG_0 = 1, DG_1 = (1 + 1)/2 = 1, DG_2 = 1  ->  DG = 3 / (2*3) = 0.5
    g = nx.path_graph(3)
    assert global_disagreement(g, np.array([-1.0, 0.0, 1.0])) == 0.5


def test_neighbor_correlation_echo_chamber():
    # two cliques joined by one edge, each clique holding one opinion -> strong positive NCI
    g = nx.barbell_graph(5, 0)
    x = np.array([-2.0] * 5 + [2.0] * 5)
    assert neighbor_correlation(g, x) > 0.8


def test_neighbor_correlation_consensus_is_nan():
    g = nx.cycle_graph(6)
    assert math.isnan(neighbor_correlation(g, np.ones(6)))
