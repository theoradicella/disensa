import networkx as nx
import numpy as np
import pytest

from disensa.config import InitialOpinions, NetworkConfig
from disensa.metrics import global_disagreement, neighbor_correlation, polarization
from disensa.network import build_network, initial_opinions


@pytest.mark.parametrize(
    "cfg",
    [
        {"topology": "erdos_renyi", "n": 30, "p": 0.2},
        {"topology": "watts_strogatz", "n": 30, "k": 4, "beta": 0.1},
        {"topology": "barabasi_albert", "n": 30, "m": 2},
    ],
)
def test_networks_are_connected_and_seeded(cfg):
    c = NetworkConfig(**cfg)
    g = build_network(c, seed=1)
    assert nx.is_connected(g) and sorted(g.nodes) == list(range(30))
    assert sorted(g.edges) == sorted(build_network(c, seed=1).edges)


def test_missing_topology_parameter_is_rejected():
    with pytest.raises(ValueError):
        NetworkConfig(topology="watts_strogatz", n=10, k=4)


@pytest.mark.parametrize("dist", ["uniform", "normal", "bimodal"])
def test_initial_opinions_in_range(dist):
    x = initial_opinions(InitialOpinions(distribution=dist, mean=1.5, std=1), 500, seed=0)
    assert x.min() >= -2 and x.max() <= 2


def test_metrics_on_a_path():
    g = nx.path_graph(3)
    x = np.array([-2.0, 0.0, 2.0])
    assert polarization(x) == pytest.approx(8 / 3)
    # DG_i: node0 4, node1 (4+4)/2=4, node2 4 -> 12 / (2*3)
    assert global_disagreement(g, x) == pytest.approx(2.0)
    assert np.isnan(neighbor_correlation(g, np.zeros(3)))


def test_redraws_do_not_reuse_other_seeds():
    from disensa.network import _redraw_seed

    redraws = {_redraw_seed(s, a) for s in range(20) for a in range(1, 10)}
    assert not redraws & set(range(20))
    assert len(redraws) == 20 * 9
