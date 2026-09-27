import networkx as nx
import numpy as np
import pytest

from agorasim.baselines import run_baseline
from agorasim.config import BaselineConfig, NetworkConfig
from agorasim.networks import build_network


@pytest.fixture
def complete():
    return nx.complete_graph(20)


def test_degroot_reaches_consensus_on_connected_graph(complete):
    x0 = np.linspace(-2, 2, 20)
    traj = run_baseline(complete, x0, BaselineConfig(model="degroot", steps=50))
    assert np.ptp(traj[-1]) < 1e-6
    assert abs(traj[-1].mean() - x0.mean()) < 1e-6  # complete graph preserves the mean


def test_bounded_confidence_wide_epsilon_gives_consensus(complete):
    x0 = np.linspace(-2, 2, 20)
    cfg = BaselineConfig(model="bounded_confidence", steps=200, epsilon=4.0, mu=0.5)
    assert np.ptp(run_baseline(complete, x0, cfg)[-1]) < 1e-3


def test_bounded_confidence_small_epsilon_fragments(complete):
    x0 = np.array([-2.0] * 10 + [2.0] * 10)
    cfg = BaselineConfig(model="bounded_confidence", steps=100, epsilon=0.5, mu=0.5)
    final = run_baseline(complete, x0, cfg)[-1]
    assert np.allclose(final, x0)  # the two camps never interact


def test_deffuant_conserves_mean(complete):
    x0 = np.random.default_rng(0).uniform(-2, 2, 20)
    cfg = BaselineConfig(model="deffuant", steps=20, epsilon=4.0, mu=0.3)
    assert abs(run_baseline(complete, x0, cfg)[-1].mean() - x0.mean()) < 1e-9


def test_friedkin_johnsen_keeps_disagreement():
    g = build_network(NetworkConfig(topology="barabasi_albert", n=30, m=2), seed=1)
    x0 = np.random.default_rng(1).uniform(-2, 2, 30)
    final = run_baseline(g, x0, BaselineConfig(model="friedkin_johnsen", steps=300))[-1]
    assert np.ptp(final) > 0.1  # stubborn agents prevent full consensus


@pytest.mark.parametrize(
    "net",
    [
        NetworkConfig(topology="erdos_renyi", n=40, p=0.1),
        NetworkConfig(topology="watts_strogatz", n=40, k=4, beta=0.1),
        NetworkConfig(topology="barabasi_albert", n=40, m=2),
    ],
)
def test_networks_are_connected_and_seeded(net):
    g1, g2 = build_network(net, seed=7), build_network(net, seed=7)
    assert nx.is_connected(g1)
    assert set(g1.edges()) == set(g2.edges())
