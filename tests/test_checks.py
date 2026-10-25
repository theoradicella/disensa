import networkx as nx
import pytest

from disensa.checks import (
    config_checks,
    er_connected_probability,
    er_min_p,
    network_checks,
    opinion_checks,
    raw_checks,
)
from disensa.config import ExperimentConfig, InitialOpinions, NetworkConfig


def fields(checks):
    return [c.field for c in checks]


def test_er_connected_probability_matches_sampling():
    n, p = 50, 0.08
    share = sum(nx.is_connected(nx.erdos_renyi_graph(n, p, seed=s)) for s in range(400)) / 400
    assert abs(er_connected_probability(n, p) - share) < 0.08


def test_er_threshold():
    assert er_min_p(50) == 0.12
    assert "network.p" in fields(network_checks(NetworkConfig(topology="erdos_renyi", p=0.08)))
    assert network_checks(NetworkConfig(topology="erdos_renyi", p=0.12)) == []


def test_er_hopeless_p_says_the_run_will_fail():
    (c,) = network_checks(NetworkConfig(topology="erdos_renyi", p=0.02))
    assert "fail" in c.message


@pytest.mark.parametrize(
    ("kw", "field"),
    [
        ({"k": 2, "beta": 0.1}, "network.k"),  # no triangles
        ({"k": 6, "beta": 0.0}, "network.beta"),  # plain lattice
        ({"k": 6, "beta": 0.005}, "network.beta"),  # almost no shortcuts
        ({"k": 6, "beta": 0.5}, "network.beta"),  # clustering gone
    ],
)
def test_watts_strogatz_warnings(kw, field):
    assert field in fields(network_checks(NetworkConfig(topology="watts_strogatz", **kw)))


def test_good_settings_are_quiet():
    assert network_checks(NetworkConfig(topology="watts_strogatz", k=6, beta=0.1)) == []
    assert network_checks(NetworkConfig(topology="barabasi_albert", m=3)) == []
    assert opinion_checks(InitialOpinions(distribution="bimodal", mean=1.2, std=0.4)) == []


def test_barabasi_albert_tree_and_dense():
    assert fields(network_checks(NetworkConfig(topology="barabasi_albert", m=1))) == ["network.m"]
    dense = network_checks(NetworkConfig(topology="erdos_renyi", n=30, p=0.7))
    assert "describing-a-network" in [c.docs for c in dense]


def test_impossible_settings_are_errors():
    with pytest.raises(ValueError):
        NetworkConfig(topology="watts_strogatz", n=10, k=10, beta=0.1)
    with pytest.raises(ValueError):
        NetworkConfig(topology="watts_strogatz", n=10, k=5, beta=0.1)
    with pytest.raises(ValueError):
        NetworkConfig(topology="barabasi_albert", n=5, m=5)


def test_opinions_clipping_and_overlap():
    assert opinion_checks(InitialOpinions(distribution="normal", mean=0, std=2))
    assert opinion_checks(InitialOpinions(distribution="bimodal", mean=0.5, std=1))


def test_numeric_epsilon_and_interactions():
    cfg = ExperimentConfig.model_validate(
        {
            "name": "t",
            "network": {"topology": "barabasi_albert", "m": 2},
            "dynamics": {"type": "numeric", "epsilon": 5},
        }
    )
    assert "dynamics.epsilon" in fields(config_checks(cfg))
    raw = {
        "network": {"topology": "barabasi_albert", "m": 2},
        "dynamics": {"type": "llm", "models": [], "interactions": {"min": 4, "max": 5}},
    }
    assert "dynamics.interactions.min" in [c["field"] for c in raw_checks(raw)]


def test_raw_checks_reports_invalid_sections_as_errors():
    out = raw_checks({"network": {"topology": "watts_strogatz", "k": 6}})
    assert out and all(c["level"] == "error" for c in out)
