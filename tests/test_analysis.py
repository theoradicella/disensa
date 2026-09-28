import numpy as np
import pytest

from disensa.analysis import (
    agent_timeline,
    change_points,
    load_run,
    model_summary,
    opinion_shifts,
)
from disensa.config import ExperimentConfig
from disensa.engine import Simulation
from disensa.runs import RunWriter, new_run_dir


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    import asyncio

    tmp = tmp_path_factory.mktemp("analysis")
    cfg = ExperimentConfig.model_validate(
        {
            "name": "a",
            "seed": 2,
            "network": {"topology": "watts_strogatz", "n": 24, "k": 4, "beta": 0.1},
            "simulation": {
                "days": 6,
                "recommender": {"enabled": False},
                "feed": {"min_posts": 2, "max_posts": 4},
                "rewiring": {"enabled": True, "probability": 0.3, "unfollow_threshold": 1.0},
            },
            "models": [{"name": "fake/conformist"}, {"name": "fake/contrarian"}],
            "scorer": {"name": "fake/scorer"},
            "llm": {"cache_dir": str(tmp / "cache"), "max_concurrency": 32},
            "output_dir": str(tmp / "runs"),
        }
    )
    run_dir = new_run_dir(cfg)
    writer = RunWriter(run_dir)
    asyncio.run(Simulation(cfg).run(writer))
    writer.close()
    return load_run(run_dir)


def test_load_run_shapes(run):
    assert run.days == 6
    assert run.opinion_matrix().shape == (7, 24)
    assert len(run.posts) == 24 * 7


def test_graph_at_follows_rewiring(run):
    assert len(run.rewires) > 0
    first_change = int(run.rewires["day"].min())
    g0, g1 = run.graph_at(0), run.graph_at(first_change)
    assert set(g0.edges()) != set(g1.edges())
    assert g0.number_of_edges() == g1.number_of_edges()


def test_agent_timeline(run):
    tl = agent_timeline(run, 3)
    assert list(tl["day"]) == list(range(7))
    assert np.allclose(tl["opinion"], run.opinion_matrix()[:, 3])


def test_shift_directions_match_models(run):
    s = opinion_shifts(run, min_delta=0.1)
    assert len(s) > 0
    conf = s[(s["model"] == "fake/conformist") & (s["direction"] != "none")]
    contra = s[(s["model"] == "fake/contrarian") & (s["direction"] != "none")]
    assert (conf["direction"] == "toward").mean() > 0.8
    assert (contra["direction"] == "away").mean() > 0.8


def test_model_summary(run):
    ms = model_summary(run, min_delta=0.1).set_index("model")
    assert ms["agents"].sum() == 24
    assert ms.loc["fake/conformist", "toward_share"] > ms.loc["fake/contrarian", "toward_share"]


def test_change_points_finds_a_jump():
    x = np.array([0, 0, 0, 0, 1.8, 1.8, 1.8, 1.8])
    assert 4 in change_points(x, penalty=0.5)
    assert change_points(np.array([1.0, 1.0])) == []
