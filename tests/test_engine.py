import json

import numpy as np
import pandas as pd
import pytest

from agorasim.config import ExperimentConfig
from agorasim.engine import Simulation
from agorasim.llm import LLMClient
from agorasim.personas import generate_personas, stance_label
from agorasim.runs import RunWriter, new_run_dir


def make_cfg(tmp_path, **overrides) -> ExperimentConfig:
    raw = {
        "name": "t",
        "seed": 3,
        "network": {"topology": "watts_strogatz", "n": 20, "k": 4, "beta": 0.1},
        "simulation": {"days": 4, "feed": {"min_posts": 1, "max_posts": 3}},
        "models": [{"name": "fake/bounded"}],
        "scorer": {"name": "fake/scorer"},
        "llm": {"cache_dir": str(tmp_path / "cache"), "max_concurrency": 32},
        "output_dir": str(tmp_path / "runs"),
    }
    for k, v in overrides.items():
        raw[k] = v
    return ExperimentConfig.model_validate(raw)


def test_personas_depend_only_on_seed():
    x = np.linspace(-2, 2, 8)
    assert generate_personas(x, 5) == generate_personas(x, 5)
    assert generate_personas(x, 5) != generate_personas(x, 6)
    assert all(18 <= p.age <= 64 for p in generate_personas(x, 5))


@pytest.mark.parametrize(
    "v,label",
    [(-2, "strongly oppose"), (0.2, "are neutral / undecided about"), (1.4, "somewhat support")],
)
def test_stance_label(v, label):
    assert stance_label(v) == label


def test_classifier_source_requires_scorer(tmp_path):
    with pytest.raises(ValueError):
        make_cfg(tmp_path, scorer=None, simulation={"opinion_source": "classifier"})


def test_recommender_only_shows_similar_neighbours(tmp_path):
    cfg = make_cfg(
        tmp_path,
        simulation={
            "recommender": {"enabled": True, "threshold": 0.5},
            "feed": {"min_posts": 10, "max_posts": 10},
        },
    )
    sim = Simulation(cfg)
    for a in sim.agents:
        a.post = f"post {a.id}"
    ops = sim.opinions()
    for i in range(len(sim.agents)):
        for author, _ in sim.build_feed(i, ops):
            assert author in sim.graph[i]
            assert abs(ops[author] - ops[i]) <= 0.5


async def test_full_run_writes_everything(tmp_path):
    cfg = make_cfg(tmp_path)
    run_dir = new_run_dir(cfg)
    writer = RunWriter(run_dir)
    sim = Simulation(cfg)
    history = await sim.run(writer)
    writer.close()

    assert len(history) == cfg.simulation.days + 1
    ops = pd.read_parquet(run_dir / "opinions.parquet")
    assert len(ops) == 20 * (cfg.simulation.days + 1)
    assert ops["opinion"].between(-2, 2).all()
    events = [json.loads(line) for line in open(run_dir / "events.jsonl")]
    assert len(events) == 20 * (cfg.simulation.days + 1)
    assert all(e["error"] is None for e in events)
    assert all(e["post"] for e in events)
    assert {"agent", "model", "gender", "trait_openness"} <= set(
        pd.read_parquet(run_dir / "agents.parquet").columns
    )


async def test_same_seed_same_run(tmp_path):
    a = Simulation(make_cfg(tmp_path))
    b = Simulation(make_cfg(tmp_path))
    await a.run()
    await b.run()
    assert np.allclose(a.opinions(), b.opinions())


class GarbageClient(LLMClient):
    """A model that never returns JSON."""

    async def complete(self, model, messages, **kw):
        from agorasim.llm import LLMResponse

        return LLMResponse(text="I refuse to use JSON.", model=model)


async def test_bad_model_output_is_logged_not_fatal(tmp_path):
    cfg = make_cfg(tmp_path)
    sim = Simulation(cfg, client=GarbageClient(cache_dir=tmp_path / "c"))
    before = sim.opinions().copy()
    await sim.run()
    assert np.allclose(sim.opinions(), before)  # beliefs unchanged
    assert sim.day_metrics()["errors"] == 20 * (cfg.simulation.days + 1)
