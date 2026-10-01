import asyncio
import json

import numpy as np
import pandas as pd
import pytest

from disensa.config import ModelSpec
from disensa.dynamics.llm_agent import assign_models
from disensa.engine import run
from disensa.llm.prompts import stance_label
from disensa.runs import load_run, new_run_dir

NUMERIC = {"type": "numeric", "rule": "bounded_confidence", "epsilon": 0.8, "mu": 0.5}


def test_numeric_run_writes_a_run_folder(make_cfg):
    cfg = make_cfg(dynamics=NUMERIC, steps=10)
    run_dir = new_run_dir(cfg)
    history = asyncio.run(run(cfg, run_dir))
    assert [h["step"] for h in history] == list(range(11))
    r = load_run(run_dir)
    assert r.steps == 10 and r.opinion_matrix().shape == (11, 20)
    assert set(r.agents["model"]) == {"numeric/bounded_confidence"}
    assert not (run_dir / "events.jsonl").exists()  # numbers only, no text
    assert json.loads((run_dir / "meta.json").read_text())["finished"]


def test_same_seed_same_run(make_cfg):
    a = asyncio.run(run(make_cfg()))
    b = asyncio.run(run(make_cfg()))
    assert a == b


def test_llm_run_logs_every_agent_every_day(make_cfg):
    cfg = make_cfg()
    run_dir = new_run_dir(cfg)
    asyncio.run(run(cfg, run_dir))
    r = load_run(run_dir)
    assert len(r.events) == 20 * (cfg.steps + 1)
    step1 = [e for e in r.events if e["step"] == 1]
    assert all(e["message"] and e["prompt"] and e["error"] is None for e in step1)
    assert all(1 <= len(e["interactions"]) <= 3 for e in step1)
    ops = r.opinion_matrix()
    assert ops.min() >= -2 and ops.max() <= 2 and not np.allclose(ops[0], ops[-1])


def test_update_is_synchronous(make_cfg):
    """Every interaction carries the author's message from step t-1, never from step t."""
    cfg = make_cfg()
    run_dir = new_run_dir(cfg)
    asyncio.run(run(cfg, run_dir))
    events = pd.DataFrame(load_run(run_dir).events)
    for t in range(1, cfg.steps + 1):
        said_before = dict(
            events[events["step"] == t - 1][["agent", "message"]].itertuples(index=False)
        )
        for interactions in events[events["step"] == t]["interactions"]:
            for it in interactions:
                assert it["message"] == said_before[it["author"]]


def test_bad_model_reply_never_crashes_a_run(make_cfg, monkeypatch):
    from disensa.llm.client import LLMClient

    async def broken(self, model, messages, **kw):
        return {"reasoning": "hm"}, '{"reasoning": "hm"}'  # no belief, no message

    monkeypatch.setattr(LLMClient, "complete_json", broken)
    cfg = make_cfg()
    history = asyncio.run(run(cfg))
    assert history[-1]["errors"] == 20
    assert history[0]["mean_opinion"] == history[-1]["mean_opinion"]  # nobody moved


def test_models_are_split_by_weight():
    models = [ModelSpec(name="a", weight=3), ModelSpec(name="b", weight=1)]
    names = [m.name for m in assign_models(models, 20, seed=1)]
    assert names.count("a") == 15 and names.count("b") == 5
    assert names == [m.name for m in assign_models(models, 20, seed=1)]


def test_mixed_models_get_a_mean_per_model(make_cfg):
    dyn = make_cfg().dynamics.model_dump(mode="json")
    dyn["models"] = [{"name": "fake/stubborn"}, {"name": "fake/conformist"}]
    history = asyncio.run(run(make_cfg(dynamics=dyn)))
    assert "mean_opinion[fake/stubborn]" in history[-1]


def test_llm_dynamics_need_a_model(make_cfg):
    with pytest.raises(ValueError):
        make_cfg(dynamics={"type": "llm", "models": []})


@pytest.mark.parametrize(
    "v,label",
    [(-2, "strongly oppose"), (0.2, "are neutral / undecided about"), (1.4, "somewhat support")],
)
def test_stance_label(v, label):
    assert stance_label(v) == label
