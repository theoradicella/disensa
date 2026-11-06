"""`seed` fixes the world (network, x0, models); `replicate` the dynamics (seeding.py)."""

import asyncio
import json

import networkx as nx
import numpy as np

from disensa.checks import config_checks
from disensa.dynamics import make_dynamics
from disensa.engine import run
from disensa.network import build_network, initial_opinions
from disensa.runs import load_run, new_run_dir
from disensa.seeding import interactions_rng, opinions_rng, sampling_seed

NUMERIC = {"type": "numeric", "rule": "degroot"}


def world(cfg):
    g = build_network(cfg.network, seed=cfg.seed)
    x0 = initial_opinions(cfg.opinions, g.number_of_nodes(), seed=cfg.seed)
    return g, x0, make_dynamics(cfg, g)


def mixed(make_cfg, **kw):
    dyn = make_cfg().dynamics.model_dump(mode="json")
    dyn["models"] = [{"name": "fake/stubborn"}, {"name": "fake/conformist"}]
    return make_cfg(dynamics={**dyn, **kw.pop("dyn", {})}, **kw)


def test_same_seed_same_world_for_numeric_and_llm(make_cfg):
    g1, x1, _ = world(make_cfg(dynamics=NUMERIC))
    g2, x2, _ = world(make_cfg())
    assert nx.utils.graphs_equal(g1, g2) and np.array_equal(x1, x2)
    # two LLM configs that differ in everything but the world place the models the same way
    _, _, a = world(mixed(make_cfg))
    _, _, b = world(mixed(make_cfg, replicate=2, dyn={"statement": "Cats are better."}))
    assert a.labels() == b.labels()


def test_replicates_share_the_world_but_not_the_dynamics(make_cfg):
    g1, x1, a = world(mixed(make_cfg))
    g2, x2, b = world(mixed(make_cfg, replicate=1))
    assert nx.utils.graphs_equal(g1, g2) and np.array_equal(x1, x2)
    assert a.labels() == b.labels()
    assert a.sampling_seed != b.sampling_seed
    draws = lambda d: [d.interactions(i) for i in range(d.n)]  # noqa: E731
    for d in (a, b):  # give everyone a message, so interactions can be drawn
        for i, s in enumerate(d.state):
            s.message = f"m{i}"
    assert draws(a) != draws(b)


def test_same_seed_and_replicate_same_run(make_cfg):
    def strip(h):
        return [{k: v for k, v in r.items() if k != "seconds"} for r in h]

    a = asyncio.run(run(make_cfg(replicate=3)))
    b = asyncio.run(run(make_cfg(replicate=3)))
    c = asyncio.run(run(make_cfg(replicate=4)))
    assert strip(a) == strip(b) and strip(a) != strip(c)
    assert strip(a)[0] == strip(c)[0]  # step 0: same world, same first messages


def test_numeric_runs_ignore_the_replicate(make_cfg):
    def strip(h):
        return [{k: v for k, v in r.items() if k != "seconds"} for r in h]

    a = asyncio.run(run(make_cfg(dynamics=NUMERIC)))
    b = asyncio.run(run(make_cfg(dynamics=NUMERIC, replicate=5)))
    assert strip(a) == strip(b)
    assert [c.field for c in config_checks(make_cfg(dynamics=NUMERIC, replicate=5))] == [
        "replicate"
    ]
    assert config_checks(make_cfg(replicate=5)) == []


def test_opinions_and_interactions_no_longer_share_numbers():
    corr = []
    for seed in range(50):
        a = opinions_rng(seed).random(200)
        b = interactions_rng(seed, 0).random(200)
        assert not np.allclose(a, b)
        corr.append(np.corrcoef(a, b)[0, 1])
    assert abs(np.mean(corr)) < 0.05
    assert len({sampling_seed(s, r) for s in range(20) for r in range(5)}) == 100


def test_negative_seeds_work():
    assert not np.allclose(opinions_rng(-3).random(5), opinions_rng(3).random(5))


def test_run_folder_records_the_scheme_and_replicate(make_cfg):
    cfg = make_cfg(replicate=2)
    run_dir = new_run_dir(cfg)
    assert run_dir.name.endswith("_s3_r2")
    assert new_run_dir(make_cfg()).name.endswith("_s3")
    asyncio.run(run(cfg, run_dir))
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["seeding"] == 2 and meta["config"]["replicate"] == 2
    assert load_run(run_dir).steps == cfg.steps
