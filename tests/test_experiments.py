"""The experiment files of configs/experiments/ stay consistent with each other.

They are run on a cluster by someone else, a sweep at a time, and compared afterwards, so a
statement typed slightly differently in one file, a forgotten statement or a model with
other sampling settings would only show up in the analysis. These tests catch it first.
"""

import csv
from pathlib import Path

import pytest

from disensa import sweep as sw

ROOT = Path("configs/experiments")
TOPICS = list(csv.DictReader(open(ROOT / "topics.csv", encoding="utf-8")))
STATEMENTS = [t["statement"] for t in TOPICS]
SWEEPS = sorted(p for p in ROOT.rglob("*.yaml") if p.name != "base.yaml")
LLM_SWEEPS = [p for p in SWEEPS if p.parent.name in ("pilot", "core", "main")]
SAMPLING = {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "think": False}


def expand(path):
    spec, base = sw.load_sweep(path)
    return spec, sw.expand(spec, base)


def test_topics_file():
    ids = [t["id"] for t in TOPICS]
    assert len(set(ids)) == len(ids) == len(set(STATEMENTS)) == 16
    for t in TOPICS:  # two framings of each topic, and only factual ones have a truth value
        assert t["framing"] in ("a", "b") and t["statement"].strip() == t["statement"]
        assert (t["true"] in ("yes", "no")) == (t["type"] == "factual")
    pairs = {}
    for t in TOPICS:
        pairs.setdefault((t["type"], t["topic"]), []).append(t["framing"])
    assert all(sorted(f) == ["a", "b"] for f in pairs.values())
    types = [t["type"] for t in TOPICS]
    assert {k: types.count(k) for k in set(types)} == dict.fromkeys(
        ("political", "moral", "factual", "trivial"), 4
    )


@pytest.mark.parametrize("path", SWEEPS, ids=str)
def test_every_sweep_expands_and_names_itself_after_its_file(path):
    spec, runs = expand(path)
    assert runs
    prefix = f"{path.parent.name}_" if path.parent.name in ("pilot", "core") else ""
    assert spec.name == prefix + path.stem  # runs/<name>/ then says which file made it


@pytest.mark.parametrize("path", LLM_SWEEPS, ids=str)
def test_llm_sweeps_use_one_vllm_model_and_only_the_topics(path):
    _, runs = expand(path)
    models = {m.name for r in runs for m in r.config.dynamics.models}
    assert len(models) == 1 and next(iter(models)).startswith("hosted_vllm/")
    for r in runs:
        d = r.config.dynamics
        assert d.statement in STATEMENTS
        assert {k: getattr(d.models[0], k) for k in SAMPLING} == SAMPLING
        assert (r.config.network.n, r.config.steps) == (50, 30)
        assert r.config.opinions.distribution == "uniform"


def test_the_grids_cover_every_statement_world_and_topology():
    for path in LLM_SWEEPS:
        _, runs = expand(path)
        used = {r.config.dynamics.statement for r in runs}
        if path.parent.name == "pilot":
            assert sorted(used) == sorted(t["statement"] for t in TOPICS if t["framing"] == "a")
            assert {r.config.seed for r in runs} == {100}  # never one of the experiment worlds
            continue
        assert used == set(STATEMENTS)
        assert {r.config.seed for r in runs} == {1, 2, 3, 4, 5}
        topologies = {r.config.network.topology for r in runs}
        if path.stem.startswith("control"):
            assert {r.config.dynamics.interactions.max for r in runs} == {0}
        else:
            assert topologies == {"watts_strogatz", "barabasi_albert", "erdos_renyi"}
            assert len(runs) == 5 * 3 * 16


def test_every_core_sweep_has_the_same_runs_apart_from_the_model():
    def key(r):
        c = r.config.model_dump(mode="json")
        c["dynamics"].pop("models")
        c.pop("name"), c.pop("output_dir")
        return str(c)

    keys = [sorted(map(key, expand(p)[1])) for p in LLM_SWEEPS if p.parent.name == "core"]
    assert len(keys) >= 3 and all(k == keys[0] for k in keys)


def test_numeric_baselines_share_the_worlds():
    _, runs = expand(ROOT / "numeric_baselines.yaml")
    assert {r.config.seed for r in runs} == {1, 2, 3, 4, 5}
    assert len(runs) == 5 * 3 * 3 and all(r.config.dynamics.type == "numeric" for r in runs)


# ---------------------------------------------------------------- the cluster scripts


def _script(name):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, f"scripts/indaco/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_preflight_finds_the_one_model_and_refuses_placeholders(tmp_path):
    preflight = _script("preflight")
    for path in LLM_SWEEPS:
        assert "/" in preflight.model_of(str(path))
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        f"name: bad\nbase: {(ROOT / 'base.yaml').resolve()}\ngrids:\n  - seed: [1]\n"
        "    dynamics.models: [[{name: hosted_vllm/x/y}]]\n"
    )
    with pytest.raises(SystemExit) as e:  # the base's statement was never replaced
        preflight.model_of(str(bad))
    assert e.value.code == 1


def test_status_summarizes_a_sweep_folder(tmp_path):
    import asyncio

    status = _script("status")
    base = {
        "name": "b",
        "network": {"topology": "watts_strogatz", "n": 10, "k": 4, "beta": 0.2},
        "steps": 2,
        "dynamics": {
            "type": "llm",
            "models": [{"name": "fake/bounded"}],
            "cache_dir": str(tmp_path / "cache"),
        },
    }
    spec = sw.SweepFile(name="st", base=base, grids=[{"seed": [1, 2]}], output_dir=tmp_path)
    runs = sw.expand(spec, base)
    asyncio.run(sw.run_sweep(spec, runs, runs, echo=lambda *a: None))
    text = status.sweep_status(tmp_path / "st", flag_above=5, parallel=1)
    assert "runs: 2/2 finished, 0 failed" in text and "failed 0 (0.00%)" in text
