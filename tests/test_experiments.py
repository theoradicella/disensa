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
LLM_SWEEPS = [p for p in SWEEPS if p.parent.name in ("pilot", "core", "main", "followup")]
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
        topologies = {r.config.network.topology for r in runs}
        if path.parent.name == "followup" and path.stem.startswith("replicate"):
            assert {r.config.seed for r in runs} == {1}  # one world: a small second replicate
            assert {r.config.replicate for r in runs} == {1}
            assert topologies == {"watts_strogatz", "barabasi_albert", "erdos_renyi"}
            assert len(runs) == 3 * 16
            continue
        assert {r.config.seed for r in runs} == {1, 2, 3, 4, 5}
        # only the controls of followup/ draw a new sampling seed at every step
        per_step = {r.config.dynamics.seed_per_step for r in runs}
        assert per_step == {path.parent.name == "followup"}
        if path.stem.startswith("control"):
            assert {r.config.dynamics.interactions.max for r in runs} == {0}
            assert len(runs) == 5 * 16
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


def _paired(r):
    """What pairs two runs: everything but the model, the replicate and the names."""
    c = r.config.model_dump(mode="json")
    c["dynamics"].pop("models")
    for k in ("name", "output_dir", "replicate"):
        c.pop(k)
    return str(c)


def test_every_followup_sweep_pairs_with_core_runs_of_its_model():
    """A control is a core run without interactions; a replicate, the core run again."""
    for path in sorted((ROOT / "followup").glob("*.yaml")):
        model = path.stem.removeprefix("control_v2_").removeprefix("replicate_")
        _, runs = expand(path)
        _, core = expand(ROOT / "core" / f"{model}.yaml")
        assert {m.name for r in runs for m in r.config.dynamics.models} == {
            m.name for r in core for m in r.config.dynamics.models
        }
        if path.stem.startswith("replicate"):
            assert {_paired(r) for r in runs} <= {_paired(r) for r in core}
        else:  # same worlds and statements as core's Watts-Strogatz runs (the network is unused)
            ws = {
                (r.config.seed, r.config.dynamics.statement)
                for r in core
                if r.config.network.topology == "watts_strogatz"
            }
            assert {(r.config.seed, r.config.dynamics.statement) for r in runs} == ws
    assert len(list((ROOT / "followup").glob("control_v2_*.yaml"))) == 4


def test_the_experiment_runs_keep_their_hashes():
    """Hashes of runs done on INDACO (archive of 2026-11-11). A change that alters them would
    make run_all.sh see every finished run as new and run the whole experiment again."""
    done = {
        "pilot/llama31_8b.yaml": {0: "0989c1804a08e02c", 7: "0ea522b5a19b9916"},
        "core/mistral03_7b.yaml": {0: "59cf80a27dd08d1c", 239: "094be6f319be2704"},
        "main/control_llama31_8b.yaml": {0: "c586ca5cd733db9a", 79: "bfaee060859cbf85"},
        "main/replicate_llama31_8b.yaml": {0: "2a38cda34709ce09", 239: "9839b8ec1055fb0e"},
    }
    for file, hashes in done.items():
        _, runs = expand(ROOT / file)
        assert {i: runs[i].hash for i in hashes} == hashes, file


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


def test_every_topic_type_is_a_quarter_of_each_sweep():
    """run_all.sh <type> runs one topic type at a time; together they must cover everything."""
    preflight = _script("preflight")
    types = ("political", "moral", "factual", "trivial")
    assert sorted(s for t in types for s in preflight.statements_of(t)) == sorted(STATEMENTS)
    for path in LLM_SWEEPS:
        if path.parent.name == "pilot":
            continue
        _, runs = expand(path)
        parts = [sw.select(runs, preflight.statements_of(t)) for t in types]
        assert all(len(p) == len(runs) // 4 for p in parts), path
        assert sorted(r.index for p in parts for r in p) == list(range(len(runs))), path
    with pytest.raises(SystemExit):
        preflight.statements_of("sports")


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
