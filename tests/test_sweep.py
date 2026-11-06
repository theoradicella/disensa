"""`disensa sweep`: grid expansion, validation, resume, shards, failures, dry run, the lab."""

import asyncio
import json
import urllib.parse

import pytest
import yaml
from typer.testing import CliRunner

from disensa import sweep as sw
from disensa.app.server import RunManager
from disensa.cli import app


@pytest.fixture
def base(tmp_path):
    return {
        "name": "base",
        "seed": 1,
        "network": {"topology": "watts_strogatz", "n": 12, "k": 4, "beta": 0.1},
        "steps": 2,
        "dynamics": {
            "type": "llm",
            "models": [{"name": "fake/bounded"}],
            "cache_dir": str(tmp_path / "cache"),
            "max_concurrency": 32,
        },
    }


def write_sweep(tmp_path, base, grids, name="s"):
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    path = tmp_path / f"{name}.yaml"
    doc = {"name": name, "base": "base.yaml", "grids": grids, "output_dir": str(tmp_path / "runs")}
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


def plan(path, cache_dir=None):
    spec, b = sw.load_sweep(path)
    return spec, sw.expand(spec, b, cache_dir)


def test_grids_are_a_union_of_products(tmp_path, base):
    path = write_sweep(
        tmp_path,
        base,
        [
            {"seed": [1, 2, 3], "dynamics.statement": ["A short one.", "Another, longer one."]},
            {
                "seed": [1, 2],
                "dynamics": [
                    {"type": "numeric", "rule": "degroot"},
                    {"type": "numeric", "rule": "bounded_confidence"},
                ],
                "dynamics.epsilon": [0.5, 1.0],
            },
        ],
    )
    _, runs = plan(path)
    # 3 x 2 LLM runs, 2 x 2 x 2 numeric ones (DeGroot ignores epsilon, but it is still a
    # config field, so its two epsilon values are two configs)
    assert len(runs) == 6 + 8
    llm = [r for r in runs if r.config.dynamics.type == "llm"]
    assert {(r.config.seed, r.config.dynamics.statement) for r in llm} == {
        (s, st) for s in (1, 2, 3) for st in ("A short one.", "Another, longer one.")
    }
    bc = [r for r in runs if getattr(r.config.dynamics, "rule", "") == "bounded_confidence"]
    assert sorted({r.config.dynamics.epsilon for r in bc}) == [0.5, 1.0]
    assert all(r.config.dynamics.mu == 0.5 for r in bc)  # replaced, then defaults
    names = [r.name for r in runs]
    assert len(set(names)) == len(names)
    assert "s__seed=2__statement=2" in names  # long strings become their position
    assert "s__seed=1__dynamics=bounded_confidence__epsilon=0.5" in names
    assert all(r.config.output_dir == tmp_path / "runs" / "s" for r in runs)
    assert sum(r.llm_calls for r in runs) == 6 * 12 * 3


def test_inline_base_and_no_grids(tmp_path, base):
    path = tmp_path / "one.yaml"
    path.write_text(yaml.safe_dump({"name": "one", "base": base}))
    _, runs = plan(path)
    assert [r.name for r in runs] == ["one"]


def test_invalid_combinations_stop_everything_before_running(tmp_path, base):
    path = write_sweep(tmp_path, base, [{"network.k": [4, 5], "steps": [0, 2]}])
    with pytest.raises(sw.SweepError) as e:
        plan(path)
    assert "3 invalid combination(s)" in str(e.value) and "even k" in str(e.value)
    res = CliRunner().invoke(app, ["sweep", str(path)])
    assert res.exit_code == 2 and not (tmp_path / "runs").exists()


def test_bad_sweep_files(tmp_path, base):
    with pytest.raises(sw.SweepError):
        plan(write_sweep(tmp_path, base, [{"seed": []}]))
    with pytest.raises(sw.SweepError):
        plan(write_sweep(tmp_path, base, [], name="bad name"))


def test_shards_are_disjoint_and_complete(tmp_path, base):
    _, runs = plan(write_sweep(tmp_path, base, [{"seed": list(range(11))}]))
    for n in (1, 3, 4, 11, 15):
        parts = [sw.shard(runs, (i, n)) for i in range(1, n + 1)]
        idx = [r.index for p in parts for r in p]
        assert sorted(idx) == list(range(11)) and len(set(idx)) == 11
    assert sw.parse_shard("2/4") == (2, 4)
    for bad in ("0/4", "5/4", "x", "1-4"):
        with pytest.raises(sw.SweepError):
            sw.parse_shard(bad)


def test_cache_dir_does_not_change_what_a_run_is(tmp_path, base):
    path = write_sweep(tmp_path, base, [{"seed": [1, 2]}])
    _, a = plan(path)
    _, b = plan(path, tmp_path / "elsewhere")
    assert [r.hash for r in a] == [r.hash for r in b]
    assert b[0].config.dynamics.cache_dir == tmp_path / "elsewhere"


def run_cli(*args):
    return CliRunner().invoke(app, ["sweep", *map(str, args)])


def test_run_resume_and_sweep_json(tmp_path, base):
    path = write_sweep(tmp_path, base, [{"seed": [1, 2, 3]}])
    res = run_cli(path)
    assert res.exit_code == 0, res.output
    root = tmp_path / "runs" / "s"
    doc = json.loads((root / "sweep.json").read_text())
    assert [r["status"] for r in doc["runs"]] == ["finished"] * 3
    assert all(r["folder"] and r["report"]["calls"] == 12 * 3 for r in doc["runs"])
    assert "run report" in res.output and "status" in res.output  # the end table

    # mark one as cancelled: only that one runs again
    folder = root / doc["runs"][1]["folder"]
    meta = json.loads((folder / "meta.json").read_text())
    meta["status"] = "cancelled"
    (folder / "meta.json").write_text(json.dumps(meta))
    res = run_cli(path)
    assert res.exit_code == 0, res.output
    assert res.output.count("already finished (") == 2
    assert len([d for d in root.iterdir() if d.is_dir()]) == 4
    doc = json.loads((root / "sweep.json").read_text())
    assert [r["status"] for r in doc["runs"]] == ["finished"] * 3

    res = run_cli(path, "--dry-run")
    assert "3 already finished, 0 to run" in res.output
    assert "estimated LLM calls: 0 to make (108 in all)" in res.output


def test_a_failed_run_does_not_stop_the_sweep(tmp_path, base):
    path = write_sweep(
        tmp_path,
        base,
        [{"dynamics.models": [[{"name": "fake/nobelief"}], [{"name": "fake/bounded"}]]}],
    )
    res = run_cli(path)
    assert res.exit_code == 1  # something failed
    doc = json.loads((tmp_path / "runs" / "s" / "sweep.json").read_text())
    assert [r["status"] for r in doc["runs"]] == ["failed", "finished"]
    assert doc["runs"][0]["report"]["failed_pct"] > 20
    assert "failed: step 1: 12/12 replies failed" in res.output


def test_dry_run_counts_llm_calls(tmp_path, base):
    path = write_sweep(
        tmp_path,
        base,
        [{"seed": [1, 2]}, {"seed": [1, 2], "dynamics": [{"type": "numeric", "rule": "degroot"}]}],
    )
    res = run_cli(path, "--dry-run", "--shard", "1/2")
    assert res.exit_code == 0, res.output
    assert "s: 4 runs, shard 1/2: 2 here (1 LLM)" in res.output
    assert "estimated LLM calls: 36 to make" in res.output
    assert not (tmp_path / "runs").exists()


def test_the_lab_lists_sweep_runs(tmp_path, base):
    path = write_sweep(tmp_path, base, [{"seed": [1, 2]}])
    spec, runs = plan(path)
    asyncio.run(sw.run_sweep(spec, runs, runs, path, echo=lambda *_: None))
    manager = RunManager(tmp_path / "runs")
    listed = manager.list()
    assert len(listed) == 2 and all(r["sweep"] == "s" and r["id"].startswith("s/") for r in listed)
    assert all(r["status"] == "done" for r in listed)
    run_id = listed[0]["id"]
    assert manager.run_dir(run_id) is not None
    assert manager.payload(run_id)["report"]["calls"] == 36
    for bad in ("s/../s", "../x", "s/.hidden", "a/b/c", ""):
        assert manager.run_dir(bad) is None
    assert urllib.parse.quote(run_id, safe="") != run_id  # the lab encodes the slash


def test_the_example_sweeps_expand():
    for name in ("numeric_bc_epsilon", "numeric_fj_stubbornness", "topics_core"):
        spec, runs = plan(f"configs/sweeps/{name}.yaml")
        assert runs and spec.name == name
    _, bc = plan("configs/sweeps/numeric_bc_epsilon.yaml")
    assert len(bc) == 20 * 10
