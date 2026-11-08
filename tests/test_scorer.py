"""The offline scorer (scorer.py): rows, keys, reruns, bad replies, the lab."""

import asyncio

import pytest
import yaml
from typer.testing import CliRunner

from disensa import scorer
from disensa.app.payload import build_payload
from disensa.cli import app
from disensa.engine import run
from disensa.llm import LLMClient
from disensa.llm.prompts import SCORER_TEMPLATE
from disensa.runs import load_run, new_run_dir


@pytest.fixture
def run_dir(make_cfg):
    cfg = make_cfg()
    d = new_run_dir(cfg)
    asyncio.run(run(cfg, d))
    return d


def score(run_dir, model, field="message", tmp=None):
    return asyncio.run(scorer.score_run(run_dir, model, field, client=LLMClient(tmp)))


def test_scores_are_written_and_keyed(run_dir, tmp_path):
    rows = score(run_dir, "fake/scorer", tmp=tmp_path / "c")
    r = load_run(run_dir)
    assert len(rows) == len(r.events) == 20 * 5
    s = r.scores
    assert list(s.columns) == scorer.COLUMNS
    assert set(s["field"]) == {"message"} and set(s["scorer_model"]) == {"fake/scorer"}
    assert s["score"].notna().all() and s["error"].isna().all()
    # the fake scorer reads the position in the message: close to the self-rating
    ops = r.opinions.set_index(["step", "agent"])["opinion"]
    for row in s.itertuples():
        assert abs(row.score - ops[(row.step, row.agent)]) <= 0.6


def test_scorers_and_fields_coexist_and_reruns_replace(run_dir, tmp_path):
    score(run_dir, "fake/scorer", tmp=tmp_path / "c")
    score(run_dir, "fake/scorer", "reasoning", tmp=tmp_path / "c")
    score(run_dir, "fake/unclear", tmp=tmp_path / "c")
    score(run_dir, "fake/unclear", tmp=tmp_path / "c")  # again: replaces, never doubles
    s = load_run(run_dir).scores
    counts = s.groupby(["scorer_model", "field"]).size().to_dict()
    assert counts == {
        ("fake/scorer", "message"): 100,
        ("fake/scorer", "reasoning"): 100,
        ("fake/unclear", "message"): 100,
    }
    reasoning = s[s["field"] == "reasoning"]
    assert (reasoning[reasoning["step"] == 0]["error"] == "no reasoning").all()  # step 0
    unclear = s[s["scorer_model"] == "fake/unclear"]
    assert unclear["score"].isna().all() and unclear["error"].isna().all()


@pytest.mark.parametrize("model,needle", [("fake/badjson", "no JSON"), ("fake/offscale", "7")])
def test_unparseable_scorer_replies_are_logged_not_fatal(run_dir, tmp_path, model, needle):
    rows = score(run_dir, model, tmp=tmp_path / "c")
    assert rows["score"].isna().all() and rows["error"].str.contains(needle).all()


def test_the_lab_shows_scores_next_to_the_self_rating(run_dir, tmp_path):
    score(run_dir, "fake/scorer", tmp=tmp_path / "c")
    p = build_payload(load_run(run_dir))
    assert p["agent_steps"][2][5]["scores"].keys() == {"fake/scorer · message"}
    assert p["agent_steps"][0][5]["scores"]["fake/scorer · message"] in (-2, -1, 0, 1, 2)


def test_disensa_score_warns_about_self_rating(run_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = CliRunner().invoke(app, ["score", str(run_dir), "--model", "fake/bounded"])
    assert res.exit_code == 0, res.output
    assert "self-rating bias" in res.output and "100 texts, 100 scored" in res.output
    res = CliRunner().invoke(app, ["score", str(run_dir), "--model", "fake/x", "--field", "x"])
    assert res.exit_code != 0


def test_disensa_score_on_a_sweep_folder(make_cfg, tmp_path, monkeypatch):
    from disensa import sweep as sw

    base = make_cfg(steps=1).model_dump(mode="json")
    path = tmp_path / "s.yaml"
    out = tmp_path / "runs"
    grid = [{"seed": [1, 2]}, {"dynamics": [{"type": "numeric", "rule": "degroot"}]}]
    path.write_text(
        yaml.safe_dump({"name": "s", "base": base, "grids": grid, "output_dir": str(out)})
    )
    spec, b = sw.load_sweep(path)
    runs = sw.expand(spec, b)
    asyncio.run(sw.run_sweep(spec, runs, runs, path, echo=lambda *_: None))
    monkeypatch.chdir(tmp_path)
    res = CliRunner().invoke(app, ["score", str(out / "s"), "--model", "fake/scorer"])
    assert res.exit_code == 0, res.output
    assert res.output.count("-> scores.parquet") == 2  # the numeric run is not scored
    resume = ["score", str(out / "s"), "--model", "fake/scorer", "--resume", "--cache-dir", "c"]
    res = CliRunner().invoke(app, resume)
    assert res.exit_code == 0 and res.output.count("already scored, skipped") == 2
    res = CliRunner().invoke(app, [*resume[:3], "fake/other", *resume[4:]])
    assert res.output.count("-> scores.parquet") == 2 and (tmp_path / "c").is_dir()


def test_the_scorer_prompt_is_quotable():
    assert "{text}" in SCORER_TEMPLATE and "{statement}" in SCORER_TEMPLATE
    assert "tweet" not in SCORER_TEMPLATE and "post" not in SCORER_TEMPLATE.lower()
