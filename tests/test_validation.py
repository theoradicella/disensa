"""Human validation tooling (validation.py): sampling, keys, alpha, agreement."""

import asyncio
import math

import pandas as pd
import pytest
from typer.testing import CliRunner

from disensa import scorer, validation
from disensa.cli import app
from disensa.engine import run
from disensa.llm import LLMClient
from disensa.runs import new_run_dir


@pytest.fixture
def runs(make_cfg, tmp_path):
    dirs = []
    for seed, dist in [(1, "uniform"), (2, "bimodal")]:
        cfg = make_cfg(seed=seed, steps=4, opinions={"distribution": dist, "mean": 1.5, "std": 0.3})
        d = new_run_dir(cfg)
        asyncio.run(run(cfg, d))
        asyncio.run(scorer.score_run(d, "fake/scorer", client=LLMClient(tmp_path / "c")))
        dirs.append(d)
    return dirs


def test_export_is_stratified_and_blind(runs, tmp_path):
    out = tmp_path / "validation" / "t"
    sample, key = validation.export(runs, out, n=40, seed=0)
    assert list(sample.columns) == ["item_id", "statement", "message", "rating"]
    assert (sample["rating"] == "").all() and sample["item_id"].is_unique
    assert "self_rating" not in sample.columns and "model" not in sample.columns
    assert list(key["item_id"]) == list(sample["item_id"])  # same items, same order
    assert {"run", "step", "agent", "model", "self_rating", "score[fake/scorer]"} <= set(key)
    assert (key["step"] >= 1).all()  # step 0 opinions are given, not self-rated
    pool = validation.collect(runs)
    available = pool["self_rating"].map(validation._level).value_counts().to_dict()
    expected = validation.stratified_counts(available, 40)
    got = key["self_rating"].map(validation._level).value_counts().to_dict()
    assert got == {k: v for k, v in expected.items() if v}
    for f in ("sample.csv", "key.csv", "INSTRUCTIONS.md"):
        assert (out / f).exists()
    assert (
        "What is the author's position on this statement?" in (out / "INSTRUCTIONS.md").read_text()
    )
    # the same seed gives the same sample
    again, _ = validation.export(runs, tmp_path / "again", n=40, seed=0)
    assert again.equals(sample)


def test_stratified_counts():
    assert validation.stratified_counts({-2: 50, -1: 50, 0: 50, 1: 50, 2: 50}, 100) == {
        -2: 20,
        -1: 20,
        0: 20,
        1: 20,
        2: 20,
    }
    # a level with too few items gives its share to the others
    got = validation.stratified_counts({-2: 3, -1: 50, 0: 50, 1: 50, 2: 50}, 100)
    assert (
        got[-2] == 3
        and sum(got.values()) == 100
        and max(got.values()) - min(v for k, v in got.items() if k != -2) <= 1
    )
    assert sum(validation.stratified_counts({0: 5, 1: 2}, 100).values()) == 7


def test_krippendorff_alpha_on_the_published_example():
    """Krippendorff (2011), example C: 4 observers, 12 units, alpha_ordinal = 0.815."""
    n = None
    data = [
        [1, 2, 3, 3, 2, 1, 4, 1, 2, n, n, n],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, n, 3],
        [n, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, n],
        [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, n],
    ]
    assert validation.krippendorff_alpha_ordinal(data) == pytest.approx(0.8154, abs=1e-4)
    assert validation.krippendorff_alpha_ordinal([[1, 2, 0], [1, 2, 0]]) == 1.0
    assert math.isnan(validation.krippendorff_alpha_ordinal([[1, 1], [1, 1]]))


def test_majority_and_agreement_on_a_toy_case():
    assert validation.majority([1, 1, 2]) == 1
    assert validation.majority([-2, 0, 2]) == 0  # no majority: the median
    assert validation.majority([1, 2]) == 1  # median 1.5, halves toward 0
    assert validation.majority([-1, -2]) == -1
    a = validation.agreement([1.2, -0.4, 2.0, None], [1, 0, 0, 1])
    assert a["exact_pct"] == 50.0 and a["within_1_pct"] == 50.0
    assert a["mean_abs_diff"] == pytest.approx((0.2 + 0.4 + 2.0) / 3, abs=1e-3)
    assert a["confusion"]["1"]["1"] == 1 and a["confusion"]["1"]["unclear"] == 1
    assert a["confusion"]["0"]["2"] == 1


def test_report_end_to_end(runs, tmp_path):
    out = tmp_path / "v"
    validation.export(runs, out, n=30, seed=1)
    key = pd.read_csv(out / "key.csv")
    sample = pd.read_csv(out / "sample.csv")
    # three raters: two agree with the self-rating, one is always off by one, one cell unclear
    truth = key["self_rating"].map(validation._level)
    for name, shift in [("ana", 0), ("ben", 0), ("cleo", 1)]:
        ratings = [str(min(t + shift, 2)) for t in truth]
        ratings[0] = "unclear" if name == "cleo" else ratings[0]
        sample.assign(rating=ratings).to_csv(out / f"ratings_{name}.csv", index=False)
    res = CliRunner().invoke(app, ["validation", "report", str(out)])
    assert res.exit_code == 0, res.output
    r = validation.report(out)
    assert r["raters"] == ["ana", "ben", "cleo"] and r["items"] == 30
    assert r["ratings_missing"] == 1
    assert r["measures"]["self_rating"]["exact_pct"] == 100.0
    assert set(r["measures"]) == {"self_rating", "fake/scorer"}
    assert 0 < r["krippendorff_alpha_ordinal"] < 1
    assert (out / "report.md").read_text().startswith("# Human validation report")


def test_cli_export(runs, tmp_path):
    out = tmp_path / "cli"
    res = CliRunner().invoke(
        app, ["validation", "export", *map(str, runs), "--n", "10", "--out", str(out)]
    )
    assert res.exit_code == 0, res.output
    assert len(pd.read_csv(out / "sample.csv")) == 10
