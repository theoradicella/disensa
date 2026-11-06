"""The analysis package: summaries, numeric fits, response curves, figures."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from disensa.analysis import (
    clusters,
    convergence_step,
    fit_run,
    load_runs,
    response_curve,
    simulate_numeric,
    summarize_run,
    wasserstein,
)
from disensa.analysis.fit import grid, world
from disensa.analysis.response import response_observations
from disensa.config import NumericDynamics
from disensa.engine import run
from disensa.runs import load_run, new_run_dir


def make_run(make_cfg, **kw):
    cfg = make_cfg(**kw)
    d = new_run_dir(cfg)
    asyncio.run(run(cfg, d))
    return d


def test_clusters():
    assert clusters([0.0, 0.1, 0.2, 1.5, 1.6, -2.0]) == 3
    assert clusters([0.0, 0.5, 1.0, 1.5]) == 1  # half steps are one cluster at δ = 0.5
    assert clusters([0.0, 0.5, 1.0, 1.5], delta=0.25) == 4
    assert clusters([]) == 0 and clusters([1.0]) == 1


def test_convergence_step():
    X = np.array([[0, 1], [0.5, 0.6], [0.55, 0.56], [0.55, 0.5501], [0.55, 0.5501]])
    assert convergence_step(X, tol=1e-3) == 3
    assert convergence_step(X, tol=0.1) == 1  # step 2 moves by 0.05 only
    assert convergence_step(np.ones((4, 3))) == 0
    assert convergence_step(np.array([[0.0], [1.0], [0.0]])) is None  # still moving


def test_wasserstein():
    assert wasserstein([0, 0, 1], [0, 0, 1]) == 0
    assert wasserstein([0.0], [1.5]) == pytest.approx(1.5)
    assert wasserstein([0, 1], [0, 1, 1, 1]) == pytest.approx(0.25)


def test_simulation_is_the_engine(make_cfg):
    dyn = {"type": "numeric", "rule": "bounded_confidence", "epsilon": 0.6, "mu": 0.5}
    r = load_run(make_run(make_cfg, dynamics=dyn, steps=12))
    g, x0 = world(r)
    sim = simulate_numeric(g, x0, NumericDynamics(**dyn), 12)
    assert np.allclose(sim, r.opinion_matrix())


def test_the_fit_recovers_a_known_rule(make_cfg):
    """An "LLM run" that is really bounded confidence with ε = 0.8 is fitted back to ε ≈ 0.8."""
    dyn = {"type": "numeric", "rule": "bounded_confidence", "epsilon": 0.8, "mu": 0.5}
    net = {"topology": "watts_strogatz", "n": 80, "k": 6, "beta": 0.2}
    r = load_run(make_run(make_cfg, dynamics=dyn, network=net, steps=30))
    cands = grid(epsilons=np.round(np.arange(0.2, 2.01, 0.1), 2), mus=[0.25, 0.5, 1.0])
    best, scanned = fit_run(r, cands)
    w = best[best["measure"] == "wasserstein"].set_index("rule")
    assert abs(w.loc["bounded_confidence", "epsilon"] - 0.8) <= 0.15
    assert w.loc["bounded_confidence", "distance"] < 1e-6
    assert w.loc["nobody_moves", "distance"] > 0.1  # the baseline is clearly worse
    fj = load_run(
        make_run(
            make_cfg,
            dynamics={"type": "numeric", "rule": "friedkin_johnsen", "stubbornness": 2.0},
            network=net,
            steps=30,
        )
    )
    b, _ = fit_run(fj, grid(epsilons=[1.0], mus=[0.5], stubbornness=[0.5, 1.0, 2.0, 4.0]))
    assert (
        b.set_index(["rule", "measure"]).loc[("friedkin_johnsen", "wasserstein"), "stubbornness"]
        == 2.0
    )


def toy_run():
    """Agent 0 hears agents 1 and 2 at step 1; agent 1 hears nobody; agent 2's reply failed."""
    ops = {0: [0.0, 0.5], 1: [1.0, 1.0], 2: [2.0, 2.0]}
    rows = [{"step": t, "agent": a, "opinion": v[t]} for a, v in ops.items() for t in (0, 1)]
    df = pd.DataFrame(rows)
    events = [
        {
            "step": 1,
            "agent": 0,
            "model": "m",
            "error": None,
            "interactions": [{"author": 1, "message": ""}, {"author": 2, "message": ""}],
        },
        {"step": 1, "agent": 1, "model": "m", "error": None, "interactions": []},
        {"step": 1, "agent": 2, "model": "m", "error": "x", "interactions": [{"author": 0}]},
    ]
    return SimpleNamespace(
        path="toy",
        opinions=df,
        events=events,
        scores=None,
        opinion_matrix=lambda: df.pivot(index="step", columns="agent", values="opinion").to_numpy(),
    )


def test_response_curve_binning_on_a_toy_run():
    r = toy_run()
    obs = response_observations(r)
    assert len(obs) == 1  # no interactions and failed replies are left out
    assert obs.iloc[0]["d"] == pytest.approx(1.5) and obs.iloc[0]["delta"] == pytest.approx(0.5)
    curve = response_curve(
        r,
        rules=[
            NumericDynamics(rule="degroot"),
            NumericDynamics(rule="bounded_confidence", epsilon=1.2, mu=1.0),
        ],
    )
    hit = curve[curve["n"] > 0]
    assert len(hit) == 1 and hit.iloc[0]["d_low"] == 1.5 and hit.iloc[0]["p_move"] == 1.0
    assert hit.iloc[0]["pred[degroot]"] == pytest.approx(2 / 3 * 1.5)  # k/(k+1) d
    # bounded confidence with ε = 1.2 only hears agent 1: mean of {1, 0} − 0 = 0.5
    assert hit.iloc[0]["pred[bounded_confidence]"] == pytest.approx(0.5)


def test_response_curve_with_a_scorer(make_cfg, tmp_path):
    from disensa import scorer
    from disensa.llm import LLMClient

    d = make_run(make_cfg)
    with pytest.raises(ValueError, match="disensa score"):
        response_curve(load_run(d), measure="fake/scorer")
    asyncio.run(scorer.score_run(d, "fake/scorer", client=LLMClient(tmp_path / "c")))
    curve = response_curve(load_run(d), measure="fake/scorer")
    assert curve["n"].sum() > 0


def test_summary_and_table(make_cfg):
    d = make_run(make_cfg, dynamics={"type": "numeric", "rule": "degroot"}, steps=60)
    s = summarize_run(load_run(d))
    assert s["clusters"] == 1 and s["convergence_step"] is not None and s["granularity"] is None
    llm = make_run(make_cfg)
    s = summarize_run(load_run(llm))
    assert sum(s["granularity"].values()) == 20 * 4
    t = load_runs(d.parent)
    assert set(t["engine"]) == {"numeric", "llm"} and (t["status"] == "finished").all()


def test_analyze_end_to_end(make_cfg, tmp_path):
    from typer.testing import CliRunner

    from disensa.cli import app

    for eps in (0.4, 1.0):
        for seed in (1, 2):
            make_run(
                make_cfg,
                seed=seed,
                dynamics={"type": "numeric", "rule": "bounded_confidence", "epsilon": eps},
            )
    make_run(
        make_cfg, dynamics={"type": "numeric", "rule": "friedkin_johnsen", "stubbornness": 0.5}
    )
    make_run(make_cfg, dynamics={"type": "numeric", "rule": "degroot"})
    make_run(
        make_cfg,
        network={"topology": "barabasi_albert", "n": 20, "m": 2},
        dynamics={"type": "numeric", "rule": "degroot"},
    )
    make_run(make_cfg)  # an LLM run
    out = tmp_path / "figs"
    res = CliRunner().invoke(app, ["analyze", str(tmp_path / "runs"), "--out", str(out)])
    assert res.exit_code == 0, res.output
    for f in (
        "bc_clusters_vs_epsilon",
        "fj_polarization_vs_stubbornness",
        "degroot_convergence_vs_topology",
        "fit_distances_wasserstein",
        "response_T1",
    ):
        assert (out / f"{f}.pdf").stat().st_size > 1000 and (out / f"{f}.csv").exists(), f
    assert len(list(out.glob("run_T1_*.pdf"))) == 1, sorted(p.name for p in out.iterdir())
    fits = pd.read_csv(out / "fits.csv")
    assert set(fits["rule"]) == {
        "nobody_moves",
        "degroot",
        "bounded_confidence",
        "friedkin_johnsen",
    }
    bc = pd.read_csv(out / "bc_clusters_vs_epsilon.csv")
    assert list(bc["epsilon"]) == [0.4, 1.0] and (bc["seeds"] == 2).all()


def test_the_whole_analysis_can_read_a_scorer(make_cfg, tmp_path):
    from typer.testing import CliRunner

    from disensa import scorer
    from disensa.analysis.summary import opinion_matrix
    from disensa.cli import app
    from disensa.llm import LLMClient

    scored = make_run(make_cfg)
    make_run(make_cfg, seed=4)  # an LLM run nobody scored
    asyncio.run(scorer.score_run(scored, "fake/scorer", client=LLMClient(tmp_path / "c")))
    X = opinion_matrix(load_run(scored), "fake/scorer")
    assert X.shape == (5, 20) and np.all(X == np.round(X))  # the scorer's whole levels
    assert not np.allclose(X, load_run(scored).opinion_matrix())
    out = tmp_path / "figs"
    res = CliRunner().invoke(
        app, ["analyze", str(tmp_path / "runs"), "--out", str(out), "--measure", "fake/scorer"]
    )
    assert res.exit_code == 0, res.output
    assert "skipped" in res.output and "has no scores.parquet" in res.output
    assert pd.read_csv(out / "fits.csv")["path"].nunique() == 1
    summary = pd.read_csv(out / "summary.csv")
    assert len(summary) == 1 and summary["final_mean"].iloc[0] == pytest.approx(X[-1].mean())
