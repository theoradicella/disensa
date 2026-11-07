"""The run report (report.py): per-call fields, step metrics, the report and the failure guard."""

import asyncio
import json
import math
import shutil

import pandas as pd
import pytest

from disensa.app.payload import build_payload
from disensa.app.server import RunManager
from disensa.engine import RunFailed, run
from disensa.report import format_report, rounding, summarize
from disensa.runs import load_run, new_run_dir


def llm_cfg(make_cfg, models, **dyn_overrides):
    dyn = make_cfg().dynamics.model_dump(mode="json")
    dyn["models"] = [{"name": m} if isinstance(m, str) else m for m in models]
    dyn.update(dyn_overrides)
    return make_cfg(dynamics=dyn)


def run_to_folder(cfg):
    run_dir = new_run_dir(cfg)
    asyncio.run(run(cfg, run_dir))
    return run_dir, load_run(run_dir)


def test_every_event_records_what_the_call_cost(make_cfg):
    run_dir, r = run_to_folder(make_cfg())
    for e in r.events:
        assert e["latency_s"] >= 0 and e["prompt_tokens"] > 0 and e["completion_tokens"] > 0
        assert e["cached"] is False and e["error_kind"] is None and e["clipped"] is False
    assert all(e["belief_raw"] is None for e in r.events if e["step"] == 0)
    assert all(isinstance(e["belief_raw"], float) for e in r.events if e["step"] >= 1)
    assert r.meta["status"] == "finished" and r.meta["finished"]


def test_a_cached_rerun_reports_the_original_cost(make_cfg):
    _, first = run_to_folder(make_cfg())
    _, again = run_to_folder(make_cfg())
    for a, b in zip(first.events, again.events, strict=True):
        assert b["cached"] is True
        assert (a["prompt_tokens"], a["completion_tokens"]) == (
            b["prompt_tokens"],
            b["completion_tokens"],
        )
        assert a["latency_s"] == pytest.approx(b["latency_s"], abs=1e-3)
    assert again.meta["report"]["cache_hits"] == again.meta["report"]["calls"]
    assert again.meta["report"]["seconds_per_call"] is None  # nothing reached a model


@pytest.mark.parametrize(
    "model,kind",
    [
        ("fake/badjson", "invalid_json"),
        ("fake/nobelief", "missing_belief"),
        ("fake/nanbelief", "invalid_belief"),
        ("fake/textbelief", "invalid_belief"),
        ("fake/down", "provider"),
    ],
)
def test_bad_replies_are_classified_and_keep_the_opinion(make_cfg, model, kind):
    cfg = llm_cfg(make_cfg, [model], max_error_rate=1.0)
    _, r = run_to_folder(cfg)
    later = [e for e in r.events if e["step"] >= 1]
    assert all(e["error_kind"] == kind and e["error"] for e in later)
    assert all(e["belief_raw"] is None for e in later)
    ops = r.opinion_matrix()
    assert (ops[-1] == ops[0]).all()  # nobody moved
    assert r.metrics["errors"].iloc[-1] == 20
    assert r.meta["report"]["failed_by_kind"][kind] >= 20 * cfg.steps


def test_a_json_repair_counts_both_requests(make_cfg):
    cfg = llm_cfg(make_cfg, ["fake/badjson"], max_error_rate=1.0)
    _, r = run_to_folder(cfg)
    e = r.events[0]
    one = len("I would rather not answer in that format.".split())
    assert e["completion_tokens"] == 2 * one  # the reply and the failed repair


def test_out_of_range_beliefs_are_clipped_and_counted(make_cfg):
    cfg = llm_cfg(make_cfg, ["fake/outofrange"])
    _, r = run_to_folder(cfg)
    later = [e for e in r.events if e["step"] >= 1]
    assert all(e["belief_raw"] == 5 and e["clipped"] and not e["error"] for e in later)
    assert (r.opinion_matrix()[1:] == 2).all()
    assert (r.metrics["out_of_range"].iloc[1:] == 20).all()
    assert r.meta["report"]["out_of_range"] == 20 * cfg.steps
    assert r.meta["report"]["rounding"] == {"whole": 20 * cfg.steps, "half": 0, "other": 0}


def test_step_metrics_and_report(make_cfg):
    cfg = llm_cfg(make_cfg, [{"name": "fake/halves", "weight": 19}, "fake/nobelief"])
    _, r = run_to_folder(cfg)  # 1 of 20 agents fails every step: under the 20 % guard
    for col in ("errors", "out_of_range", "prompt_tokens", "completion_tokens", "seconds"):
        assert col in r.metrics.columns
    assert r.metrics["prompt_tokens"].iloc[1] > 0
    rep = r.meta["report"]
    assert rep["calls"] == 20 * (cfg.steps + 1)
    assert rep["failed"] == cfg.steps and rep["failed_by_kind"] == {"missing_belief": cfg.steps}
    assert rep["failed_by_model"]["fake/nobelief"]["pct"] == pytest.approx(100 * 4 / 5)
    assert rep["failed_by_model"]["fake/halves"]["failed"] == 0
    assert rep["prompt_tokens"] == r.metrics["prompt_tokens"].sum()
    g = rep["rounding"]
    assert g["other"] == 0 and g["whole"] + g["half"] == 19 * cfg.steps
    assert "failed" in format_report(rep) and "beliefs" in format_report(rep)


def test_the_guard_stops_a_run_that_mostly_fails(make_cfg):
    cfg = llm_cfg(make_cfg, ["fake/nobelief"])
    run_dir = new_run_dir(cfg)
    with pytest.raises(RunFailed) as exc:
        asyncio.run(run(cfg, run_dir))
    assert exc.value.reason.startswith("step 1: 20/20 replies failed (missing_belief: ")
    r = load_run(run_dir)
    assert r.meta["status"] == "failed" and r.meta["reason"] == exc.value.reason
    assert r.steps == 1  # step 0 and the failed step are kept
    assert not r.meta.get("finished")
    assert RunManager(run_dir.parent).status(run_dir.name)["status"] == "failed"


def test_the_guard_watches_step_0_too(make_cfg):
    cfg = llm_cfg(make_cfg, ["fake/down"])
    with pytest.raises(RunFailed, match=r"step 0: 20/20 .*provider: .*connection refused"):
        asyncio.run(run(cfg))


def test_a_few_failures_do_not_stop_a_run(make_cfg):
    cfg = llm_cfg(make_cfg, ["fake/bounded", {"name": "fake/down", "weight": 0.25}])
    history = asyncio.run(run(cfg))  # 4 of 20 agents fail: exactly 20 %, not above it
    assert len(history) == cfg.steps + 1 and history[1]["errors"] == 4


def test_rounding():
    assert [rounding(v) for v in (1.0, -2, 0.5, -1.5, 0.7, 1.25)] == [
        "whole",
        "whole",
        "half",
        "half",
        "other",
        "other",
    ]


def test_old_run_folders_still_load(make_cfg, tmp_path):
    """A run folder from 0.2: no status, no report, events without the new fields."""
    run_dir, _ = run_to_folder(make_cfg())
    meta = json.loads((run_dir / "meta.json").read_text())
    for key in ("status", "report", "reason"):
        meta.pop(key, None)
    (run_dir / "meta.json").write_text(json.dumps(meta))
    new_fields = {"error_kind", "belief_raw", "clipped", "latency_s", "prompt_tokens"}
    new_fields |= {"completion_tokens", "cached"}
    lines = [json.loads(x) for x in (run_dir / "events.jsonl").read_text().splitlines()]
    lines[5]["error"] = "missing or invalid 'belief' in reply"
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps({k: v for k, v in e.items() if k not in new_fields}) for e in lines)
    )
    m = pd.read_parquet(run_dir / "metrics.parquet")
    m.drop(columns=["out_of_range", "prompt_tokens", "completion_tokens", "seconds"]).to_parquet(
        run_dir / "metrics.parquet", index=False
    )
    old = tmp_path / "old_runs" / run_dir.name
    shutil.copytree(run_dir, old)

    r = load_run(old)
    assert r.status == "finished"
    payload = build_payload(r)
    rep = payload["report"]
    assert rep["failed"] == 1 and rep["failed_by_kind"] == {"not recorded": 1}
    assert rep["prompt_tokens"] is None and rep["rounding"] is None
    assert rep["out_of_range"] is None and rep["cache_hits"] is None
    json.dumps(payload, allow_nan=False)
    listed = RunManager(old.parent).list()
    assert listed[0]["status"] == "done" and listed[0]["failed_pct"] is None
    assert "not recorded" in format_report(rep)


def test_summarize_with_no_events():
    rep = summarize([], 1.0)
    assert rep["calls"] == 0 and rep["failed_pct"] == 0.0 and not math.isnan(rep["seconds"])


def test_disensa_run_prints_the_report_and_fails_with_the_guard(make_cfg, tmp_path):
    import yaml
    from typer.testing import CliRunner

    from disensa.cli import app

    for model, code in [("fake/bounded", 0), ("fake/nobelief", 1)]:
        cfg = llm_cfg(make_cfg, [model])
        path = tmp_path / f"{model.split('/')[1]}.yaml"
        path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
        res = CliRunner().invoke(app, ["run", str(path)])
        assert res.exit_code == code, res.output
        assert "run report" in res.output and "failed" in res.output


def test_report_wording_for_cached_and_empty_runs():
    cached = summarize([{"step": 1, "cached": True, "latency_s": 2.0, "belief_raw": 1.0}], 0.0)
    assert "all from the cache" in format_report(cached)
    stopped = summarize(
        [{"step": 0, "error": "x", "error_kind": "provider", "cached": False, "belief_raw": None}],
        1.0,
    )
    assert "no valid self-ratings" in format_report(stopped)
