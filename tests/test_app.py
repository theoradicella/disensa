import json
import re
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from agorasim.app import RunManager, make_handler

CAMPS = {
    "name": "t",
    "seed": 1,
    "network": {"topology": "watts_strogatz", "n": 16, "k": 4, "beta": 0.1},
    "simulation": {"days": 3, "rewiring": {"enabled": True, "probability": 0.3}},
    "models": [{"name": "fake/tribal"}],
    "scorer": {"name": "fake/scorer"},
    "llm": {"max_concurrency": 32},
}


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # LLM cache goes to tmp
    manager = RunManager(tmp_path / "runs")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "a.yaml").write_text(json.dumps(CAMPS))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(manager, tmp_path / "configs"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def call(url, body=None):
    req = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def wait_done(base, run_id):
    for _ in range(100):
        _, s = call(f"{base}/api/runs/{run_id}/status")
        if s["status"] != "running":
            return s
        time.sleep(0.1)
    raise AssertionError("run did not finish")


def test_options_list_presets(server):
    code, o = call(f"{server}/api/options")
    assert code == 200
    assert [p["file"] for p in o["presets"]] == ["a.yaml"]
    assert "fake/tribal" in o["fake_models"]


def test_llm_run_end_to_end(server):
    code, r = call(f"{server}/api/runs", {"engine": "llm", "config": CAMPS})
    assert code == 201
    assert wait_done(server, r["id"])["status"] == "done"
    _, d = call(f"{server}/api/runs/{r['id']}")
    assert len(d["data"]["opinions"]) == 4 and len(d["data"]["agents"]) == 16
    _, runs = call(f"{server}/api/runs")
    assert runs[0]["id"] == r["id"] and runs[0]["status"] == "done"


def test_numeric_run(server):
    cfg = {**CAMPS, "baseline": {"model": "deffuant", "steps": 5, "epsilon": 1.0}}
    code, r = call(f"{server}/api/runs", {"engine": "numeric", "config": cfg})
    assert code == 201
    assert wait_done(server, r["id"])["status"] == "done"
    _, d = call(f"{server}/api/runs/{r['id']}")
    assert d["data"]["models"] == ["numeric/deffuant"] and d["data"]["topic"] == ""


def test_invalid_config_is_a_400(server):
    bad = {**CAMPS, "network": {"topology": "erdos_renyi", "n": 10}}
    code, r = call(f"{server}/api/runs", {"engine": "llm", "config": bad})
    assert code == 400 and "requires: p" in r["error"]


def test_unknown_run_is_a_404(server):
    assert call(f"{server}/api/runs/nope")[0] == 404
    assert call(f"{server}/api/runs/..%2Fetc")[0] == 404


def test_landing_and_lab_pages(server):
    with urllib.request.urlopen(f"{server}/") as r:
        landing = r.read().decode()
    with urllib.request.urlopen(f"{server}/lab") as r:
        lab = r.read().decode()
    assert 'href="/lab"' in landing and "<canvas" in landing
    assert 'id="form"' in lab
    # the project name is not shown on the page (links to GitHub/PyPI may contain it)
    visible = re.sub(
        r"<[^>]+>", " ", re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", landing, flags=re.S)
    )
    assert "agorasim" not in visible.lower()
