import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from disensa.app.server import RunManager, make_handler

BASE = {
    "name": "t",
    "seed": 1,
    "network": {"topology": "watts_strogatz", "n": 16, "k": 4, "beta": 0.1},
    "steps": 3,
}
LLM = {
    **BASE,
    "dynamics": {"type": "llm", "models": [{"name": "fake/tribal"}], "max_concurrency": 32},
}
NUMERIC = {**BASE, "dynamics": {"type": "numeric", "rule": "degroot"}}


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # LLM cache goes to tmp
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "a.yaml").write_text(json.dumps(LLM))
    (tmp_path / "configs" / "old.yaml").write_text("name: x\nsimulation: {days: 3}\n")  # invalid
    handler = make_handler(RunManager(tmp_path / "runs"), tmp_path / "configs")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
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
            raw = r.read()
            return r.status, json.loads(raw) if "json" in r.headers["Content-Type"] else raw
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def wait_done(base, run_id):
    for _ in range(100):
        _, s = call(f"{base}/api/runs/{run_id}/status")
        if s["status"] != "running":
            return s
        time.sleep(0.1)
    raise AssertionError("run did not finish")


def test_pages_and_assets_are_served(server):
    for path in [
        "/",
        "/lab",
        "/docs",
        "/shared/base.css",
        "/shared/header.js",
        "/lab/lab.js",
        "/vendor/d3.min.js",
    ]:
        code, body = call(server + path)
        assert code == 200 and len(body) > 100, path
    assert call(server + "/lab/../../server.py")[0] == 404
    assert call(server + "/nope.css")[0] == 404


def test_options_list_only_valid_presets(server):
    code, o = call(f"{server}/api/options")
    assert code == 200
    assert [p["file"] for p in o["presets"]] == ["a.yaml"]
    assert "fake/tribal" in o["fake_models"]


@pytest.mark.parametrize("config,engine", [(LLM, "llm"), (NUMERIC, "numeric")])
def test_start_run_and_read_it_back(server, config, engine):
    code, res = call(f"{server}/api/runs", {"config": config})
    assert code == 201
    assert wait_done(server, res["id"])["status"] == "done"
    _, run = call(f"{server}/api/runs/{res['id']}")
    data = run["data"]
    assert len(data["opinions"]) == 4 and len(data["agents"]) == 16 and data["edges"]
    assert (data["posts"][1][0] is not None) == (engine == "llm")
    _, runs = call(f"{server}/api/runs")
    assert runs[0]["id"] == res["id"] and runs[0]["engine"] == engine


def test_invalid_config_is_a_400(server):
    code, err = call(f"{server}/api/runs", {"config": {**BASE, "dynamics": {"type": "llm"}}})
    assert code == 400 and "models" in err["error"]
