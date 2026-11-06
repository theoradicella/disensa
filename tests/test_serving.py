"""meta.json records what served each model (llm/serving.py), against a local stand-in."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from conftest import REAL_DESCRIBE

from disensa.config import ModelSpec
from disensa.engine import run
from disensa.runs import load_run, new_run_dir

TAGS = {
    "models": [
        {
            "name": "llama3.2:3b",
            "digest": "a80c4f17acd5",
            "details": {"quantization_level": "Q4_K_M", "parameter_size": "3.2B"},
        }
    ]
}
MODELS = {"data": [{"id": "meta-llama/Llama-3.1-8B-Instruct", "max_model_len": 8192}]}


@pytest.fixture
def stand_in():
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):  # noqa: N802
            body = {"/api/tags": TAGS, "/v1/models": MODELS}.get(self.path)
            self.send_response(200 if body else 404)
            self.end_headers()
            self.wfile.write(json.dumps(body or {}).encode())

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_describe_ollama_and_vllm(stand_in):
    o = REAL_DESCRIBE(ModelSpec(name="ollama_chat/llama3.2:3b", api_base=stand_in))
    assert o["quantization"] == "Q4_K_M" and o["digest"] == "a80c4f17acd5"
    missing = REAL_DESCRIBE(ModelSpec(name="ollama_chat/qwen3:4b", api_base=stand_in))
    assert "not among" in missing["error"]
    v = REAL_DESCRIBE(
        ModelSpec(name="hosted_vllm/meta-llama/Llama-3.1-8B-Instruct", api_base=f"{stand_in}/v1")
    )
    assert v["served_models"] == ["meta-llama/Llama-3.1-8B-Instruct"]
    assert v["max_model_len"] == 8192
    down = REAL_DESCRIBE(ModelSpec(name="hosted_vllm/x", api_base="http://127.0.0.1:9/v1"))
    assert down["error"]  # never raises


def test_runs_record_serving_version_and_hash(make_cfg):
    cfg = make_cfg()
    run_dir = new_run_dir(cfg)
    asyncio.run(run(cfg, run_dir))
    meta = load_run(run_dir).meta
    assert meta["serving"] == {"fake/bounded": {"provider": "fake"}}
    assert meta["disensa_version"] and len(meta["config_hash"]) == 16
