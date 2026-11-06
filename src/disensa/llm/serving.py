"""What actually served each model of a run, recorded in meta.json as `serving`.

The same model name is not the same model everywhere: Ollama usually serves 4-bit quantized
weights, vLLM the full ones. So at the start of an LLM run disensa asks each server, once,
what it is serving:

  ollama_chat/<m>   GET <api_base or localhost:11434>/api/tags -> digest, quantization, size
  hosted_vllm/<m>   GET <api_base>/models                      -> the model ids it serves
  openai/<m>        (same, when an api_base is set)

This is best effort: a server that does not answer within 2 s is recorded as an error and
the run goes on (if it really is down, the replies fail and the run report says so).
"""

from __future__ import annotations

import json
import urllib.request

from disensa.config import ModelSpec
from disensa.llm.ollama import OLLAMA_URL


def _get(url: str, timeout: float = 2.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def describe(m: ModelSpec) -> dict:
    provider, _, name = m.name.partition("/")
    out: dict = {"provider": provider, "api_base": m.api_base}
    try:
        if provider in ("ollama", "ollama_chat"):
            base = (m.api_base or OLLAMA_URL).rstrip("/")
            tags = _get(f"{base}/api/tags").get("models", [])
            want = name if ":" in name else f"{name}:latest"
            hit = next((t for t in tags if t.get("name") in (name, want)), None)
            if hit is None:
                out["error"] = "not among the installed models"
            else:
                d = hit.get("details") or {}
                out.update(
                    digest=hit.get("digest"),
                    quantization=d.get("quantization_level"),
                    parameter_size=d.get("parameter_size"),
                    family=d.get("family"),
                )
        elif provider in ("hosted_vllm", "openai") and m.api_base:
            data = _get(f"{m.api_base.rstrip('/')}/models").get("data", [])
            out["served_models"] = [x.get("id") for x in data]
            hit = next((x for x in data if x.get("id") == name), None)
            if hit:
                out.update(root=hit.get("root"), max_model_len=hit.get("max_model_len"))
    except Exception as e:  # noqa: BLE001 - never fail a run over this
        out["error"] = f"{type(e).__name__}: {e}"[:200]
    return out


def serving_info(models: list[ModelSpec]) -> dict[str, dict]:
    return {m.name: describe(m) for m in models}
