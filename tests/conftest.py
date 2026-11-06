from types import SimpleNamespace

import pytest
from fake_llm import fake_reply

from disensa.config import ExperimentConfig
from disensa.llm import client as llm_client

_REAL_CALL = llm_client.LLMClient._call_litellm  # before the autouse fixture replaces it


@pytest.fixture(autouse=True)
def offline_models(monkeypatch):
    """Every model call in the tests goes to the offline stand-in in fake_llm.py."""

    async def fake_call(self, model, messages, params, api_base, json_mode):
        if model.endswith("/down"):
            raise ConnectionError("connection refused")
        text = fake_reply(model, messages)
        usage = {
            "prompt_tokens": sum(len(m["content"].split()) for m in messages),
            "completion_tokens": len(text.split()),
        }
        return llm_client.LLMResponse(text=text, model=model, usage=usage)

    monkeypatch.setattr(llm_client.LLMClient, "_call_litellm", fake_call)


@pytest.fixture
def make_cfg(tmp_path):
    """Build a small config; `dynamics` defaults to LLM agents on the offline stand-in."""

    def make(**overrides) -> ExperimentConfig:
        raw = {
            "name": "t",
            "seed": 3,
            "network": {"topology": "watts_strogatz", "n": 20, "k": 4, "beta": 0.1},
            "steps": 4,
            "dynamics": {
                "type": "llm",
                "models": [{"name": "fake/bounded"}],
                "interactions": {"min": 1, "max": 3},
                "cache_dir": str(tmp_path / "cache"),
                "max_concurrency": 32,
            },
            "output_dir": str(tmp_path / "runs"),
        }
        raw.update(overrides)
        return ExperimentConfig.model_validate(raw)

    return make


@pytest.fixture
def litellm_kwargs(monkeypatch):
    """Run the real `_call_litellm`, but stop at `litellm.acompletion` and record its kwargs.

    Returns the list the kwargs of every call are appended to. Nothing leaves the machine.
    """
    import litellm

    seen: list[dict] = []

    async def fake_acompletion(**kwargs):
        seen.append(kwargs)
        msg = SimpleNamespace(content='{"belief": 1}')
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg)],
            usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        )

    monkeypatch.setattr(llm_client.LLMClient, "_call_litellm", _REAL_CALL)
    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    return seen
