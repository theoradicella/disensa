import pytest
from fake_llm import fake_reply

from disensa.config import ExperimentConfig
from disensa.llm import client as llm_client


@pytest.fixture(autouse=True)
def offline_models(monkeypatch):
    """Every model call in the tests goes to the offline stand-in in fake_llm.py."""

    async def fake_call(self, model, messages, params, api_base, json_mode):
        return llm_client.LLMResponse(text=fake_reply(model, messages), model=model)

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
