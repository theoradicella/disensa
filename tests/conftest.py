import pytest

from disensa.config import ExperimentConfig


@pytest.fixture
def make_cfg(tmp_path):
    """Build a small config; `dynamics` defaults to offline fake LLM agents."""

    def make(**overrides) -> ExperimentConfig:
        raw = {
            "name": "t",
            "seed": 3,
            "network": {"topology": "watts_strogatz", "n": 20, "k": 4, "beta": 0.1},
            "steps": 4,
            "dynamics": {
                "type": "llm",
                "models": [{"name": "fake/bounded"}],
                "feed": {"min_posts": 1, "max_posts": 3},
                "cache_dir": str(tmp_path / "cache"),
                "max_concurrency": 32,
            },
            "output_dir": str(tmp_path / "runs"),
        }
        raw.update(overrides)
        return ExperimentConfig.model_validate(raw)

    return make
