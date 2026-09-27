"""Typed experiment configuration, loaded from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class NetworkConfig(BaseModel):
    topology: Literal["erdos_renyi", "watts_strogatz", "barabasi_albert"]
    n: int = Field(50, gt=1)
    # Erdős–Rényi
    p: float | None = Field(None, ge=0, le=1)
    # Watts–Strogatz
    k: int | None = Field(None, gt=0)
    beta: float | None = Field(None, ge=0, le=1)
    # Barabási–Albert
    m: int | None = Field(None, gt=0)

    @model_validator(mode="after")
    def _check_params(self) -> NetworkConfig:
        required = {
            "erdos_renyi": ["p"],
            "watts_strogatz": ["k", "beta"],
            "barabasi_albert": ["m"],
        }[self.topology]
        missing = [name for name in required if getattr(self, name) is None]
        if missing:
            raise ValueError(f"{self.topology} requires: {', '.join(missing)}")
        return self


class InitialOpinions(BaseModel):
    distribution: Literal["uniform", "normal", "bimodal"] = "uniform"
    # normal: mean/std; bimodal: two normals at ±mean with std
    mean: float = 0.0
    std: float = 1.0


class BaselineConfig(BaseModel):
    model: Literal["degroot", "bounded_confidence", "deffuant", "friedkin_johnsen"]
    steps: int = Field(100, gt=0)
    epsilon: float = Field(1.0, gt=0, description="confidence bound (BC / Deffuant)")
    mu: float = Field(0.5, gt=0, le=1, description="convergence rate (BC / Deffuant)")
    stubbornness: float = Field(1.0, ge=0, description="FJ weight on innate opinion")


class ModelSpec(BaseModel):
    """One LLM in the ensemble. `name` is a LiteLLM model string."""

    name: str  # e.g. "anthropic/claude-sonnet-4-5", "ollama/llama3.1", "openai/gpt-4o-mini"
    weight: float = Field(1.0, gt=0, description="share of agents using this model")
    api_base: str | None = None  # e.g. a vLLM endpoint on INDACO
    temperature: float = 0.7


class ExperimentConfig(BaseModel):
    name: str
    seed: int = 0
    network: NetworkConfig
    opinions: InitialOpinions = InitialOpinions()
    baseline: BaselineConfig | None = None
    models: list[ModelSpec] = []
    output_dir: Path = Path("runs")

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))
