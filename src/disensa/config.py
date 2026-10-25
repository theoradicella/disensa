"""Experiment configuration: the YAML file you run, as typed Python objects.

An experiment is: a network, initial opinions, a number of steps, and the *dynamics* that
decide how an agent updates its opinion each step. `dynamics.type` switches between

  numeric   a classic opinion-dynamics rule (DeGroot, bounded confidence)
  llm       every agent is a language model that interacts with others through messages

Everything else (network, opinions, seed, steps, metrics, storage) is shared by both.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

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
        if self.topology == "watts_strogatz" and not 2 <= self.k < self.n:
            raise ValueError(f"watts_strogatz needs 2 ≤ k < n (got k = {self.k}, n = {self.n})")
        if self.topology == "watts_strogatz" and self.k % 2:
            raise ValueError(f"watts_strogatz needs an even k (got k = {self.k})")
        if self.topology == "barabasi_albert" and self.m >= self.n:
            raise ValueError(f"barabasi_albert needs m < n (got m = {self.m}, n = {self.n})")
        return self


class InitialOpinions(BaseModel):
    distribution: Literal["uniform", "normal", "bimodal"] = "uniform"
    # normal: mean/std; bimodal: two normals at ±mean with std
    mean: float = 0.0
    std: float = 1.0


# ---------------------------------------------------------------- dynamics


class NumericDynamics(BaseModel):
    """A fixed mathematical update rule (see dynamics/numeric.py)."""

    type: Literal["numeric"] = "numeric"
    rule: Literal["degroot", "bounded_confidence"] = "bounded_confidence"
    epsilon: float = Field(1.0, gt=0, description="confidence bound (bounded confidence)")
    mu: float = Field(0.5, gt=0, le=1, description="convergence rate (bounded confidence)")


class ModelSpec(BaseModel):
    """One LLM. `name` is a LiteLLM model string, e.g. "ollama_chat/llama3.2:3b"."""

    name: str
    weight: float = Field(1.0, gt=0, description="share of agents using this model")
    api_base: str | None = None  # e.g. a vLLM endpoint
    temperature: float = 0.7


class InteractionsConfig(BaseModel):
    """How many other agents an agent interacts with per step (Wang: a random number).

    For now the authors are drawn from the agent's network neighbours.
    """

    min: int = Field(1, ge=0)
    max: int = Field(5, ge=1)

    @model_validator(mode="after")
    def _check_range(self) -> InteractionsConfig:
        if self.min > self.max:
            raise ValueError(f"interactions min ({self.min}) is above max ({self.max})")
        return self


class LLMDynamics(BaseModel):
    """Agents are language models: they interact through messages and report a new belief."""

    type: Literal["llm"] = "llm"
    statement: str = "Euthanasia should be legal for terminally ill adults who clearly request it."
    models: list[ModelSpec] = Field(min_length=1)
    interactions: InteractionsConfig = InteractionsConfig()
    memory_words: int = Field(60, gt=0)
    max_tokens: int = Field(400, gt=0)
    max_concurrency: int = Field(4, gt=0, description="parallel requests per provider")
    cache_dir: Path = Path(".cache/llm")


Dynamics = Annotated[NumericDynamics | LLMDynamics, Field(discriminator="type")]


# ---------------------------------------------------------------- experiment


class ExperimentConfig(BaseModel):
    name: str
    seed: int = 0
    network: NetworkConfig
    opinions: InitialOpinions = InitialOpinions()
    steps: int = Field(10, gt=0, description="number of update steps")
    dynamics: Dynamics
    output_dir: Path = Path("runs")

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))
