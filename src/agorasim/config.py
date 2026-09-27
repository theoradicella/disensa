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


class TopicConfig(BaseModel):
    """The statement agents take a position on. +2 = strongly support, -2 = strongly oppose."""

    name: str = "euthanasia"
    statement: str = "Euthanasia should be legal for terminally ill adults who clearly request it."


class FeedConfig(BaseModel):
    """How many neighbour posts an agent reads per day (Wang: a random number)."""

    min_posts: int = Field(1, ge=0)
    max_posts: int = Field(5, ge=1)


class RecommenderConfig(BaseModel):
    """Wang et al. §3.3: only show neighbours whose opinion is within `threshold`."""

    enabled: bool = True
    threshold: float = Field(2.0, gt=0)


class ScorerConfig(BaseModel):
    """Fixed classifier mapping a post to [-2, 2] (Chuang et al.). Same model for all agents."""

    name: str
    api_base: str | None = None
    temperature: float = 0.0


class RewiringConfig(BaseModel):
    """End-of-day unfollow/follow dynamics (see rewiring.py). Off by default, as in Wang."""

    enabled: bool = False
    probability: float = Field(0.1, ge=0, le=1, description="chance an agent rewires per day")
    unfollow_threshold: float = Field(1.5, ge=0)
    follow: Literal["similar", "random"] = "similar"
    follow_threshold: float = Field(0.5, ge=0)


class SimulationConfig(BaseModel):
    days: int = Field(5, gt=0)
    feed: FeedConfig = FeedConfig()
    recommender: RecommenderConfig = RecommenderConfig()
    rewiring: RewiringConfig = RewiringConfig()
    # Which value drives metrics, the recommender and the next prompt:
    #   "self"       = the belief the agent reports (Wang et al.)
    #   "classifier" = the scorer's rating of the agent's post (Chuang et al.)
    opinion_source: Literal["self", "classifier"] = "self"
    # Chuang et al.: confirmation bias must be prompted explicitly
    bias: Literal["none", "confirmation"] = "none"
    memory_words: int = Field(60, gt=0)


class LLMSettings(BaseModel):
    cache_dir: Path = Path(".cache/llm")
    max_concurrency: int = Field(4, gt=0, description="parallel requests per provider")
    max_tokens: int = Field(400, gt=0)


class ExperimentConfig(BaseModel):
    name: str
    seed: int = 0
    network: NetworkConfig
    opinions: InitialOpinions = InitialOpinions()
    baseline: BaselineConfig | None = None
    topic: TopicConfig = TopicConfig()
    simulation: SimulationConfig = SimulationConfig()
    models: list[ModelSpec] = []
    # how agents are mapped to models, see ensemble.py
    assignment: Literal["balanced", "random", "clustered"] = "balanced"
    scorer: ScorerConfig | None = None
    llm: LLMSettings = LLMSettings()
    output_dir: Path = Path("runs")

    @model_validator(mode="after")
    def _check_scorer(self) -> ExperimentConfig:
        if self.simulation.opinion_source == "classifier" and self.scorer is None:
            raise ValueError("opinion_source 'classifier' requires a `scorer` section")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))
