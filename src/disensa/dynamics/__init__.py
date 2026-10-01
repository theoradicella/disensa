"""Dynamics: the one pluggable part of a simulation — how agents update their opinions.

The engine owns the loop; a dynamics object only answers "given everyone's opinions at t-1,
what are the opinions at t?". Two implementations:

  NumericAgents  (numeric.py)    a fixed rule on numbers: DeGroot, bounded confidence
  LLMAgents      (llm_agent.py)  each agent interacts with others and asks its LLM

Both are async so the engine does not care which one it runs; numeric steps simply never wait.
"""

from __future__ import annotations

from typing import Protocol

import networkx as nx
import numpy as np

from disensa.config import ExperimentConfig


class Dynamics(Protocol):
    def labels(self) -> list[str]:
        """What drives each agent, e.g. "ollama_chat/llama3.2:3b" or "numeric/degroot"."""
        ...

    async def start(self, x0: np.ndarray) -> list[dict]:
        """Step 0. Returns events to log (LLM agents state a first message; numeric: none)."""
        ...

    async def step(self, x: np.ndarray, t: int) -> tuple[np.ndarray, list[dict]]:
        """One synchronous update: use the state at t-1, return (new opinions, events)."""
        ...


def make_dynamics(cfg: ExperimentConfig, graph: nx.Graph) -> Dynamics:
    if cfg.dynamics.type == "numeric":
        from disensa.dynamics.numeric import NumericAgents

        return NumericAgents(cfg.dynamics, graph)
    from disensa.dynamics.llm_agent import LLMAgents

    return LLMAgents(cfg.dynamics, graph, seed=cfg.seed)
