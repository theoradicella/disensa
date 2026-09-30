"""LLM agents, following the Social Simulation Framework of Wang et al. (COLING 2025, §3).

  step 0   every agent writes a first post from its initial opinion
  step t   every agent reads a few posts by its neighbours (from step t-1), reasons over them
           with its memory, and replies with a new belief, a new post and an updated memory

All agents of a step call their models concurrently; results are applied together, so the
order in which agents are processed never matters. A malformed reply never crashes a run:
it is logged in `error` and the agent keeps its previous state.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import networkx as nx
import numpy as np

from disensa.config import LLMDynamics, ModelSpec
from disensa.llm import LLMClient
from disensa.llm.prompts import initial_post_prompt, system_prompt, update_prompt


@dataclass
class TextState:
    """What an LLM agent carries between steps besides its opinion."""

    post: str = ""
    memory: str = ""


def assign_models(models: list[ModelSpec], n: int, seed: int) -> list[ModelSpec]:
    """Exact shares by weight (largest-remainder rounding), randomly placed on the network.

    Uses its own RNG stream (seed + 2_000) so feeds are unaffected by the model mix.
    """
    w = np.array([m.weight for m in models], dtype=float)
    raw = w / w.sum() * n
    counts = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - counts), kind="stable")[: n - counts.sum()]:
        counts[i] += 1
    pool = [m for m, c in zip(models, counts, strict=True) for _ in range(c)]
    rng = np.random.default_rng(seed + 2_000)
    return [pool[i] for i in rng.permutation(n)]


class LLMAgents:
    def __init__(self, cfg: LLMDynamics, graph: nx.Graph, seed: int = 0):
        self.cfg = cfg
        self.graph = graph
        self.seed = seed
        self.n = graph.number_of_nodes()
        self.models = assign_models(cfg.models, self.n, seed)
        self.state = [TextState() for _ in range(self.n)]
        self.rng = np.random.default_rng(seed)  # which posts each agent reads
        self.client = LLMClient(cfg.cache_dir, cfg.max_concurrency)

    def labels(self) -> list[str]:
        return [m.name for m in self.models]

    # ---------------------------------------------------------------- one model call

    async def ask(self, i: int, user: str) -> tuple[dict, str, str | None]:
        """Returns (parsed JSON, raw reply, error). Never raises for bad model output."""
        m = self.models[i]
        messages = [
            {"role": "system", "content": system_prompt(i)},
            {"role": "user", "content": user},
        ]
        try:
            obj, raw = await self.client.complete_json(
                m.name,
                messages,
                temperature=m.temperature,
                max_tokens=self.cfg.max_tokens,
                api_base=m.api_base,
                seed=self.seed,
            )
            return obj, raw, None
        except Exception as e:  # malformed JSON after repair, provider error, ...
            return {}, "", f"{type(e).__name__}: {e}"[:500]

    # ---------------------------------------------------------------- reading

    def feed(self, i: int) -> list[tuple[int, str]]:
        """A random number of posts, from a random subset of i's neighbours."""
        authors = [j for j in sorted(self.graph.neighbors(i)) if self.state[j].post]
        k = int(self.rng.integers(self.cfg.feed.min_posts, self.cfg.feed.max_posts + 1))
        k = min(k, len(authors))
        if k == 0:
            return []
        chosen = sorted(self.rng.choice(authors, size=k, replace=False).tolist())
        return [(j, self.state[j].post) for j in chosen]

    # ---------------------------------------------------------------- the two phases

    async def start(self, x0: np.ndarray) -> list[dict]:
        prompts = [initial_post_prompt(self.cfg.statement, float(v)) for v in x0]
        replies = await asyncio.gather(*(self.ask(i, p) for i, p in enumerate(prompts)))
        events = []
        for i, (obj, raw, err) in enumerate(replies):
            self.state[i].post = str(obj.get("post", "")).strip()
            events.append(self._event(0, i, [], "", self.state[i], prompts[i], raw, err))
        return events

    async def step(self, x: np.ndarray, day: int) -> tuple[np.ndarray, list[dict]]:
        feeds = [self.feed(i) for i in range(self.n)]  # everyone reads yesterday's posts
        prompts = [
            update_prompt(
                self.cfg.statement,
                float(x[i]),
                self.state[i].memory,
                feeds[i],
                self.cfg.memory_words,
            )
            for i in range(self.n)
        ]
        replies = await asyncio.gather(*(self.ask(i, p) for i, p in enumerate(prompts)))

        new_x = x.copy()
        new_state, events = [], []
        for i, (obj, raw, err) in enumerate(replies):
            try:
                if not err:
                    new_x[i] = float(obj["belief"])
            except (KeyError, TypeError, ValueError):
                err = f"missing or invalid 'belief' in reply: {raw[:200]!r}"
            old = self.state[i]
            s = TextState(
                post=str(obj.get("post", "")).strip() or old.post,
                memory=str(obj.get("memory", "")).strip() or old.memory,
            )
            reasoning = str(obj.get("reasoning", "")).strip()
            new_state.append(s)
            events.append(self._event(day, i, feeds[i], reasoning, s, prompts[i], raw, err))
        self.state = new_state  # applied together: synchronous update
        return new_x, events

    def _event(self, day, i, feed, reasoning, s: TextState, prompt, raw, err) -> dict:
        return {
            "day": day,
            "agent": i,
            "model": self.models[i].name,
            "feed": [{"author": j, "post": p} for j, p in feed],
            "reasoning": reasoning,
            "post": s.post,
            "memory": s.memory,
            "prompt": prompt,
            "raw": raw,
            "error": err,
        }
