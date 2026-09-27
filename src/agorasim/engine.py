"""The LLM social simulation: agents on a network reading, reasoning and posting, day by day.

Follows Wang et al. (COLING 2025) §3:
  day 0   every agent writes an initial post from its persona and initial opinion
  day t   each agent reads a random number of posts by its neighbours (optionally filtered by
          a similarity recommender), reasons over them with its long-term memory, then reports
          a new belief, writes a new post and updates its memory.

Days are synchronous: every agent reads the state of day t-1 and all updates are applied
together, so the order in which agents are processed never matters. All agents in a day
call their models concurrently.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from agorasim import OPINION_MAX, OPINION_MIN
from agorasim.config import ExperimentConfig, ModelSpec
from agorasim.ensemble import assign_models, model_homophily
from agorasim.llm import LLMClient
from agorasim.metrics import all_metrics
from agorasim.networks import build_network, initial_opinions
from agorasim.personas import Persona, generate_personas
from agorasim.prompts import (
    SCORER_SYSTEM,
    initial_post_prompt,
    scorer_prompt,
    system_prompt,
    update_prompt,
)
from agorasim.runs import RunWriter


@dataclass
class AgentState:
    persona: Persona
    model: ModelSpec
    belief: float  # self-reported (Wang)
    score: float | None = None  # classifier rating of the latest post (Chuang)
    post: str = ""
    memory: str = ""
    errors: int = 0
    history: list[float] = field(default_factory=list)

    @property
    def id(self) -> int:
        return self.persona.agent_id

    def opinion(self, source: str) -> float:
        if source == "classifier" and self.score is not None:
            return self.score
        return self.belief


def _clip(v: float) -> float:
    return float(np.clip(v, OPINION_MIN, OPINION_MAX))


class Simulation:
    def __init__(
        self,
        cfg: ExperimentConfig,
        client: LLMClient | None = None,
        assignment: list[ModelSpec] | None = None,
    ):
        self.cfg = cfg
        self.sim = cfg.simulation
        self.statement = cfg.topic.statement
        self.rng = np.random.default_rng(cfg.seed)  # feed sampling
        self.graph: nx.Graph = build_network(cfg.network, seed=cfg.seed)
        n = self.graph.number_of_nodes()
        x0 = initial_opinions(cfg.opinions, n, seed=cfg.seed)
        self.personas = generate_personas(x0, cfg.seed)
        models = assignment or assign_models(cfg.models, self.graph, cfg.assignment, cfg.seed)
        self.agents = [
            AgentState(persona=p, model=m, belief=p.initial_opinion)
            for p, m in zip(self.personas, models, strict=True)
        ]
        self.client = client or LLMClient(cfg.llm.cache_dir, cfg.llm.max_concurrency)
        self.day = 0
        self.writer: RunWriter | None = None

    # ------------------------------------------------------------------ helpers

    def opinions(self) -> np.ndarray:
        return np.array([a.opinion(self.sim.opinion_source) for a in self.agents])

    async def _ask(self, model: ModelSpec, system: str, user: str) -> tuple[dict, str, str | None]:
        """Returns (parsed, raw, error). Never raises for bad model output."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            obj, raw = await self.client.complete_json(
                model.name,
                messages,
                temperature=model.temperature,
                max_tokens=self.cfg.llm.max_tokens,
                api_base=model.api_base,
                seed=self.cfg.seed,
            )
            return obj, raw, None
        except Exception as e:  # malformed JSON after repair, provider error, ...
            return {}, "", f"{type(e).__name__}: {e}"[:500]

    async def _score(self, post: str) -> float | None:
        if self.cfg.scorer is None or not post:
            return None
        s = self.cfg.scorer
        messages = [
            {"role": "system", "content": SCORER_SYSTEM},
            {"role": "user", "content": scorer_prompt(self.statement, post)},
        ]
        try:
            obj, _ = await self.client.complete_json(
                s.name, messages, temperature=s.temperature, max_tokens=50, api_base=s.api_base
            )
            return _clip(float(obj["score"]))
        except Exception:
            return None

    def build_feed(self, i: int, opinions: np.ndarray) -> list[tuple[int, str]]:
        """Posts from a random subset of i's neighbours, optionally filtered by similarity."""
        nbrs = sorted(self.graph.neighbors(i))
        rec = self.sim.recommender
        if rec.enabled:
            nbrs = [j for j in nbrs if abs(opinions[i] - opinions[j]) <= rec.threshold]
        nbrs = [j for j in nbrs if self.agents[j].post]
        k = int(self.rng.integers(self.sim.feed.min_posts, self.sim.feed.max_posts + 1))
        k = min(k, len(nbrs))
        if k == 0:
            return []
        chosen = sorted(self.rng.choice(nbrs, size=k, replace=False).tolist())
        return [(j, self.agents[j].post) for j in chosen]

    def _record_state(self) -> None:
        if self.writer is None:
            return
        src = self.sim.opinion_source
        self.writer.opinions(
            [
                {
                    "day": self.day,
                    "agent": a.id,
                    "model": a.model.name,
                    "belief": a.belief,
                    "score": a.score,
                    "opinion": a.opinion(src),
                }
                for a in self.agents
            ]
        )
        self.writer.metrics({"day": self.day, **self.day_metrics()})

    def day_metrics(self) -> dict[str, float]:
        ops = self.opinions()
        m = all_metrics(self.graph, ops)
        m["errors"] = sum(a.errors for a in self.agents)
        models = [a.model for a in self.agents]
        if len({x.name for x in models}) > 1:
            m["model_homophily"] = model_homophily(self.graph, models)
            for name in sorted({x.name for x in models}):
                mask = np.array([x.name == name for x in models])
                m[f"mean_opinion[{name}]"] = float(ops[mask].mean())
        return m

    # ------------------------------------------------------------------ simulation

    async def initialize(self) -> None:
        """Day 0: every agent writes an initial post from its persona and initial opinion."""

        async def one(a: AgentState):
            system = system_prompt(a.persona, self.sim.bias)
            user = initial_post_prompt(self.statement, a.belief)
            obj, raw, err = await self._ask(a.model, system, user)
            post = str(obj.get("post", "")).strip() if not err else ""
            return a, user, raw, err, post

        results = await asyncio.gather(*(one(a) for a in self.agents))
        scores = await asyncio.gather(*(self._score(r[4]) for r in results))
        for (a, user, raw, err, post), score in zip(results, scores, strict=True):
            a.post, a.score = post, score
            a.errors += bool(err)
            a.history.append(a.opinion(self.sim.opinion_source))
            if self.writer:
                self.writer.event(
                    {
                        "day": 0,
                        "agent": a.id,
                        "model": a.model.name,
                        "feed": [],
                        "reasoning": "",
                        "belief": a.belief,
                        "score": score,
                        "post": post,
                        "memory": "",
                        "prompt": user,
                        "raw": raw,
                        "error": err,
                    }
                )
        if self.writer:
            self.writer.edges(0, sorted(self.graph.edges()))
        self._record_state()

    async def step(self) -> None:
        """One day. Feeds are built from yesterday's state; all updates apply together."""
        self.day += 1
        before = self.opinions()
        feeds = [self.build_feed(a.id, before) for a in self.agents]

        async def one(a: AgentState, feed: list[tuple[int, str]]):
            system = system_prompt(a.persona, self.sim.bias)
            user = update_prompt(self.statement, a.belief, a.memory, feed, self.sim.memory_words)
            obj, raw, err = await self._ask(a.model, system, user)
            return a, feed, user, raw, err, obj

        results = await asyncio.gather(
            *(one(a, f) for a, f in zip(self.agents, feeds, strict=True))
        )

        updates = []
        for a, feed, user, raw, err, obj in results:
            belief = a.belief
            try:
                if not err:
                    belief = _clip(float(obj["belief"]))
            except (KeyError, TypeError, ValueError):
                err = f"missing or invalid 'belief' in reply: {raw[:200]!r}"
            post = str(obj.get("post", "")).strip() or a.post
            memory = str(obj.get("memory", "")).strip() or a.memory
            reasoning = str(obj.get("reasoning", "")).strip()
            updates.append((a, feed, user, raw, err, belief, post, memory, reasoning))

        scores = await asyncio.gather(*(self._score(u[6]) for u in updates))

        for (a, feed, user, raw, err, belief, post, memory, reasoning), score in zip(
            updates, scores, strict=True
        ):
            a.belief, a.post, a.memory = belief, post, memory
            a.score = score if score is not None else a.score
            a.errors += bool(err)
            a.history.append(a.opinion(self.sim.opinion_source))
            if self.writer:
                self.writer.event(
                    {
                        "day": self.day,
                        "agent": a.id,
                        "model": a.model.name,
                        "feed": [{"author": j, "post": p} for j, p in feed],
                        "reasoning": reasoning,
                        "belief": belief,
                        "score": a.score,
                        "post": post,
                        "memory": memory,
                        "prompt": user,
                        "raw": raw,
                        "error": err,
                    }
                )
        self.after_day()
        self._record_state()

    def after_day(self) -> None:
        """Hook for end-of-day network changes (rewiring)."""

    async def run(
        self, writer: RunWriter | None = None, progress: Callable[[int, dict], None] | None = None
    ) -> list[dict]:
        self.writer = writer
        if writer:
            writer.agents(
                [
                    {
                        **a.persona.to_dict(),
                        "model": a.model.name,
                        "degree": self.graph.degree(a.id),
                    }
                    for a in self.agents
                ]
            )
        await self.initialize()
        history = [{"day": 0, **self.day_metrics()}]
        if progress:
            progress(0, history[-1])
        for _ in range(self.sim.days):
            await self.step()
            history.append({"day": self.day, **self.day_metrics()})
            if writer:
                writer.flush()
            if progress:
                progress(self.day, history[-1])
        return history
