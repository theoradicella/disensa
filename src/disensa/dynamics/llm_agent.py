"""LLM agents, following the Social Simulation Framework of Wang et al. (COLING 2025, §3).

  step 0   every agent states a first message from its initial opinion
  step t   every agent interacts with a few others: it hears the messages they shared at
           step t-1, reasons over them with its memory, and replies with a new belief, a new
           message and an updated memory

Wang et al. frame this as users reading posts on a social network; here the framing is
domain-neutral (agents in a community exchanging messages), the mechanism is the same.

All agents of a step call their models concurrently; results are applied together, so the
order in which agents are processed never matters. A malformed reply never crashes a run:
it is logged in `error` (with its `error_kind`, see report.py) and the agent keeps its
previous state. A belief outside [-2, 2] is clipped; the number the model gave is kept in
`belief_raw` and the event is marked `clipped`.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from disensa import OPINION_MAX, OPINION_MIN
from disensa.config import LLMDynamics, ModelSpec
from disensa.llm import CallInfo, LLMCallError, LLMClient
from disensa.llm.prompts import initial_message_prompt, system_prompt, update_prompt
from disensa.seeding import interactions_rng, models_rng, sampling_seed


@dataclass
class TextState:
    """What an LLM agent carries between steps besides its opinion."""

    message: str = ""  # the latest message it shared
    memory: str = ""


@dataclass
class Reply:
    """One agent's model call: the parsed JSON, the raw text, and what went wrong if anything."""

    obj: dict = field(default_factory=dict)
    raw: str = ""
    error: str | None = None
    error_kind: str | None = None
    info: CallInfo = field(default_factory=CallInfo)


def parse_belief(value) -> float | None:
    """The belief as a finite number, or None (NaN, inf, true/false, non-numeric text)."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    if not isinstance(value, int | float):
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def assign_models(models: list[ModelSpec], n: int, seed: int) -> list[ModelSpec]:
    """Exact shares by weight (largest-remainder rounding), randomly placed on the network.

    Uses its own RNG stream (seeding.models_rng), so interactions are unaffected by the
    model mix, and the same seed places the models the same way whatever else changes.
    """
    w = np.array([m.weight for m in models], dtype=float)
    raw = w / w.sum() * n
    counts = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - counts), kind="stable")[: n - counts.sum()]:
        counts[i] += 1
    pool = [m for m, c in zip(models, counts, strict=True) for _ in range(c)]
    rng = models_rng(seed)
    return [pool[i] for i in rng.permutation(n)]


class LLMAgents:
    def __init__(self, cfg: LLMDynamics, graph: nx.Graph, seed: int = 0, replicate: int = 0):
        self.cfg = cfg
        self.graph = graph
        self.seed = seed
        self.replicate = replicate
        self.n = graph.number_of_nodes()
        self.models = assign_models(cfg.models, self.n, seed)  # the world: seed only
        self.state = [TextState() for _ in range(self.n)]
        self.rng = interactions_rng(seed, replicate)  # whom each agent interacts with
        self.sampling_seed = sampling_seed(seed, replicate)  # sent with every model call
        self.client = LLMClient(cfg.cache_dir, cfg.max_concurrency)
        # called after every reply with (step, agent, error or None); used for live progress
        self.on_reply: Callable[[int, int, str | None], None] | None = None

    def labels(self) -> list[str]:
        return [m.name for m in self.models]

    # ---------------------------------------------------------------- one model call

    async def ask(self, i: int, user: str) -> Reply:
        """One model call. Never raises for bad model output or a failing server."""
        m = self.models[i]
        messages = [
            {"role": "system", "content": system_prompt(i)},
            {"role": "user", "content": user},
        ]
        try:
            obj, raw, info = await self.client.complete_json(
                m.name,
                messages,
                temperature=m.temperature,
                top_p=m.top_p,
                top_k=m.top_k,
                think=m.think,
                max_tokens=self.cfg.max_tokens,
                api_base=m.api_base,
                seed=self.sampling_seed,
            )
            return Reply(obj, raw, info=info)
        except LLMCallError as e:  # malformed JSON after repair, or the server failed
            return Reply({}, e.raw, f"{e}"[:500], e.kind, e.info)
        except Exception as e:  # anything else on the way to the model
            return Reply({}, "", f"{type(e).__name__}: {e}"[:500], "provider")

    async def ask_all(self, t: int, prompts: list[str]) -> list[Reply]:
        """Ask every agent concurrently; report each reply to `on_reply` as it arrives."""

        async def one(i: int, p: str) -> Reply:
            reply = await self.ask(i, p)
            if self.on_reply:
                self.on_reply(t, i, reply.error)
            return reply

        return await asyncio.gather(*(one(i, p) for i, p in enumerate(prompts)))

    # ---------------------------------------------------------------- interactions

    def interactions(self, i: int) -> list[tuple[int, str]]:
        """Whom i interacts with this step: (author, message) for a random number of authors.

        Authors are, for now, a random subset of i's network neighbours.
        """
        authors = [j for j in sorted(self.graph.neighbors(i)) if self.state[j].message]
        lo, hi = self.cfg.interactions.min, self.cfg.interactions.max
        k = min(int(self.rng.integers(lo, hi + 1)), len(authors))
        if k == 0:
            return []
        chosen = sorted(self.rng.choice(authors, size=k, replace=False).tolist())
        return [(j, self.state[j].message) for j in chosen]

    # ---------------------------------------------------------------- the two phases

    async def start(self, x0: np.ndarray) -> list[dict]:
        prompts = [initial_message_prompt(self.cfg.statement, float(v)) for v in x0]
        replies = await self.ask_all(0, prompts)
        events = []
        for i, r in enumerate(replies):
            self.state[i].message = str(r.obj.get("message", "")).strip()
            events.append(self._event(0, i, [], "", self.state[i], prompts[i], r))
        return events

    async def step(self, x: np.ndarray, t: int) -> tuple[np.ndarray, list[dict]]:
        # everyone hears the messages shared at step t-1
        interactions = [self.interactions(i) for i in range(self.n)]
        prompts = [
            update_prompt(
                self.cfg.statement,
                float(x[i]),
                self.state[i].memory,
                interactions[i],
                self.cfg.memory_words,
            )
            for i in range(self.n)
        ]
        replies = await self.ask_all(t, prompts)

        new_x = x.copy()
        new_state, events = [], []
        for i, r in enumerate(replies):
            belief_raw, clipped = None, False
            if not r.error:
                if "belief" not in r.obj:
                    r.error, r.error_kind = (
                        f"no 'belief' in reply: {r.raw[:200]!r}",
                        "missing_belief",
                    )
                elif (belief_raw := parse_belief(r.obj["belief"])) is None:
                    r.error, r.error_kind = (
                        f"'belief' is not a finite number: {r.obj['belief']!r}"[:300],
                        "invalid_belief",
                    )
                else:
                    new_x[i] = min(max(belief_raw, OPINION_MIN), OPINION_MAX)
                    clipped = bool(new_x[i] != belief_raw)
            old = self.state[i]
            s = TextState(
                message=str(r.obj.get("message", "")).strip() or old.message,
                memory=str(r.obj.get("memory", "")).strip() or old.memory,
            )
            reasoning = str(r.obj.get("reasoning", "")).strip()
            new_state.append(s)
            events.append(
                self._event(t, i, interactions[i], reasoning, s, prompts[i], r, belief_raw, clipped)
            )
        self.state = new_state  # applied together: synchronous update
        return new_x, events

    def _event(
        self,
        t,
        i,
        interactions,
        reasoning,
        s: TextState,
        prompt,
        r: Reply,
        belief_raw=None,
        clipped=False,
    ) -> dict:
        return {
            "step": t,
            "agent": i,
            "model": self.models[i].name,
            "interactions": [{"author": j, "message": m} for j, m in interactions],
            "reasoning": reasoning,
            "message": s.message,
            "memory": s.memory,
            "prompt": prompt,
            "raw": r.raw,
            "error": r.error,
            "error_kind": r.error_kind,
            "belief_raw": belief_raw,
            "clipped": clipped,
            **r.info.to_dict(),  # latency_s, prompt_tokens, completion_tokens, cached
        }
