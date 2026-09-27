"""Unified async LLM client: one call for every provider, with disk cache and retries.

Model strings follow LiteLLM conventions, e.g.
  "anthropic/claude-sonnet-4-5", "openai/gpt-4o-mini", "xai/grok-3",
  "ollama/llama3.1", "openai/<model>" + api_base=<vLLM endpoint on INDACO>,
  "fake/echo" for tests (no network, deterministic).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import diskcache
from tenacity import retry, stop_after_attempt, wait_exponential

Message = dict[str, str]


@dataclass
class LLMResponse:
    text: str
    model: str
    cached: bool = False
    usage: dict = field(default_factory=dict)


def _cache_key(model: str, messages: list[Message], params: dict) -> str:
    blob = json.dumps({"m": model, "msgs": messages, "p": params}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class LLMClient:
    """Rate-limited, cached access to any LiteLLM-supported model."""

    def __init__(self, cache_dir: str | Path = ".cache/llm", max_concurrency: int = 16):
        self.cache = diskcache.Cache(str(cache_dir))
        self._sem: dict[str, asyncio.Semaphore] = {}
        self.max_concurrency = max_concurrency

    def _semaphore(self, model: str) -> asyncio.Semaphore:
        provider = model.split("/", 1)[0]
        if provider not in self._sem:
            self._sem[provider] = asyncio.Semaphore(self.max_concurrency)
        return self._sem[provider]

    async def complete(
        self,
        model: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
        api_base: str | None = None,
        seed: int | None = None,
        use_cache: bool = True,
    ) -> LLMResponse:
        # `seed` is part of the key so repeated stochastic samples can be cached separately.
        params = {"temperature": temperature, "max_tokens": max_tokens, "seed": seed}
        key = _cache_key(model, messages, params)
        if use_cache and key in self.cache:
            hit = self.cache[key]
            return LLMResponse(text=hit["text"], model=model, cached=True, usage=hit["usage"])

        async with self._semaphore(model):
            if model.startswith("fake/"):
                resp = _fake_completion(model, messages)
            else:
                resp = await self._call_litellm(model, messages, params, api_base)

        if use_cache:
            self.cache[key] = {"text": resp.text, "usage": resp.usage}
        return resp

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30), reraise=True)
    async def _call_litellm(
        self, model: str, messages: list[Message], params: dict, api_base: str | None
    ) -> LLMResponse:
        import litellm  # imported lazily: heavy, and not needed for fake/offline runs

        kwargs = {k: v for k, v in params.items() if v is not None}
        if api_base:
            kwargs["api_base"] = api_base
        r = await litellm.acompletion(model=model, messages=messages, **kwargs)
        usage = dict(r.usage) if getattr(r, "usage", None) else {}
        return LLMResponse(text=r.choices[0].message.content or "", model=model, usage=usage)


def _fake_completion(model: str, messages: list[Message]) -> LLMResponse:
    """Deterministic offline backend for tests and dry runs."""
    last = messages[-1]["content"]
    digest = int(hashlib.md5(last.encode()).hexdigest(), 16)
    stance = (digest % 5) - 2  # deterministic value in [-2, 2]
    text = json.dumps({"post": f"[{model}] reply to: {last[:60]}", "stance": stance})
    return LLMResponse(text=text, model=model)
