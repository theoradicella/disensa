"""One async client for local LLMs (via LiteLLM), with a disk cache.

Model strings follow LiteLLM conventions, e.g.
  "ollama_chat/llama3.2:3b"            local Ollama (default http://localhost:11434)
  "openai/<model>" + api_base=<url>    any OpenAI-compatible server, e.g. vLLM on INDACO
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import diskcache

Message = dict[str, str]

# Providers where LiteLLM's JSON mode is reliable; others rely on the prompt + parser.
_JSON_MODE_PROVIDERS = ("ollama", "ollama_chat", "openai")


@dataclass
class LLMResponse:
    text: str
    model: str
    cached: bool = False
    usage: dict = field(default_factory=dict)


def _cache_key(model: str, messages: list[Message], params: dict) -> str:
    blob = json.dumps({"m": model, "msgs": messages, "p": params}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def parse_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model reply (tolerates prose and code fences)."""
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        # common small-model slip: trailing commas
                        try:
                            return json.loads(re.sub(r",\s*([}\]])", r"\1", candidate))
                        except json.JSONDecodeError:
                            break
        start = text.find("{", start + 1)
    raise ValueError(f"no JSON object found in reply: {text[:200]!r}")


class LLMClient:
    """Rate-limited, cached access to any LiteLLM-supported model."""

    def __init__(self, cache_dir: str | Path = ".cache/llm", max_concurrency: int = 16):
        self.cache = diskcache.Cache(str(cache_dir))
        self._sem: dict[str, asyncio.Semaphore] = {}
        self.max_concurrency = max_concurrency
        self.calls = 0
        self.cache_hits = 0

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
        json_mode: bool = False,
        use_cache: bool = True,
    ) -> LLMResponse:
        # `seed` is part of the key so repeated stochastic samples can be cached separately.
        params = {"temperature": temperature, "max_tokens": max_tokens, "seed": seed}
        key = _cache_key(model, messages, params)
        self.calls += 1
        if use_cache and key in self.cache:
            self.cache_hits += 1
            hit = self.cache[key]
            return LLMResponse(text=hit["text"], model=model, cached=True, usage=hit["usage"])

        async with self._semaphore(model):
            resp = await self._call_litellm(model, messages, params, api_base, json_mode)

        if use_cache:
            self.cache[key] = {"text": resp.text, "usage": resp.usage}
        return resp

    async def complete_json(
        self, model: str, messages: list[Message], *, retries: int = 1, **kwargs
    ) -> tuple[dict[str, Any], str]:
        """Complete and parse JSON. On a parse failure, ask the model once to fix its reply.

        Returns (parsed_object, raw_text). Raises ValueError if it never parses.
        """
        resp = await self.complete(model, messages, json_mode=True, **kwargs)
        try:
            return parse_json(resp.text), resp.text
        except ValueError:
            if retries <= 0:
                raise
        repair = messages + [
            {"role": "assistant", "content": resp.text},
            {
                "role": "user",
                "content": "That was not valid JSON. Reply again with ONLY the JSON object.",
            },
        ]
        return await self.complete_json(model, repair, retries=retries - 1, **kwargs)

    async def _call_litellm(
        self,
        model: str,
        messages: list[Message],
        params: dict,
        api_base: str | None,
        json_mode: bool,
    ) -> LLMResponse:
        import litellm  # imported lazily: heavy

        litellm.suppress_debug_info = True
        litellm.drop_params = True  # a server that rejects a setting (e.g. `seed`) drops it instead of failing
        kwargs = {k: v for k, v in params.items() if v is not None}
        kwargs["num_retries"] = 4  # LiteLLM retries rate limits / transient errors with backoff
        if api_base:
            kwargs["api_base"] = api_base
        if json_mode and model.split("/", 1)[0] in _JSON_MODE_PROVIDERS:
            kwargs["response_format"] = {"type": "json_object"}
        r = await litellm.acompletion(model=model, messages=messages, **kwargs)
        usage = dict(r.usage) if getattr(r, "usage", None) else {}
        return LLMResponse(text=r.choices[0].message.content or "", model=model, usage=usage)
