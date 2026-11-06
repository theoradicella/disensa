import asyncio
import json

import pytest

from disensa.llm import LLMClient, parse_json


@pytest.mark.parametrize(
    "text",
    [
        '{"belief": 1}',
        'Sure! ```json\n{"belief": 1}\n```',
        'Here: {"belief": 1,}',
    ],
)
def test_parse_json_tolerates_common_slips(text):
    assert parse_json(text) == {"belief": 1}


def test_parse_json_rejects_prose():
    with pytest.raises(ValueError):
        parse_json("I think I agree.")


def test_cache_makes_reruns_free(tmp_path):
    client = LLMClient(tmp_path / "cache")
    msgs = [{"role": "user", "content": 'your position: +1.0 {"message": "<your message>"}'}]
    a = asyncio.run(client.complete("fake/bounded", msgs))
    b = asyncio.run(client.complete("fake/bounded", msgs))
    assert not a.cached and b.cached and a.text == b.text
    assert json.loads(a.text)["message"]


def test_vllm_connector_gets_json_mode(tmp_path, litellm_kwargs):
    client = LLMClient(tmp_path / "cache")
    msgs = [{"role": "user", "content": "hi"}]
    asyncio.run(
        client.complete("hosted_vllm/x", msgs, api_base="http://localhost:8000/v1", json_mode=True)
    )
    (kw,) = litellm_kwargs
    assert kw["model"] == "hosted_vllm/x"
    assert kw["api_base"] == "http://localhost:8000/v1"
    assert kw["response_format"] == {"type": "json_object"}


def test_sampling_settings_are_sent_and_cached_separately(tmp_path, litellm_kwargs):
    client = LLMClient(tmp_path / "cache")
    msgs = [{"role": "user", "content": "hi"}]
    a = asyncio.run(client.complete("ollama_chat/x", msgs, temperature=0.7, top_p=0.9, top_k=40))
    (kw,) = litellm_kwargs
    assert (kw["temperature"], kw["top_p"], kw["top_k"]) == (0.7, 0.9, 40)
    b = asyncio.run(client.complete("ollama_chat/x", msgs, temperature=0.7, top_p=0.5, top_k=40))
    assert not a.cached and not b.cached and len(litellm_kwargs) == 2  # a new cache key
    c = asyncio.run(client.complete("ollama_chat/x", msgs, temperature=0.7, top_p=0.5, top_k=40))
    assert c.cached


def test_agents_send_their_model_sampling_settings(make_cfg, litellm_kwargs):
    from disensa.engine import run

    dyn = make_cfg().dynamics.model_dump(mode="json")
    dyn["models"] = [{"name": "ollama_chat/x", "api_base": "http://h", "top_p": 0.8, "top_k": 20}]
    asyncio.run(run(make_cfg(dynamics=dyn, steps=1)))
    assert {(k["temperature"], k["top_p"], k["top_k"]) for k in litellm_kwargs} == {(0.7, 0.8, 20)}


def test_parse_json_skips_a_think_block():
    text = '<think>\nThey said {"belief": 2}, but {hm} I disagree.\n</think>\n{"belief": -1}'
    assert parse_json(text) == {"belief": -1}


@pytest.mark.parametrize(
    "model,expected",
    [
        ("ollama_chat/qwen3:4b", {"think": False}),
        (
            "hosted_vllm/Qwen/Qwen3-8B",
            {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        ),
        ("fake/other", {}),
    ],
)
def test_thinking_is_switched_off_per_provider(tmp_path, litellm_kwargs, model, expected):
    client = LLMClient(tmp_path / "cache")
    asyncio.run(client.complete(model, [{"role": "user", "content": "hi"}], think=False))
    (kw,) = litellm_kwargs
    assert {k: kw[k] for k in ("think", "extra_body") if k in kw} == expected


def test_agents_send_think_false_by_default(make_cfg, litellm_kwargs):
    from disensa.engine import run

    dyn = make_cfg().dynamics.model_dump(mode="json")
    dyn["models"] = [{"name": "ollama_chat/qwen3:4b", "api_base": "http://h"}]
    cfg = make_cfg(dynamics=dyn, steps=1)
    assert cfg.dynamics.models[0].think is False
    asyncio.run(run(cfg))
    assert litellm_kwargs and all(k["think"] is False for k in litellm_kwargs)


def test_think_is_part_of_the_cache_key(tmp_path, litellm_kwargs):
    client = LLMClient(tmp_path / "cache")
    msgs = [{"role": "user", "content": "hi"}]
    asyncio.run(client.complete("ollama_chat/x", msgs, think=False))
    b = asyncio.run(client.complete("ollama_chat/x", msgs, think=True))
    assert not b.cached and len(litellm_kwargs) == 2


def test_one_client_serves_several_event_loops(tmp_path, monkeypatch):
    """`disensa score` on a sweep reuses one client across asyncio.run() calls: its
    concurrency limit must not stay bound to the first loop (found in a live test)."""
    from disensa.llm import client as llm_client

    async def slow(self, model, messages, params, api_base, json_mode):
        await asyncio.sleep(0.01)  # so requests really wait on the limit
        return llm_client.LLMResponse(text='{"a": 1}', model=model)

    monkeypatch.setattr(llm_client.LLMClient, "_call_litellm", slow)
    client = LLMClient(tmp_path / "cache", max_concurrency=1)

    async def many(tag):
        msgs = [[{"role": "user", "content": f"{tag} {i}"}] for i in range(4)]
        return await asyncio.gather(*(client.complete("fake/x", m) for m in msgs))

    for tag in ("first loop", "second loop"):
        assert len(asyncio.run(many(tag))) == 4
