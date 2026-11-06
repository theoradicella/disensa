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
