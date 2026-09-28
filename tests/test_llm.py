import json

import pytest

from disensa.llm import LLMClient, parse_json


async def test_fake_backend_is_deterministic_and_cached(tmp_path):
    client = LLMClient(cache_dir=tmp_path)
    msgs = [
        {
            "role": "user",
            "content": 'Respond ONLY with JSON: {"score": x}. Post: "My position is +1.40."',
        }
    ]
    a = await client.complete("fake/x", msgs)
    b = await client.complete("fake/x", msgs)
    assert not a.cached and b.cached
    assert a.text == b.text
    assert json.loads(a.text)["score"] == 1


async def test_different_models_are_cached_separately(tmp_path):
    client = LLMClient(cache_dir=tmp_path)
    msgs = [{"role": "user", "content": "hello"}]
    await client.complete("fake/one", msgs)
    b = await client.complete("fake/two", msgs)
    assert not b.cached


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"belief": 1}', {"belief": 1}),
        (
            'Sure! Here it is:\n```json\n{"belief": -1.5, "post": "hi"}\n```',
            {"belief": -1.5, "post": "hi"},
        ),
        ('{"a": {"b": 2}, "c": 3,}', {"a": {"b": 2}, "c": 3}),  # trailing comma
        ('noise {not json} then {"score": 2}', {"score": 2}),
    ],
)
def test_parse_json_tolerates_messy_replies(text, expected):
    assert parse_json(text) == expected


def test_parse_json_raises_without_object():
    with pytest.raises(ValueError):
        parse_json("I would rather not answer in JSON.")
