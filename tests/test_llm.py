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
    msgs = [{"role": "user", "content": 'your position: +1.0 {"post": "<your post>"}'}]
    a = asyncio.run(client.complete("fake/bounded", msgs))
    b = asyncio.run(client.complete("fake/bounded", msgs))
    assert not a.cached and b.cached and a.text == b.text
    assert json.loads(a.text)["post"]
