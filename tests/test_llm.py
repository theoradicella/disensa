import json

from agorasim.llm import LLMClient


async def test_fake_backend_is_deterministic_and_cached(tmp_path):
    client = LLMClient(cache_dir=tmp_path)
    msgs = [{"role": "user", "content": "What do you think about remote work?"}]
    a = await client.complete("fake/echo", msgs)
    b = await client.complete("fake/echo", msgs)
    assert not a.cached and b.cached
    assert a.text == b.text
    assert -2 <= json.loads(a.text)["stance"] <= 2


async def test_different_models_are_cached_separately(tmp_path):
    client = LLMClient(cache_dir=tmp_path)
    msgs = [{"role": "user", "content": "hello"}]
    a = await client.complete("fake/one", msgs)
    b = await client.complete("fake/two", msgs)
    assert not b.cached and a.text != b.text
