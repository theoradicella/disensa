"""Deterministic offline LLM for tests, demos and dry runs (no network, no cost).

A fake agent embeds its position in its posts ("My position is +0.75.") and, when it reads
a feed, moves towards (or away from) the positions it sees, bounded-confidence style.
That gives realistic-looking dynamics, so the whole pipeline — engine, metrics, rewiring,
trajectory tools, visualization — can be exercised without a real model.

Behaviour is chosen by the model name: fake/<behaviour>
  conformist   moves strongly towards everyone it reads
  stubborn     barely moves
  contrarian   moves away from what it reads
  tribal       only listens to posts within 0.7, so the population splits into camps
  anything else  bounded confidence (only listens to posts within 1.5)
"""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np

from disensa import OPINION_MAX, OPINION_MIN

BEHAVIOURS = {  # name: (mu, epsilon)
    "conformist": (0.5, 4.0),
    "stubborn": (0.05, 4.0),
    "contrarian": (-0.2, 4.0),
    "tribal": (0.35, 0.7),  # only listens to people who already agree: splits into camps
}
DEFAULT = (0.3, 1.5)

_POSITION_IN_POST = re.compile(r"position is ([-+]?\d+(?:\.\d+)?)")
_OWN_POSITION = re.compile(r"\(([-+]?\d+(?:\.\d+)?) on a scale")
_INITIAL_POSITION = re.compile(r"your position: ([-+]?\d+(?:\.\d+)?)")


def _noise(text: str, scale: float = 0.08) -> float:
    h = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    return (h / 0xFFFFFFFF * 2 - 1) * scale


def _clip(v: float) -> float:
    return float(np.clip(v, OPINION_MIN, OPINION_MAX))


def fake_reply(model: str, messages: list[dict]) -> str:
    prompt = messages[-1]["content"]
    behaviour = model.split("/", 1)[1] if "/" in model else model

    if '{"score"' in prompt:  # scorer
        m = _POSITION_IN_POST.search(prompt)
        score = round(float(m.group(1))) if m else 0
        return json.dumps({"score": int(np.clip(score, -2, 2))})

    if '{"post": "<your post>"}' in prompt:  # initial post
        m = _INITIAL_POSITION.search(prompt)
        v = float(m.group(1)) if m else 0.0
        return json.dumps({"post": f"[{behaviour}] My position is {v:+.2f}."})

    # daily update
    m = _OWN_POSITION.search(prompt)
    own = float(m.group(1)) if m else 0.0
    feed_section = prompt.split("Today you read these posts", 1)[-1]
    seen = [float(x) for x in _POSITION_IN_POST.findall(feed_section)]
    mu, eps = BEHAVIOURS.get(behaviour, DEFAULT)
    close = [x for x in seen if abs(x - own) <= eps]
    if close:
        target = float(np.mean(close))
        new = own + mu * (target - own) + _noise(prompt)
        reasoning = (
            f"I read {len(seen)} posts; {len(close)} were close enough to take seriously, "
            f"averaging {target:+.2f}."
        )
    else:
        new = own + _noise(prompt, 0.03)
        reasoning = f"I read {len(seen)} posts but none changed my mind."
    new = _clip(new)
    return json.dumps(
        {
            "reasoning": reasoning,
            "belief": round(new, 3),
            "post": f"[{behaviour}] My position is {new:+.2f}.",
            "memory": f"Read {len(seen)} posts today; now at {new:+.2f}.",
        }
    )
