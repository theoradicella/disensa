"""Offline stand-in for a language model, used only by the tests (no network, no cost).

A fake agent embeds its position in its messages ("My position is +0.75.") and, after its
interactions, moves towards (or away from) the positions it heard, bounded-confidence style.
That gives realistic-looking dynamics, so the whole pipeline (engine, metrics, storage, the
lab) can be exercised without a real model.

Behaviour is chosen by the model name: fake/<behaviour>
  conformist   moves strongly towards everyone it interacts with
  stubborn     barely moves
  contrarian   moves away from what it hears
  tribal       only listens to messages within 0.7, so the population splits into camps
  anything else  bounded confidence (only listens to messages within 1.5)
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

_POSITION_IN_MESSAGE = re.compile(r"position is ([-+]?\d+(?:\.\d+)?)")
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

    if '{"message": "<your message>"}' in prompt:  # initial message
        m = _INITIAL_POSITION.search(prompt)
        v = float(m.group(1)) if m else 0.0
        return json.dumps({"message": f"[{behaviour}] My position is {v:+.2f}."})

    # update after interactions
    m = _OWN_POSITION.search(prompt)
    own = float(m.group(1)) if m else 0.0
    heard = prompt.split("In this step you interacted with", 1)[-1]
    seen = [float(x) for x in _POSITION_IN_MESSAGE.findall(heard)]
    mu, eps = BEHAVIOURS.get(behaviour, DEFAULT)
    close = [x for x in seen if abs(x - own) <= eps]
    if close:
        target = float(np.mean(close))
        new = own + mu * (target - own) + _noise(prompt)
        reasoning = (
            f"I interacted with {len(seen)} people; {len(close)} were close enough to take "
            f"seriously, averaging {target:+.2f}."
        )
    else:
        new = own + _noise(prompt, 0.03)
        reasoning = f"I interacted with {len(seen)} people but none changed my mind."
    new = _clip(new)
    return json.dumps(
        {
            "reasoning": reasoning,
            "belief": round(new, 3),
            "message": f"[{behaviour}] My position is {new:+.2f}.",
            "memory": f"{len(seen)} interactions this step; now at {new:+.2f}.",
        }
    )
