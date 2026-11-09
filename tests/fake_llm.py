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

Broken models, for the run report and the failure guard (step 0 still works for the
belief ones, so the run gets to step 1):
  badjson      never replies with JSON, not even when asked to fix it
  nobelief     valid JSON without a belief
  nanbelief    "belief": NaN
  textbelief   "belief": "quite a lot"
  outofrange   "belief": 5 (clipped to 2)
  halves       a bounded-confidence agent that reports whole or half numbers only
  down         the server cannot be reached (raised in conftest.py)

As a scorer (scorer.py), a fake model reads the position out of the text and rounds it;
fake/unclear always answers "unclear", fake/offscale answers 7.
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
_OWN_POSITION = re.compile(r"the statement \(([-+]?\d+(?:\.\d+)?)\)")
_INITIAL_POSITION = re.compile(r"your position: ([-+]?\d+(?:\.\d+)?)")


def _noise(text: str, scale: float = 0.08) -> float:
    h = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    return (h / 0xFFFFFFFF * 2 - 1) * scale


def _clip(v: float) -> float:
    return float(np.clip(v, OPINION_MIN, OPINION_MAX))


def fake_reply(model: str, messages: list[dict]) -> str:
    prompt = messages[-1]["content"]
    behaviour = model.split("/", 1)[1] if "/" in model else model

    if behaviour == "badjson":
        return "I would rather not answer in that format."
    if "What is the author's position on this statement" in prompt:  # the scorer
        if behaviour == "unclear":
            return json.dumps({"score": "unclear"})
        if behaviour == "offscale":
            return json.dumps({"score": 7})
        m = _POSITION_IN_MESSAGE.search(prompt)
        return json.dumps({"score": round(float(m.group(1))) if m else "unclear"})
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
    broken = {"nobelief": None, "nanbelief": float("nan"), "textbelief": "quite a lot"}
    if behaviour in broken:
        reply = {"reasoning": reasoning, "message": f"[{behaviour}] My position is {own:+.2f}."}
        if broken[behaviour] is not None:
            reply["belief"] = broken[behaviour]
        return json.dumps(reply)  # NaN is written as the bare literal NaN
    if behaviour == "outofrange":
        new = 5
    if behaviour == "halves":
        new = round(new * 2) / 2
    return json.dumps(
        {
            "reasoning": reasoning,
            "belief": round(new, 3),
            "message": f"[{behaviour}] My position is {new:+.2f}.",
            "memory": f"{len(seen)} interactions this step; now at {new:+.2f}.",
        }
    )
