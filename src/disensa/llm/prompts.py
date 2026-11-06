"""Prompt templates. Kept in one place so the thesis can quote them verbatim."""

from __future__ import annotations

import numpy as np

from disensa import OPINION_MAX, OPINION_MIN

SYSTEM_TEMPLATE = """You are Agent {agent_id}, a member of a community. You interact with other people and share your views with them.
Think and speak as this person would, in plain everyday language."""

INITIAL_MESSAGE_TEMPLATE = """Topic: "{statement}"

You {label} this statement (your position: {value:+.1f} on a scale from -2 = strongly oppose to +2 = strongly support).

Write a short message (1-3 sentences) stating your view.
Respond ONLY with JSON: {{"message": "<your message>"}}"""

UPDATE_TEMPLATE = """Topic: "{statement}"

What you remember from previous interactions: {memory}

Your current position: you {label} the statement ({value:+.1f} on a scale from -2 = strongly oppose to +2 = strongly support).

In this step you interacted with:
{interactions}

Reflect on these interactions, as the person you are. You may keep, soften, strengthen or change your position.
Respond ONLY with JSON:
{{"reasoning": "<2-4 sentences: how these interactions affect your view>",
 "belief": <number from -2 to 2>,
 "message": "<your new message, 1-3 sentences>",
 "memory": "<updated summary of what you have heard and thought so far, at most {memory_words} words>"}}"""


# The separate opinion scorer (scorer.py): a measurement after the run, never fed back into it.
# Adapted from the classifier prompt of Chuang et al. (2024, App. I), with disensa's wording:
# a text by a member of a community instead of "a statement made by a person after seeing a
# tweet", and disensa's five stance labels. "unclear" is allowed, so the scorer is not forced
# to call a text that takes no position neutral.
SCORER_SYSTEM = "You rate the opinions people express. Answer only with the JSON asked for."

SCORER_TEMPLATE = """Here is a text written by a member of a community.
---
{text}
---
What is the author's position on this statement: "{statement}"

Rate it on a 5-point scale:
-2: strongly oppose, -1: somewhat oppose, 0: neutral, 1: somewhat support, 2: strongly support.
If the text does not show the author's position on the statement, answer "unclear".
Respond ONLY with JSON: {{"score": <one of -2, -1, 0, 1, 2, or "unclear">}}"""


def stance_label(value: float) -> str:
    """Verbal label for an opinion on [-2, 2]."""
    v = round(float(np.clip(value, OPINION_MIN, OPINION_MAX)))
    return {
        -2: "strongly oppose",
        -1: "somewhat oppose",
        0: "are neutral / undecided about",
        1: "somewhat support",
        2: "strongly support",
    }[v]


def system_prompt(agent_id: int) -> str:
    return SYSTEM_TEMPLATE.format(agent_id=agent_id)


def initial_message_prompt(statement: str, value: float) -> str:
    return INITIAL_MESSAGE_TEMPLATE.format(
        statement=statement, label=stance_label(value), value=value
    )


def format_interactions(interactions: list[tuple[int, str]]) -> str:
    if not interactions:
        return "(nobody in this step)"
    return "\n".join(f'- Agent {author} said: "{message}"' for author, message in interactions)


def update_prompt(
    statement: str,
    value: float,
    memory: str,
    interactions: list[tuple[int, str]],
    memory_words: int,
) -> str:
    return UPDATE_TEMPLATE.format(
        statement=statement,
        memory=memory or "(nothing yet)",
        label=stance_label(value),
        value=value,
        interactions=format_interactions(interactions),
        memory_words=memory_words,
    )


def scorer_prompt(statement: str, text: str) -> str:
    return SCORER_TEMPLATE.format(statement=statement, text=text)
