"""Prompt templates. Kept in one place so the thesis can quote them verbatim."""

from __future__ import annotations

import numpy as np

from disensa import OPINION_MAX, OPINION_MIN

SYSTEM_TEMPLATE = """You are Agent {agent_id}, an ordinary person using a social media platform where you read posts from people you follow and share your own views.
Think and write as this person would, in plain everyday language."""

INITIAL_POST_TEMPLATE = """Topic: "{statement}"

You {label} this statement (your position: {value:+.1f} on a scale from -2 = strongly oppose to +2 = strongly support).

Write a short social media post (1-3 sentences) sharing your view.
Respond ONLY with JSON: {{"post": "<your post>"}}"""

UPDATE_TEMPLATE = """Topic: "{statement}"

What you remember from previous days: {memory}

Your current position: you {label} the statement ({value:+.1f} on a scale from -2 = strongly oppose to +2 = strongly support).

Today you read these posts from people you follow:
{feed}

Reflect on what you read, as the person you are. You may keep, soften, strengthen or change your position.
Respond ONLY with JSON:
{{"reasoning": "<2-4 sentences: how today's posts affect your view>",
 "belief": <number from -2 to 2>,
 "post": "<your new post for today, 1-3 sentences>",
 "memory": "<updated summary of what you have seen and thought so far, at most {memory_words} words>"}}"""


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


def initial_post_prompt(statement: str, value: float) -> str:
    return INITIAL_POST_TEMPLATE.format(statement=statement, label=stance_label(value), value=value)


def format_feed(feed: list[tuple[int, str]]) -> str:
    if not feed:
        return "(nothing today)"
    return "\n".join(f'- Agent {author}: "{post}"' for author, post in feed)


def update_prompt(
    statement: str, value: float, memory: str, feed: list[tuple[int, str]], memory_words: int
) -> str:
    return UPDATE_TEMPLATE.format(
        statement=statement,
        memory=memory or "(nothing yet)",
        label=stance_label(value),
        value=value,
        feed=format_feed(feed),
        memory_words=memory_words,
    )
