"""Prompt templates. Kept in one place so the thesis can quote them verbatim."""

from __future__ import annotations

from agorasim.personas import Persona, stance_label

CONFIRMATION_BIAS = (
    "Like most people, you tend to give more weight to arguments that agree with what you "
    "already believe and to find reasons to dismiss arguments that contradict it."
)

SYSTEM_TEMPLATE = """{persona}
You use a social media platform where you read posts from people you follow and share your own views.
Stay in character: think and write as this person would, in plain everyday language.
{bias}"""

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

SCORER_SYSTEM = "You are a careful annotator of stance in social media posts."

SCORER_TEMPLATE = """Statement: "{statement}"

Post: "{post}"

How much does the author of the post support the statement?
-2 = strongly oppose, -1 = somewhat oppose, 0 = neutral or unclear, 1 = somewhat support, 2 = strongly support.
Respond ONLY with JSON: {{"score": <integer from -2 to 2>}}"""


def system_prompt(persona: Persona, bias: str) -> str:
    return SYSTEM_TEMPLATE.format(
        persona=persona.describe(), bias=CONFIRMATION_BIAS if bias == "confirmation" else ""
    ).strip()


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


def scorer_prompt(statement: str, post: str) -> str:
    return SCORER_TEMPLATE.format(statement=statement, post=post)
