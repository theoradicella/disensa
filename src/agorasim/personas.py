"""Seeded persona generation, following Wang et al. (§4.1).

Agents are identified only by an index (no names, to avoid name bias). Gender is random,
age uniform in [18, 64], and each Big Five trait is independently positive or negative.
Personas depend only on the run seed, never on which LLM an agent uses, so the same
population can be replayed under different model mixes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from agorasim import OPINION_MAX, OPINION_MIN

BIG_FIVE = {
    "openness": ("open to new ideas and experiences", "prefers the familiar and conventional"),
    "conscientiousness": ("organised and dependable", "spontaneous and careless"),
    "extraversion": ("outgoing and talkative", "reserved and quiet"),
    "agreeableness": ("cooperative and trusting", "critical and sceptical of others"),
    "neuroticism": ("anxious and easily upset", "calm and emotionally stable"),
}
GENDERS = ("male", "female")
EDUCATION = (
    "high school diploma",
    "vocational training",
    "bachelor's degree",
    "master's degree",
    "doctorate",
)


@dataclass(frozen=True)
class Persona:
    agent_id: int
    gender: str
    age: int
    education: str
    traits: dict[str, bool]  # trait -> True (positive pole) / False (negative pole)
    initial_opinion: float

    def describe(self) -> str:
        trait_text = "; ".join(
            BIG_FIVE[t][0] if positive else BIG_FIVE[t][1] for t, positive in self.traits.items()
        )
        return (
            f"You are Agent {self.agent_id}, a {self.age}-year-old {self.gender} "
            f"with a {self.education}. Personality: {trait_text}."
        )

    def to_dict(self) -> dict:
        return asdict(self)


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


def generate_personas(initial_opinions: np.ndarray, seed: int) -> list[Persona]:
    """One persona per initial opinion. Uses its own RNG stream (seed + 1_000)."""
    rng = np.random.default_rng(seed + 1_000)
    personas = []
    for i, op in enumerate(initial_opinions):
        personas.append(
            Persona(
                agent_id=i,
                gender=str(rng.choice(GENDERS)),
                age=int(rng.integers(18, 65)),
                education=str(rng.choice(EDUCATION)),
                traits={t: bool(rng.random() < 0.5) for t in BIG_FIVE},
                initial_opinion=float(op),
            )
        )
    return personas
