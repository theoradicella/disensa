"""Random streams: `seed` fixes the world, `replicate` the dynamics.

  seed       the network, the starting opinions and which agent gets which model. The same
             seed gives the same world for numeric and LLM runs alike, which is what pairs an
             LLM run with its numeric baselines.
  replicate  who interacts with whom at each step, and the sampling seed sent with every
             LLM call. Rerunning with another replicate keeps the world and measures how much
             of the outcome comes from the models' own randomness.

Each stream comes from its own `np.random.SeedSequence`, keyed by the stream's number, so no
two streams share random numbers (before seeding scheme 2 the starting opinions and the
interactions both used `default_rng(seed)` and drew the same numbers). The network is built
by NetworkX from `seed` itself (network.py). Numeric dynamics are deterministic and ignore
`replicate`.
"""

from __future__ import annotations

import numpy as np

SEEDING = 2  # recorded in meta.json; runs without it used the old scheme

OPINIONS, MODELS, INTERACTIONS, SAMPLING = 1, 2, 3, 4


def _entropy(seed: int) -> list[int]:
    return [abs(seed), int(seed < 0)]  # SeedSequence takes non-negative integers only


def opinions_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([*_entropy(seed), OPINIONS]))


def models_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([*_entropy(seed), MODELS]))


def interactions_rng(seed: int, replicate: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([*_entropy(seed), replicate, INTERACTIONS]))


def sampling_seed(seed: int, replicate: int) -> int:
    """The seed sent with every LLM call of a run: a non-negative 31-bit integer."""
    ss = np.random.SeedSequence([*_entropy(seed), replicate, SAMPLING])
    return int(ss.generate_state(1, dtype=np.uint32)[0] & 0x7FFFFFFF)
