"""Tiny random advice generator."""

from __future__ import annotations

import random

ADVICE = [
    "Hydrate first, optimize later.",
    "One clean rep beats three sloppy reps.",
    "Sleep is legal performance enhancement.",
    "Slow down the eccentric and own the movement.",
    "If form breaks, ego leaves the room.",
]


def pick_advice(seed: int | None = None) -> str:
    rng = random.Random(seed)
    return rng.choice(ADVICE)


if __name__ == "__main__":
    print(pick_advice())

