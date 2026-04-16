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


def preview_advice(count: int = 3, seed: int | None = None) -> list[str]:
    if count <= 0:
        return []
    rng = random.Random(seed)
    return [rng.choice(ADVICE) for _ in range(count)]


if __name__ == "__main__":
    print("Single:", pick_advice())
    print("Preview:")
    for tip in preview_advice(3):
        print("-", tip)
