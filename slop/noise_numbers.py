"""Generate noisy integer data and show basic stats."""

from __future__ import annotations

import random
from statistics import mean


def make_numbers(n: int = 20, low: int = 10, high: int = 99) -> list[int]:
    if n <= 0:
        return []
    rng = random.Random()
    return [rng.randint(low, high) for _ in range(n)]


def summarize(values: list[int]) -> dict[str, float]:
    if not values:
        return {"count": 0, "avg": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": float(len(values)),
        "avg": round(mean(values), 2),
        "min": float(min(values)),
        "max": float(max(values)),
    }


if __name__ == "__main__":
    nums = make_numbers()
    print("values:", nums)
    print("summary:", summarize(nums))

