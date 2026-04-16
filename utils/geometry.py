"""3-D geometry helpers shared across perception and reasoning layers."""

from __future__ import annotations

import math

import numpy as np

_EPS = 1e-8


def angle_between_vectors_deg(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> float:
    """Angle at vertex *b* formed by points (a, b, c), in degrees.

    Uses the standard dot-product formula:

        cos(theta) = (BA . BC) / (|BA| |BC|)

    Returns a value in [0, 180].
    """
    ba = a - b
    bc = c - b
    norm_product = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm_product < _EPS:
        return 0.0
    cos_theta = float(np.clip(np.dot(ba, bc) / norm_product, -1.0, 1.0))
    return math.degrees(math.acos(cos_theta))
