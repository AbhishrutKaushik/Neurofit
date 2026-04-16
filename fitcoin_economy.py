"""
fitcoin_economy.py — Phase 6: FitCoin Gamification Engine
==========================================================

Reward calculation:
    Total_Coins = (Reps × Form_Score) × Velocity_Multiplier

Rewards shop:
    Queries the Open Food Facts API for real-world dietary supplements
    and assigns a FitCoin price to each product for the in-app store.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Price range for mock FitCoin cost assignment
_MIN_COIN_PRICE = 50
_MAX_COIN_PRICE = 500
_COIN_STEP = 25

_STORE_QUERIES = ["whey protein", "creatine", "protein bar", "bcaa", "electrolyte"]
_RESULTS_PER_QUERY = 4
_PLACEHOLDER_IMG = "https://placehold.co/200x200?text=No+Image"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Reward calculation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def calculate_rewards(
    reps: int,
    form_score: float,
    velocity_multiplier: float = 1.0,
) -> float:
    """Compute FitCoins earned for a set.

    Formula:  Total_Coins = (Reps × Form_Score) × Velocity_Multiplier

    Parameters
    ----------
    reps : int
        Number of completed reps.
    form_score : float
        Form quality score from the SR-RAG pipeline (e.g. 0-10 scale).
    velocity_multiplier : float
        Tempo bonus.  1.0 = normal, >1.0 = controlled eccentric.

    Returns
    -------
    float
        Total FitCoins earned (always >= 0).
    """
    coins = (max(0, reps) * max(0.0, form_score)) * max(0.0, velocity_multiplier)
    return round(coins, 2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rewards shop — Open Food Facts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _deterministic_price(product_name: str) -> int:
    """Assign a stable FitCoin price based on the product name hash.

    Using a hash ensures the same product always gets the same price
    across page reloads (no flickering in the Streamlit UI).
    """
    digest = int(hashlib.md5(product_name.encode("utf-8")).hexdigest(), 16)
    steps = (_MAX_COIN_PRICE - _MIN_COIN_PRICE) // _COIN_STEP
    return _MIN_COIN_PRICE + (digest % (steps + 1)) * _COIN_STEP


def fetch_store_rewards(
    queries: Optional[List[str]] = None,
    max_per_query: int = _RESULTS_PER_QUERY,
) -> List[Dict[str, Any]]:
    """Query Open Food Facts for dietary supplements and return a
    store-ready list of products with FitCoin prices.

    Each dict in the returned list contains:
        name       : str   — product display name
        image_url  : str   — product image (or placeholder)
        fitcoin_cost : int — price in FitCoins
        category   : str   — the search query that found it
        barcode    : str   — Open Food Facts barcode

    Falls back to a hardcoded mock catalog on network failure so the
    Streamlit UI always has something to render.
    """
    try:
        from openfoodfacts import API
        api = API(user_agent="Neuro-Fit/1.0 (fitness gamification store)")
    except ImportError:
        logger.warning("openfoodfacts SDK not installed — using mock catalog.")
        return _mock_catalog()

    queries = queries or _STORE_QUERIES
    products: List[Dict[str, Any]] = []
    seen_names: set = set()

    for query in queries:
        try:
            response = api.product.text_search(query, page_size=max_per_query)
            items = response.get("products", []) if isinstance(response, dict) else []
        except Exception as exc:
            logger.warning("Open Food Facts search failed for '%s': %s", query, exc)
            continue

        for item in items[:max_per_query]:
            name = item.get("product_name", "").strip()
            if not name or name.lower() in seen_names:
                continue
            seen_names.add(name.lower())

            image = (
                item.get("image_front_small_url")
                or item.get("image_front_url")
                or item.get("image_url")
                or _PLACEHOLDER_IMG
            )

            products.append({
                "name": name,
                "image_url": image,
                "fitcoin_cost": _deterministic_price(name),
                "category": query,
                "barcode": item.get("code", ""),
            })

    if not products:
        logger.warning("No products found from Open Food Facts — using mock catalog.")
        return _mock_catalog()

    return products


def _mock_catalog() -> List[Dict[str, Any]]:
    """Hardcoded fallback so the shop UI is never empty."""
    catalog = [
        {"name": "Optimum Nutrition Gold Standard Whey", "category": "whey protein"},
        {"name": "Dymatize ISO100 Hydrolyzed Whey", "category": "whey protein"},
        {"name": "MuscleTech Nitro-Tech Whey Protein", "category": "whey protein"},
        {"name": "Creapure Creatine Monohydrate", "category": "creatine"},
        {"name": "Optimum Nutrition Micronized Creatine", "category": "creatine"},
        {"name": "Gatorade Endurance Electrolyte Mix", "category": "electrolyte"},
        {"name": "KIND Protein Bar Dark Chocolate", "category": "protein bar"},
        {"name": "Xtend BCAA Recovery Powder", "category": "bcaa"},
    ]
    return [
        {
            "name": p["name"],
            "image_url": _PLACEHOLDER_IMG,
            "fitcoin_cost": _deterministic_price(p["name"]),
            "category": p["category"],
            "barcode": "",
        }
        for p in catalog
    ]
