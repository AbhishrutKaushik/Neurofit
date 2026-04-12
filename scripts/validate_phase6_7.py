"""
validate_phase6_7.py — Phase 6 & 7 Validation Script
======================================================

Phase 6 checks:
    1. FitCoin math: calculate_rewards(10, 8, 1.2) == 96
    2. Open Food Facts API: search "whey protein", verify product names + image URLs

Phase 7 checks:
    3. TTS non-blocking: speak() must return in < 50 ms (audio plays in background)
    4. TTS engine initialises without crashing

Run:
    python scripts/validate_phase6_7.py
    python scripts/validate_phase6_7.py --verbose
    python scripts/validate_phase6_7.py --skip-network   (skip Open Food Facts API call)

Phase 7 UI-level checks (manual — see checklist printed at the end).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DIVIDER = "=" * 60
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"


def section(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 6, Test 1: FitCoin Math
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_fitcoin_math(verbose: bool) -> bool:
    section("PHASE 6 · TEST 1 — FitCoin Math")
    from fitcoin_economy import calculate_rewards

    reps, form_score, velocity_multiplier = 10, 8, 1.2
    expected = 96.0
    result = calculate_rewards(reps, form_score, velocity_multiplier)

    print(f"  Formula : (Reps × Form_Score) × Velocity_Multiplier")
    print(f"  Input   : Reps={reps}, Form_Score={form_score}, Velocity_Multiplier={velocity_multiplier}")
    print(f"  Expected: {expected}")
    print(f"  Got     : {result}")

    if result == expected:
        print(f"\n  {PASS}  calculate_rewards({reps}, {form_score}, {velocity_multiplier}) == {expected}")
        return True
    else:
        print(f"\n  {FAIL}  Expected {expected}, got {result}")
        return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 6, Test 2: Open Food Facts API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_openfoodfacts_api(verbose: bool) -> bool:
    section("PHASE 6 · TEST 2 — Open Food Facts API")
    from fitcoin_economy import fetch_store_rewards

    print("  Querying Open Food Facts for 'whey protein' ...")
    t0 = time.perf_counter()
    products = fetch_store_rewards(queries=["whey protein"], max_per_query=5)
    elapsed = time.perf_counter() - t0
    print(f"  API call took {elapsed:.2f}s")

    if not products:
        print(f"\n  {FAIL}  No products returned.")
        return False

    print(f"  Products returned: {len(products)}")
    ok = True

    for i, p in enumerate(products):
        name = p.get("name", "")
        image = p.get("image_url", "")
        cost = p.get("fitcoin_cost", 0)

        if verbose:
            print(f"\n  [{i+1}] {name}")
            print(f"      Image URL   : {image[:80]}{'...' if len(image) > 80 else ''}")
            print(f"      FitCoin Cost: {cost}")
            print(f"      Category    : {p.get('category', '?')}")
            print(f"      Barcode     : {p.get('barcode', '?')}")

        if not name:
            print(f"  {FAIL}  Product {i} has empty name")
            ok = False
        if not image or not image.startswith("http"):
            print(f"  {WARN}  Product '{name}' missing valid image URL (got: {image[:50]})")

    has_names = all(p.get("name") for p in products)
    has_costs = all(isinstance(p.get("fitcoin_cost"), int) for p in products)

    checks = []
    if has_names:
        checks.append("names present")
    if has_costs:
        checks.append("fitcoin_cost assigned")
    if len(products) >= 1:
        checks.append(f"{len(products)} product(s)")

    if ok:
        print(f"\n  {PASS}  Open Food Facts → [{', '.join(checks)}]")
    else:
        print(f"\n  {FAIL}  Some products had issues")

    if verbose:
        print(f"\n  Full JSON response:")
        print(json.dumps(products, indent=2, ensure_ascii=False))

    return ok


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 7, Test 1: TTS Engine Init
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_tts_init(verbose: bool) -> bool:
    section("PHASE 7 · TEST 1 — TTS Engine Initialisation")
    from utils.audio import TTSEngine

    t0 = time.perf_counter()
    engine = TTSEngine()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"  TTSEngine() created in {elapsed_ms:.0f} ms")
    print(f"  pyttsx3 available: {engine.available}")

    if engine.available:
        print(f"\n  {PASS}  TTS engine initialised successfully")
        return True
    else:
        print(f"\n  {WARN}  pyttsx3 not available — TTS will be silent.")
        print(f"         Install with: pip install pyttsx3")
        print(f"         (This is OK on Mac/CI — test on lab PC with speakers)")
        return True  # non-fatal


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 7, Test 2: TTS Non-Blocking (speak returns immediately)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SPEAK_RETURN_LIMIT_MS = 50  # speak() must return within this


def test_tts_nonblocking(verbose: bool) -> bool:
    section("PHASE 7 · TEST 2 — TTS Non-Blocking Check")
    from utils.audio import TTSEngine

    engine = TTSEngine()

    if not engine.available:
        print(f"  {WARN}  pyttsx3 not installed — skipping non-blocking test.")
        return True

    test_text = "Great job on that set, your form is looking much better."
    print(f"  Speaking: \"{test_text}\"")
    print(f"  Measuring time for speak() to return ...")

    t0 = time.perf_counter()
    engine.speak(test_text)
    return_ms = (time.perf_counter() - t0) * 1000

    print(f"  speak() returned in {return_ms:.1f} ms (limit: {SPEAK_RETURN_LIMIT_MS} ms)")

    if return_ms <= SPEAK_RETURN_LIMIT_MS:
        print(f"\n  {PASS}  speak() is non-blocking ({return_ms:.1f} ms)")
    else:
        print(f"\n  {FAIL}  speak() took {return_ms:.1f} ms — it is BLOCKING the main thread!")
        print("         The WebRTC video feed would freeze during audio playback.")
        engine.stop()
        return False

    # Let the audio actually play so the user can hear it
    print("  (Audio is playing in background — you should hear the voice now)")
    print("  Waiting 4 seconds for audio to finish ...")
    time.sleep(4)
    engine.stop()
    print(f"  TTS engine stopped cleanly.")

    return True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 7, Test 3: Multiple queued speaks (queue doesn't crash)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_tts_queue(verbose: bool) -> bool:
    section("PHASE 7 · TEST 3 — TTS Queue (multiple cues)")
    from utils.audio import TTSEngine

    engine = TTSEngine()

    if not engine.available:
        print(f"  {WARN}  pyttsx3 not installed — skipping queue test.")
        return True

    cues = [
        "Keep your core braced.",
        "Slow down the eccentric phase.",
        "Great form, keep it up!",
    ]

    print(f"  Queuing {len(cues)} coaching cues rapidly ...")
    t0 = time.perf_counter()
    for cue in cues:
        engine.speak(cue)
    total_ms = (time.perf_counter() - t0) * 1000
    print(f"  All {len(cues)} calls to speak() returned in {total_ms:.1f} ms total")

    if total_ms < SPEAK_RETURN_LIMIT_MS * len(cues):
        print(f"\n  {PASS}  Queue accepts multiple cues without blocking")
    else:
        print(f"\n  {FAIL}  Queuing was too slow — possible blocking")
        engine.stop()
        return False

    print("  (Audio playing queued cues in background ...)")
    time.sleep(8)
    engine.stop()
    return True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Manual checklist
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def print_manual_checklist() -> None:
    section("MANUAL CHECKLIST — Phase 7 UI Validation")
    print("""
  Run this on the lab PC after the automated tests pass:

    streamlit run app.py

  ┌──────────────────────────────────────────────────────────────┐
  │  CHECK 1: UI State Persistence                              │
  ├──────────────────────────────────────────────────────────────┤
  │  1. Open the app in Chrome                                  │
  │  2. Start the webcam (click START)                          │
  │  3. Do 3 bicep curl reps in front of the camera             │
  │  4. Watch the FitCoin balance in the top-right header       │
  │  5. Switch to the "Rewards Shop" tab and back               │
  │                                                             │
  │  ✅ PASS: FitCoin balance increments and stays at the       │
  │           new total after switching tabs                     │
  │  ❌ FAIL: Balance resets to 0 → st.session_state is broken  │
  └──────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │  CHECK 2: Audio Non-Blocking (THE Critical Test)            │
  ├──────────────────────────────────────────────────────────────┤
  │  1. Start the webcam feed                                   │
  │  2. Do a rep to trigger a coaching cue + TTS audio          │
  │  3. WHILE the AI voice is speaking, wave your hand          │
  │     in front of the camera                                  │
  │                                                             │
  │  ✅ PASS: Video feed stays smooth and responsive while      │
  │           the voice plays — skeleton tracks your hand        │
  │  ❌ FAIL: Video freezes/stutters while audio plays          │
  │           → pyttsx3 is blocking the main thread             │
  └──────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │  CHECK 3: Rewards Shop                                      │
  ├──────────────────────────────────────────────────────────────┤
  │  1. Switch to the "Rewards Shop" tab                        │
  │  2. Verify product images, names, and FitCoin prices load   │
  │  3. If you have enough FitCoins, click "Buy" on a product   │
  │  4. Check the sidebar for the purchased item                │
  │                                                             │
  │  ✅ PASS: Products display, buy button works, sidebar       │
  │           shows purchase, balance deducted                   │
  │  ❌ FAIL: Empty shop or buy doesn't deduct balance          │
  └──────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │  CHECK 4: Sidebar Pipeline Status                           │
  ├──────────────────────────────────────────────────────────────┤
  │  1. Open the sidebar (click ☰ top-left)                     │
  │  2. Verify green dots next to active pipeline modules       │
  │  3. Yellow dots for modules missing deps (e.g., no          │
  │     GROQ_API_KEY → SR-RAG shows yellow)                     │
  │                                                             │
  │  ✅ PASS: Status reflects actual module availability         │
  └──────────────────────────────────────────────────────────────┘
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 & 7 validation")
    parser.add_argument("--verbose", action="store_true", help="Show full API responses")
    parser.add_argument("--skip-network", action="store_true", help="Skip Open Food Facts API test")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  Neuro-Fit · Phase 6 & 7 Validation")
    print(f"{'=' * 60}")

    results: dict[str, bool] = {}

    # Phase 6
    results["FitCoin Math"] = test_fitcoin_math(args.verbose)

    if not args.skip_network:
        results["Open Food Facts API"] = test_openfoodfacts_api(args.verbose)
    else:
        print(f"\n  (Skipping Open Food Facts API test — --skip-network)")

    # Phase 7
    results["TTS Init"] = test_tts_init(args.verbose)
    results["TTS Non-Blocking"] = test_tts_nonblocking(args.verbose)
    results["TTS Queue"] = test_tts_queue(args.verbose)

    # Summary
    section("SUMMARY")
    all_pass = True
    for name, passed in results.items():
        status = PASS if passed else FAIL
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print(f"\n  ✅ All automated tests passed!")
    else:
        print(f"\n  ❌ Some tests failed — see above for details.")

    print_manual_checklist()

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
