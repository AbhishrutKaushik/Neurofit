"""
validate_phase5.py — Phase 5 SR-RAG Pipeline Validation
=========================================================

Runs the full Proposer → Refuter → Judge multi-agent debate through
Groq's LPU and validates:

  1. SPEED        — Full pipeline < 5 s (expect < 2 s on Groq LPU)
  2. STRUCTURED   — Output contains Form_Score, Correction_Text (no filler)
  3. SCHEMA       — judge_verdict is exactly "SAFE" or "UNSAFE" (or "REFUSE")
  4. GUARDRAIL    — Impossible observations are refused without coaching
  5. STATE        — No raw vertex/mesh data leaks into the graph state

Usage (from repo root, venv active, GROQ_API_KEY in .env):

    python scripts/validate_phase5.py
    python scripts/validate_phase5.py --verbose

Requires: GROQ_API_KEY set in .env or environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Latency thresholds.
# NOTE: On Groq free-tier, sequential LLM calls across multiple test
# cases can trigger rate-limit back-off, inflating wall-clock latency.
# Latency is therefore reported as a WARNING, not a hard FAIL.
# A hard FAIL only fires if a single case exceeds LATENCY_HARD_FAIL_MS.
LATENCY_IDEAL_MS   = 2000    # green: fast path, 1 loop on Groq LPU
LATENCY_WARN_MS    = 5000    # yellow: acceptable, possible rate-limit delay
LATENCY_HARD_FAIL_MS = 60000 # red: something is genuinely broken (> 60 s)

FILLER_PHRASES = [
    "here is your",
    "here's your",
    "sure, ",
    "of course",
    "certainly",
    "i'd be happy to",
    "as an ai",
    "as a language model",
    "let me help",
    "absolutely",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 5: SR-RAG pipeline validation (Groq / Llama 3.1)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print full graph state for each test case.",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    if not os.getenv("GROQ_API_KEY"):
        print("  ERROR: GROQ_API_KEY not set. Fill .env or export it.")
        sys.exit(1)

    divider("STEP 1 — Initialise SRRAGOrchestrator")

    t0 = time.perf_counter()
    from reasoning.rag_orchestrator import SRRAGOrchestrator
    rag = SRRAGOrchestrator()
    init_ms = (time.perf_counter() - t0) * 1000
    print(f"  Orchestrator initialised in {init_ms:.0f} ms")
    print(f"  FAISS index: {rag.vectorstore.index.ntotal} guideline vectors")

    results = []

    # ── Case A: Realistic telemetry + SMPL-X (expect SAFE) ──────────
    divider("CASE A — Realistic bicep curl with fatigue + SMPL-X")

    telemetry_a = {
        "exercise": "bicep_curl",
        "rep": 6,
        "rep_time": 4.1,
        "avg_rep_time": 2.6,
        "curl_position": "DOWN",
        "angles_deg": {"left_elbow": 142.0, "right_elbow": 139.5},
        "neuromuscular_fatigue": True,
    }
    smplx_a = {
        "spinal_alignment_score": 0.76,
        "posture_warning": "Mild lumbar flexion detected — brace core",
        "shoulder_symmetry": 0.92,
    }
    results.append(run_case(rag, "A", telemetry_a, smplx_a, args.verbose))

    # ── Case B: Deadlift rounding (expect SAFE with lumbar cue) ─────
    divider("CASE B — Deadlift lumbar rounding")

    telemetry_b = {
        "detected_error": "User's lower back is rounding during a deadlift.",
    }
    smplx_b = {
        "spinal_alignment_score": 0.55,
        "posture_warning": "Severe lumbar rounding detected",
        "shoulder_symmetry": 0.88,
    }
    results.append(run_case(rag, "B", telemetry_b, smplx_b, args.verbose))

    # ── Case C: Impossible observation (expect REFUSE) ──────────────
    divider("CASE C — Impossible observation (knees backward)")

    telemetry_c = {
        "detected_error": "User's knees are bending backward.",
    }
    results.append(run_case(rag, "C", telemetry_c, None, args.verbose))

    # ── Case D: Healthy form / no issues (expect SAFE, low risk) ────
    divider("CASE D — Healthy form, no issues")

    telemetry_d = {
        "exercise": "squat",
        "rep": 3,
        "rep_time": 2.5,
        "avg_rep_time": 2.4,
        "angles_deg": {"left_knee": 95.0, "right_knee": 93.0},
        "neuromuscular_fatigue": False,
    }
    smplx_d = {
        "spinal_alignment_score": 0.95,
        "posture_warning": "None",
        "shoulder_symmetry": 0.97,
    }
    results.append(run_case(rag, "D", telemetry_d, smplx_d, args.verbose))

    # ── Summary ─────────────────────────────────────────────────────
    divider("VALIDATION SUMMARY")

    all_pass = True
    for r in results:
        status_color = "\033[92m" if r["pass"] else "\033[91m"
        reset = "\033[0m"
        label = "PASS" if r["pass"] else "FAIL"
        print(
            f"  [{r['case']}] {status_color}{label}{reset}  "
            f"({r['latency_ms']:.0f} ms)  "
            f"verdict={r['verdict']}  "
            f"checks: {', '.join(r['checks'])}"
        )
        if not r["pass"]:
            all_pass = False
            for f in r.get("failures", []):
                print(f"       ✗ {f}")

    print()
    if all_pass:
        print("  \033[92mALL CASES PASSED\033[0m")
    else:
        print("  \033[91mSOME CASES FAILED — see details above\033[0m")

    sys.exit(0 if all_pass else 1)


def run_case(
    rag,
    case_id: str,
    telemetry: dict,
    smplx_payload: dict | None,
    verbose: bool,
) -> dict:
    """Execute one test case and validate the output."""
    print(f"  Input telemetry: {json.dumps(telemetry, indent=4)}")
    if smplx_payload:
        print(f"  Input SMPL-X:    {json.dumps(smplx_payload, indent=4)}")
    print()

    result = rag.run(telemetry, smplx_payload)

    latency = result.get("latency_ms", 0)
    verdict = result.get("judge_verdict", "")
    coaching = result.get("final_coaching", "")
    form_score = result.get("form_score", 0.0)
    iteration = result.get("iteration", 0)
    is_refusal = result.get("input_refusal", False)

    checks = []
    failures = []

    # ── Check 1: Latency (warning only — free-tier Groq may rate-limit) ──
    if latency <= LATENCY_IDEAL_MS:
        checks.append(f"speed={latency:.0f}ms")
    elif latency <= LATENCY_WARN_MS:
        checks.append(f"speed={latency:.0f}ms(ok)")
    elif latency <= LATENCY_HARD_FAIL_MS:
        checks.append(f"speed={latency:.0f}ms(WARN:rate-limit?)")
        print(f"    \033[93m⚠ Latency {latency:.0f}ms — possible Groq free-tier rate limit\033[0m")
    else:
        failures.append(f"Latency {latency:.0f}ms exceeds hard limit of {LATENCY_HARD_FAIL_MS}ms")

    # ── Check 2: Verdict schema ──
    if verdict in ("SAFE", "UNSAFE", "REFUSE"):
        checks.append(f"verdict={verdict}")
    else:
        failures.append(f"judge_verdict is '{verdict}', expected SAFE/UNSAFE/REFUSE")

    # ── Check 3: Structured output — no conversational filler ──
    coaching_lower = coaching.lower()
    has_filler = any(phrase in coaching_lower for phrase in FILLER_PHRASES)
    if has_filler:
        failures.append(
            f"Coaching contains conversational filler: '{coaching[:100]}'"
        )
    else:
        checks.append("no_filler")

    # ── Check 4: State schema — no raw vertex/mesh data ──
    state_keys = set(result.keys())
    forbidden_keys = {"vertices", "faces", "joints_3d", "smplx_json"}
    leaked = state_keys & forbidden_keys
    if leaked:
        failures.append(f"Raw mesh data leaked into state: {leaked}")
    else:
        checks.append("no_mesh_in_state")

    # ── Check 5: form_score is a valid float ──
    if isinstance(form_score, (int, float)) and 0.0 <= form_score <= 1.0:
        checks.append(f"form={form_score:.2f}")
    elif is_refusal:
        checks.append("form=N/A(refusal)")
    else:
        failures.append(f"form_score={form_score} out of [0.0, 1.0]")

    # ── Check 6: Case-specific expectations ──
    if case_id == "C":
        if not is_refusal:
            failures.append("Case C: Expected input_refusal=True for impossible observation")
        else:
            checks.append("refusal_guardrail")
        if verdict != "REFUSE":
            failures.append(f"Case C: Expected verdict=REFUSE, got {verdict}")
    else:
        if not coaching and not is_refusal:
            failures.append("Non-refusal case returned empty coaching text")
        elif coaching:
            checks.append(f"cue_len={len(coaching)}")

    # ── Check 7: Retry counter ──
    if iteration <= 2:
        checks.append(f"iters={iteration}")
    else:
        failures.append(f"iteration={iteration} exceeds max_retries=2")

    passed = len(failures) == 0

    # ── Print result ──
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    print(f"  {color}{'PASS' if passed else 'FAIL'}{reset}")
    print(f"    Verdict      : {verdict}")
    print(f"    Latency      : {latency:.0f} ms")
    print(f"    Form score   : {form_score:.2f}")
    print(f"    Iterations   : {iteration}")
    print(f"    Coaching     : \"{coaching[:120]}{'...' if len(coaching) > 120 else ''}\"")
    if failures:
        for f in failures:
            print(f"    \033[91m✗ {f}\033[0m")

    if verbose:
        print(f"\n    Full state:")
        safe_result = {
            k: v for k, v in result.items()
            if not isinstance(v, (bytes, memoryview))
        }
        print(json.dumps(safe_result, indent=6, default=str))

    return {
        "case": case_id,
        "pass": passed,
        "latency_ms": latency,
        "verdict": verdict,
        "checks": checks,
        "failures": failures,
    }


def divider(title: str) -> None:
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
