"""
Phase 4 lab validation — SR-RAG without video or perception.

Run (from repo root, with GROQ_API_KEY set):

    .venv/bin/python -m reasoning.lab_validation

Tests:
  A) Realistic error (deadlift rounded back) → expect approved clinical cue.
  B) Nonsense error (knees bending backward) → expect Judge REFUSAL
     (hallucination / impossible-observation guardrail).
"""

from __future__ import annotations

import json
import os
import sys

from reasoning.rag_orchestrator import SRRAGOrchestrator, _BOLD, _RESET


def _run_case(name: str, telemetry: dict) -> dict:
    print(f"\n{_BOLD}{'═' * 64}{_RESET}")
    print(f"{_BOLD}CASE: {name}{_RESET}")
    print(f"{_BOLD}Input (telemetry JSON):{_RESET}")
    print(json.dumps(telemetry, indent=2))
    print(f"\n{_BOLD}── LangGraph trace (stderr) ──{_RESET}\n", file=sys.stderr)

    rag = SRRAGOrchestrator()
    result = rag.run(telemetry, smplx_payload={})

    print(f"\n{_BOLD}── Result (stdout) ──{_RESET}")
    out = {
        "judge_verdict": result.get("judge_verdict", ""),
        "input_refusal": result.get("input_refusal", False),
        "final_coaching": result.get("final_coaching", ""),
        "is_safe": result.get("is_safe", False),
        "iterations": result.get("iteration", 0),
    }
    print(json.dumps(out, indent=2))
    return {**out, "_raw": result}


def main() -> None:
    if not os.getenv("GROQ_API_KEY"):
        print("Set GROQ_API_KEY in the environment.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{_BOLD}{'=' * 64}")
    print("  NEURO-FIT  ·  Phase 4 Lab Validation  (SR-RAG, no video)")
    print(f"{'=' * 64}{_RESET}")

    # --- Case A: realistic observation (expect APPROVE + lumbar-safe cue) ---
    case_a = _run_case(
        "A — Deadlift lumbar rounding (realistic)",
        {
            "detected_error": "User's lower back is rounding during a deadlift.",
        },
    )

    # --- Case B: impossible / nonsense (expect REFUSAL) ---
    case_b = _run_case(
        "B — Nonsense observation (guardrail)",
        {"detected_error": "User's knees are bending backward."},
    )

    # --- Mandatory checks (best-effort; LLM may vary wording) ---
    print(f"\n{_BOLD}{'─' * 64}")
    print("  VALIDATION CHECKS")
    print(f"{'─' * 64}{_RESET}")

    ok_a = (
        case_a["judge_verdict"] == "approve_safe_cue"
        and not case_a["input_refusal"]
        and case_a["final_coaching"]
    )
    print(
        f"  [A] Judge approved realistic error & returned coaching: "
        f"{'PASS' if ok_a else 'REVIEW (LLM output may differ)'}",
    )

    ok_b = case_b["judge_verdict"] == "refuse_invalid_observation" and case_b.get(
        "input_refusal",
    )
    print(
        f"  [B] Judge refused nonsense observation: "
        f"{'PASS' if ok_b else 'REVIEW (re-run or tighten prompt)'}",
    )

    if ok_b:
        print(
            "\n  Refusal text (guardrail):\n  "
            + repr(case_b["final_coaching"][:280]),
        )


if __name__ == "__main__":
    main()
