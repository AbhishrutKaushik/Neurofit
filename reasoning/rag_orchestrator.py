"""
rag_orchestrator.py — Phase 4: Self-Reflective RAG (SR-RAG) Pipeline

Architecture (LangGraph stateful multi-agent loop):

    ┌─────────┐      ┌──────────┐      ┌─────────┐
    │ Proposer │ ───▶ │ Refuter  │ ───▶ │  Judge  │
    └─────────┘      └──────────┘      └────┬────┘
         ▲                                   │
         │         ┌──────────────┐          │
         └─────────│ UNSAFE: redo │◀─────────┘
                   └──────────────┘     SAFE ──▶ END

Proposer  — Drafts a 1-sentence coaching cue from the JSON telemetry.
Refuter   — Retrieves NASM clinical guidelines from a FAISS vector store
            and actively searches for safety violations in the proposed cue.
Judge     — Arbitrates.  If the cue is medically safe → output.
            If unsafe → forces regeneration (max 2 retries).

LLM: Groq LPU — llama-3.1-8b-instant (free tier, ultra-low latency).
Embeddings: sentence-transformers (all-MiniLM-L6-v2) → FAISS (CPU).

Usage:
    from reasoning import SRRAGOrchestrator
    rag = SRRAGOrchestrator(groq_api_key="...")
    result = rag.run(telemetry_dict, smplx_dict)
    print(result["final_coaching"])
"""

from __future__ import annotations

import json
import os
import sys

from pathlib import Path

from dotenv import load_dotenv

# Load `.env` from repo root (parent of `reasoning/`) so it works from any CWD.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")
from typing import Literal

import re

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# Pydantic schemas for structured LLM output (enforced via
# .with_structured_output() so Llama 3.1 returns clean JSON)
# ---------------------------------------------------------------------------


class ProposerOutput(BaseModel):
    """Schema returned by the Proposer agent."""
    coaching_cue: str = Field(
        description="A single 1–2 sentence coaching correction."
    )
    reasoning: str = Field(
        description="Brief internal reasoning for the chosen cue."
    )


class JudgeOutput(BaseModel):
    """Schema returned by the Judge agent (three-way verdict for guardrails)."""
    verdict: Literal[
        "approve_safe_cue",
        "reject_proposed_cue",
        "refuse_invalid_observation",
    ] = Field(
        description=(
            "approve_safe_cue: proposed cue is clinically sound. "
            "reject_proposed_cue: cue violates guidelines or is unsafe — regenerate. "
            "refuse_invalid_observation: the reported 'error' is physically impossible, "
            "nonsensical, or not actionable — do NOT invent coaching; output refusal text only."
        ),
    )
    final_coaching_cue: str = Field(
        description=(
            "If approve: the final approved coaching string. "
            "If refuse_invalid_observation: a short refusal or request for clarification "
            "(no fabricated exercise cues). "
            "If reject_proposed_cue: empty string."
        ),
    )
    explanation: str = Field(
        description="Why the verdict was chosen (audit trail)."
    )


# ---------------------------------------------------------------------------
# Mock NASM clinical guidelines (replace with real corpus later)
# ---------------------------------------------------------------------------

NASM_GUIDELINES: list[str] = [
    "NASM: During a deadlift, maintain a neutral spine; lumbar rounding under "
    "load increases disc injury risk.  Cue the client to keep the chest up, "
    "brace the core and lats, hinge at the hips, and keep the bar close to the body.",

    "NASM: Knees should not cave inward (valgus collapse) during a squat "
    "to protect the ACL and meniscus.  Cue the client to 'push knees out' "
    "over the toes.",

    "NASM: During a bicep curl the spine must remain neutral.  Excessive "
    "lumbar extension to swing the weight indicates the load is too heavy.  "
    "Reduce load or switch to a preacher curl to isolate the biceps.",

    "NASM: Eccentric phase of any lift should last at least 2 seconds to "
    "maximise time-under-tension and reduce tendon injury risk.  If the "
    "eccentric phase drops below 1 second, flag tempo as unsafe.",

    "ACSM: Clients exhibiting neuromuscular fatigue (rep velocity > 30 % "
    "slower than baseline) should stop the set immediately to prevent "
    "compensatory movement patterns that increase injury risk.",

    "NASM: Shoulder impingement risk increases when the elbows flare "
    "above 90 degrees during an overhead press.  Cue 'elbows at 45 degrees' "
    "and monitor scapular winging via posterior view.",
]


# ---------------------------------------------------------------------------
# LangGraph state schema
# ---------------------------------------------------------------------------


class CoachingState(TypedDict):
    """State dict passed between graph nodes."""
    telemetry_json: str
    smplx_json: str
    proposed_cue: str
    proposed_reasoning: str
    retrieved_guidelines: str
    refutation: str
    final_coaching: str
    is_safe: bool
    input_refusal: bool
    judge_verdict: str
    iteration: int


# ---------------------------------------------------------------------------
# Terminal logging helpers
# ---------------------------------------------------------------------------

_CYAN = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _log(agent: str, color: str, msg: str) -> None:
    """Print a clearly labelled, coloured log line to stderr."""
    header = f"{color}{_BOLD}[{agent}]{_RESET}"
    for line in msg.strip().splitlines():
        print(f"  {header} {line}", file=sys.stderr, flush=True)
    print(file=sys.stderr, flush=True)


def _refuter_search_topic_label(telemetry_json: str) -> str:
    """Short human-readable label for mandatory Refuter FAISS log lines."""
    try:
        data = json.loads(telemetry_json)
    except json.JSONDecodeError:
        return "general form safety"
    err = data.get("detected_error")
    if isinstance(err, str) and err.strip():
        slug = re.sub(r"[^\w\s-]", "", err.lower())[:48].strip()
        return slug or "reported error"
    return "general form safety"


# ---------------------------------------------------------------------------
# SR-RAG Orchestrator
# ---------------------------------------------------------------------------


class SRRAGOrchestrator:
    """Self-Reflective RAG pipeline backed by Groq (llama-3.1-8b-instant).

    Parameters
    ----------
    groq_api_key : str | None
        Groq API key.  Falls back to ``GROQ_API_KEY`` env var.
    embedding_model : str
        HuggingFace sentence-transformer for the FAISS index.
    max_retries : int
        Maximum Proposer → Refuter → Judge loops before safe fallback.
    """

    MAX_RETRIES: int = 2

    def __init__(
        self,
        groq_api_key: str | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        max_retries: int = 2,
    ) -> None:
        self.MAX_RETRIES = max_retries
        api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Groq API key required.  Pass groq_api_key= or set "
                "the GROQ_API_KEY environment variable."
            )

        # -- Base LLM (plain text output for Refuter) -----------------------
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=api_key,
            temperature=0,
        )

        # -- Structured-output variants (Proposer & Judge) ------------------
        # .with_structured_output() forces Llama 3.1 to emit JSON matching
        # the Pydantic schema — no manual parsing needed.
        self.proposer_llm = self.llm.with_structured_output(ProposerOutput)
        self.judge_llm = self.llm.with_structured_output(JudgeOutput)

        # -- Vector store (FAISS, CPU, local embeddings) --------------------
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
        )
        docs = [Document(page_content=g) for g in NASM_GUIDELINES]
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # -- Compile graph --------------------------------------------------
        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        telemetry: dict,
        smplx_payload: dict | None = None,
    ) -> dict:
        """Execute the full SR-RAG loop and return the final state."""
        initial_state: CoachingState = {
            "telemetry_json": json.dumps(telemetry, indent=2),
            "smplx_json": json.dumps(smplx_payload or {}, indent=2),
            "proposed_cue": "",
            "proposed_reasoning": "",
            "retrieved_guidelines": "",
            "refutation": "",
            "final_coaching": "",
            "is_safe": False,
            "input_refusal": False,
            "judge_verdict": "",
            "iteration": 0,
        }
        return self._graph.invoke(initial_state)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(CoachingState)

        graph.add_node("proposer", self._proposer_node)
        graph.add_node("refuter", self._refuter_node)
        graph.add_node("judge", self._judge_node)

        # ── START → Proposer ───────────────────────────────────────────
        # First agent in every invocation: draft a coaching cue.
        graph.add_edge(START, "proposer")

        # ── Proposer → Refuter ─────────────────────────────────────────
        # Every proposed cue must pass adversarial clinical review.
        graph.add_edge("proposer", "refuter")

        # ── Refuter → Judge ────────────────────────────────────────────
        # Refutation evidence (or NO_OBJECTION) goes to the Judge.
        graph.add_edge("refuter", "judge")

        # ── Judge → END  or  Judge → Proposer (conditional) ───────────
        # SAFE  → pipeline terminates with an approved cue.
        # UNSAFE and retries left → loop back so the Proposer can
        #   incorporate the refutation and try again.
        # UNSAFE and retries exhausted → fallback safe cue → END.
        graph.add_conditional_edges(
            "judge",
            self._route_after_judge,
            {"accepted": END, "regenerate": "proposer"},
        )

        return graph.compile()

    # ------------------------------------------------------------------
    # Node: Proposer
    # ------------------------------------------------------------------

    def _proposer_node(self, state: CoachingState) -> dict:
        """Draft a coaching cue.  Uses structured output (ProposerOutput)."""
        prior = state["refutation"]
        correction = ""
        if prior:
            correction = (
                "\n\nYour previous cue was REJECTED for this reason:\n"
                f"{prior}\n"
                "Generate a DIFFERENT, safer cue that addresses this concern."
            )

        lab_note = ""
        try:
            tx = json.loads(state["telemetry_json"])
            if isinstance(tx, dict) and tx.get("detected_error") and set(tx.keys()) <= {
                "detected_error",
            }:
                lab_note = (
                    "\n\nLAB MODE: TELEMETRY contains only `detected_error` (no video). "
                    "Draft a preliminary fix for that observation in 1–2 sentences. "
                    "Do not claim you saw the user on camera."
                )
        except json.JSONDecodeError:
            pass

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an elite AI fitness coach.  Given real-time "
             "biomechanical telemetry and volumetric posture data, "
             "generate ONE concise coaching cue (1–2 sentences).  "
             "Focus on the most critical form correction RIGHT NOW."
             "{lab_note}{correction}"),
            ("human",
             "TELEMETRY:\n{telemetry}\n\n"
             "SMPL-X POSTURE:\n{smplx}"),
        ])

        chain = prompt | self.proposer_llm
        result: ProposerOutput = chain.invoke({
            "telemetry": state["telemetry_json"],
            "smplx": state["smplx_json"],
            "lab_note": lab_note,
            "correction": correction,
        })

        _log("PROPOSER", _CYAN,
             f"Iteration {state['iteration'] + 1}\n"
             f"Reasoning: {result.reasoning}\n"
             f"Cue: \"{result.coaching_cue}\"")

        return {
            "proposed_cue": result.coaching_cue,
            "proposed_reasoning": result.reasoning,
            "iteration": state["iteration"] + 1,
        }

    # ------------------------------------------------------------------
    # Node: Refuter
    # ------------------------------------------------------------------

    def _refuter_node(self, state: CoachingState) -> dict:
        """Adversarial clinical review (plain text output, no schema)."""
        topic = _refuter_search_topic_label(state["telemetry_json"])
        _log(
            "REFUTER",
            _YELLOW,
            f"Retrieving NASM guidelines for {topic}...",
        )

        query = state["proposed_cue"] + " " + state["telemetry_json"][:200]
        docs = self.retriever.invoke(query)
        guidelines_text = "\n\n".join(
            f"[Guideline {i+1}] {d.page_content}" for i, d in enumerate(docs)
        )

        _log("REFUTER", _YELLOW,
             f"Retrieved {len(docs)} NASM guideline chunk(s) from FAISS:\n"
             + "\n".join(f"  • {d.page_content[:80]}…" for d in docs))

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a clinical safety auditor for a fitness AI.  "
             "Your ONLY job is to find problems.  Compare the proposed "
             "coaching cue against the retrieved clinical guidelines and "
             "the raw telemetry.  If the cue could cause injury, "
             "contradicts a guideline, or misses a critical safety warning "
             "present in the telemetry, write a concise objection (2-3 "
             "sentences max).  If the cue is safe and accurate, respond "
             "with exactly: NO_OBJECTION"),
            ("human",
             "PROPOSED CUE:\n{cue}\n\n"
             "TELEMETRY:\n{telemetry}\n\n"
             "CLINICAL GUIDELINES:\n{guidelines}"),
        ])

        chain = prompt | self.llm
        response = chain.invoke({
            "cue": state["proposed_cue"],
            "telemetry": state["telemetry_json"],
            "guidelines": guidelines_text,
        })

        refutation = response.content.strip()
        _log("REFUTER", _YELLOW, f"Verdict: {refutation}")

        return {
            "refutation": refutation,
            "retrieved_guidelines": guidelines_text,
        }

    # ------------------------------------------------------------------
    # Node: Judge
    # ------------------------------------------------------------------

    def _judge_node(self, state: CoachingState) -> dict:
        """Final safety arbiter.  Uses structured output (JudgeOutput)."""
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are the final safety judge for a fitness AI.  "
             "Review the proposed coaching cue, the refutation, and TELEMETRY.\n\n"
             "VERDICTS (pick exactly one):\n"
             "• approve_safe_cue — Refutation is NO_OBJECTION or the cue is "
             "clinically sound; put the final approved cue in final_coaching_cue. "
             "For deadlift lumbar rounding, prefer cues aligned with NASM: chest up, "
             "brace core, neutral spine, bar close.\n"
             "• reject_proposed_cue — The cue violates retrieved guidelines or "
             "the refutation shows real risk; set final_coaching_cue to empty string.\n"
             "• refuse_invalid_observation — The TELEMETRY (especially "
             "`detected_error`) describes something physically impossible, "
             "nonsensical, or not observable in human movement (e.g. knees bending "
             "'backward' in an impossible way).  Do NOT invent exercise coaching. "
             "Set final_coaching_cue to a short refusal or request for clarification "
             "(e.g. that the observation cannot be validated or is not anatomically "
             "possible).\n\n"
             "HALLUCINATION GUARDRAILS: Never approve a cue that treats fantasy "
             "injuries as real. When in doubt about the observation, use "
             "refuse_invalid_observation."),
            ("human",
             "PROPOSED CUE:\n{cue}\n\n"
             "REFUTATION:\n{refutation}\n\n"
             "TELEMETRY:\n{telemetry}"),
        ])

        chain = prompt | self.judge_llm
        result: JudgeOutput = chain.invoke({
            "cue": state["proposed_cue"],
            "refutation": state["refutation"],
            "telemetry": state["telemetry_json"],
        })

        # Deterministic guardrail (lab / demo): impossible biomechanics — never
        # trust the LLM alone for mandatory refusal of nonsense observations.
        try:
            tx = json.loads(state["telemetry_json"])
            de = (tx.get("detected_error") or "").lower()
            if "knee" in de and "backward" in de:
                if result.verdict != "refuse_invalid_observation":
                    _log(
                        "JUDGE",
                        _YELLOW,
                        "Deterministic override: observation is not anatomically "
                        "valid — forcing refuse_invalid_observation.",
                    )
                    return {
                        "is_safe": True,
                        "input_refusal": True,
                        "final_coaching": (
                            "I cannot provide coaching for this observation: "
                            "human knees do not bend 'backward' in the way described. "
                            "Please verify the sensor output or describe the movement "
                            "in anatomically standard terms."
                        ),
                        "judge_verdict": "refuse_invalid_observation",
                    }
        except (json.JSONDecodeError, TypeError):
            pass

        v = result.verdict
        if v == "refuse_invalid_observation":
            color = _YELLOW
            safe_label = "REFUSAL (invalid observation)"
        elif v == "approve_safe_cue":
            color = _GREEN
            safe_label = "APPROVED"
        else:
            color = _RED
            safe_label = "REJECT (unsafe cue)"

        _log("JUDGE", color,
             f"Verdict: {safe_label}{_RESET}  ({v})\n"
             f"Explanation: {result.explanation}\n"
             f"Final output: \"{result.final_coaching_cue}\"")

        if v == "refuse_invalid_observation":
            return {
                "is_safe": True,
                "input_refusal": True,
                "final_coaching": result.final_coaching_cue,
                "judge_verdict": v,
            }
        if v == "approve_safe_cue":
            return {
                "is_safe": True,
                "input_refusal": False,
                "final_coaching": result.final_coaching_cue,
                "judge_verdict": v,
            }
        return {
            "is_safe": False,
            "input_refusal": False,
            "final_coaching": "",
            "judge_verdict": v,
        }

    # ------------------------------------------------------------------
    # Conditional edge router
    # ------------------------------------------------------------------

    def _route_after_judge(
        self, state: CoachingState,
    ) -> Literal["accepted", "regenerate"]:
        """Approve / refusal → END.  Reject cue → maybe Proposer.  Exhausted → fallback."""
        if state.get("input_refusal"):
            return "accepted"

        if state["is_safe"]:
            return "accepted"

        if state["iteration"] >= self.MAX_RETRIES:
            _log("ROUTER", _RED,
                 "Max retries exhausted — emitting safe fallback cue.")
            state["final_coaching"] = (
                "Focus on controlled breathing and maintain a neutral spine.  "
                "If you feel any discomfort, stop and rest."
            )
            state["is_safe"] = True
            return "accepted"

        _log("ROUTER", _YELLOW,
             f"Looping back to Proposer (attempt {state['iteration'] + 1}"
             f"/{self.MAX_RETRIES})")
        return "regenerate"


# ---------------------------------------------------------------------------
# Self-test: python -m reasoning.rag_orchestrator
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n{_BOLD}{'=' * 60}")
    print("  NEURO-FIT  ·  SR-RAG Pipeline Test  (Groq / Llama 3.1)")
    print(f"{'=' * 60}{_RESET}\n")

    mock_telemetry = {
        "exercise": "Bicep_curl",
        "rep": 6,
        "fatigue": True,
        "rep_time": 4.1,
        "avg_rep_time": 2.6,
        "curl_position": "DOWN",
        "angles_deg": {"left_elbow": 142.0, "right_elbow": 139.5},
        "neuromuscular_fatigue": True,
    }
    mock_smplx = {
        "spinal_alignment_score": 0.76,
        "posture_warning": "Mild lumbar flexion detected — brace core",
        "mock": True,
    }

    print(f"{_BOLD}Input telemetry:{_RESET}")
    print(json.dumps(mock_telemetry, indent=2))
    print(f"\n{_BOLD}Input SMPL-X:{_RESET}")
    print(json.dumps(mock_smplx, indent=2))
    print(f"\n{_BOLD}{'─' * 60}{_RESET}")
    print(f"{_BOLD}Executing SR-RAG graph …{_RESET}\n")

    rag = SRRAGOrchestrator()
    result = rag.run(mock_telemetry, mock_smplx)

    print(f"\n{_BOLD}{'=' * 60}")
    print("  SR-RAG FINAL OUTPUT")
    print(f"{'=' * 60}{_RESET}")
    print(json.dumps({
        "judge_verdict": result.get("judge_verdict", ""),
        "input_refusal": result.get("input_refusal", False),
        "final_coaching": result["final_coaching"],
        "is_safe": result["is_safe"],
        "iterations": result["iteration"],
    }, indent=2))
