"""
rag_orchestrator.py — Phase 5: Intelligent Coaching Brain (SR-RAG)
===================================================================

Multi-agent LangGraph workflow that fuses real-time MediaPipe telemetry
with pre-parsed SMPL-X kinematic features to produce safe, evidence-
grounded coaching cues at interactive latency.

Architecture (canonical SR-RAG naming)
───────────────────────────────────────
    ┌───────────┐      ┌───────────┐      ┌───────────┐
    │ Proposer  │ ───▶ │  Refuter  │ ───▶ │   Judge   │
    └───────────┘      └───────────┘      └─────┬─────┘
         ▲                                      │
         │         ┌────────────────┐           │
         └─────────│ UNSAFE: redo   │◀──────────┘
                   └────────────────┘      SAFE ──▶ END

    Refuse (impossible observation) ──▶ END  (short-circuit before Proposer)

Proposer  — Drafts a 1-sentence coaching cue from fused telemetry + SMPL-X
            kinematic features.
Refuter   — Retrieves NASM clinical guidelines from FAISS and actively
            searches for safety violations in the proposed cue.
Judge     — Arbitrates with an explicit binary verdict: SAFE or UNSAFE.
            UNSAFE routes back to Proposer (max 2 retries).

LLM : Groq LPU — llama-3.1-8b-instant (ultra-low latency)
RAG : sentence-transformers / FAISS (CPU)

Usage:
    from reasoning import SRRAGOrchestrator
    rag = SRRAGOrchestrator()
    result = rag.run(telemetry_dict, smplx_dict)
    print(result["final_coaching"])
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from utils.constants import CYAN, GREEN, RED, YELLOW
from utils.logging_helpers import agent_log

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pydantic schemas — structured LLM output
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ProposerOutput(BaseModel):
    """Schema returned by the Proposer node."""

    coaching_cue: str = Field(
        description="A single 1-2 sentence coaching correction."
    )
    reasoning: str = Field(
        description="Brief internal reasoning for the chosen cue."
    )
    form_score: float = Field(
        description="Estimated form quality [0.0 - 1.0]."
    )


class JudgeOutput(BaseModel):
    """Schema returned by the Judge node.

    The verdict is a strict binary: SAFE or UNSAFE.
    """

    verdict: Literal["SAFE", "UNSAFE"] = Field(
        description=(
            "SAFE — the proposed cue is clinically sound and can be "
            "delivered to the user.  "
            "UNSAFE — the cue violates a guideline or could cause injury; "
            "it must be regenerated."
        )
    )
    final_coaching_cue: str = Field(
        description=(
            "If SAFE: the final approved coaching string.  "
            "If UNSAFE: empty string."
        )
    )
    explanation: str = Field(
        description="Why the verdict was chosen (audit trail)."
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NASM clinical guidelines corpus
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NASM_GUIDELINES: List[str] = [
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

    "NASM: During squats, the torso should maintain a forward lean between "
    "30-45 degrees.  Excessive forward lean (>60 degrees) shifts load to the "
    "lumbar spine.  Cue 'chest up, drive through heels'.",

    "NASM: Hip hinge movements require posterior chain activation.  If the "
    "spinal_alignment_score drops below 0.70, stop the set and cue the "
    "client to practice the hip hinge pattern with a dowel before adding load.",

    "ACSM: Shoulder symmetry below 0.80 during overhead movements suggests "
    "muscular imbalance or compensatory patterns.  Recommend unilateral "
    "corrective exercises before resuming bilateral lifts.",
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LangGraph state schema — lightweight, no raw vertex arrays
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# SMPL-X data is pre-parsed into scalar kinematic features BEFORE
# entering the graph.  No np.ndarray, no vertex buffers, no mesh data.
# Every field is a Python primitive (str, float, bool, int).
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class CoachingState(TypedDict):
    """Lightweight state flowing through the SR-RAG graph.

    All SMPL-X data is reduced to scalar kinematic features before
    injection — no raw vertices, joints, or mesh buffers ever enter
    this dict.
    """

    # ── Pre-parsed inputs (scalars + short strings only) ──
    telemetry_json: str          # MediaPipe angles / rep data (small JSON)
    spinal_alignment_score: float  # from SMPL-X, pre-extracted [0.0-1.0]
    shoulder_symmetry: float       # from SMPL-X, pre-extracted [0.0-1.0]
    posture_warning: str           # from SMPL-X, e.g. "Mild lumbar flexion"
    risk_level: str                # low / moderate / high / critical
    detected_error: str            # user/sensor reported error string

    # ── Proposer output ──
    proposed_cue: str
    proposed_reasoning: str
    form_score: float

    # ── Refuter output ──
    retrieved_guidelines: str
    refutation: str

    # ── Judge output ──
    judge_verdict: str        # explicit "SAFE" or "UNSAFE"
    final_coaching: str
    judge_explanation: str

    # ── Control flow ──
    is_safe: bool
    input_refusal: bool
    iteration: int            # retry counter (max 2)
    latency_ms: float


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Orchestrator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SRRAGOrchestrator:
    """Phase 5 Intelligent Coaching Brain.

    Three-node SR-RAG LangGraph workflow:
        1. **Proposer**  — drafts a coaching cue from fused kinematic data
        2. **Refuter**   — retrieves NASM guidelines and adversarially
                           searches for safety violations
        3. **Judge**     — binary SAFE / UNSAFE verdict; UNSAFE loops
                           back to Proposer (max 2 retries)

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

        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=api_key,
            temperature=0,
        )

        self.proposer_llm = self.llm.with_structured_output(ProposerOutput)
        self.judge_llm = self.llm.with_structured_output(JudgeOutput)

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
        )
        docs = [Document(page_content=g) for g in NASM_GUIDELINES]
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        telemetry: dict,
        smplx_payload: Optional[dict] = None,
    ) -> dict:
        """Execute the full SR-RAG coaching pipeline.

        SMPL-X data is pre-parsed here into scalar kinematic features
        so the graph state never carries raw vertex arrays.

        Parameters
        ----------
        telemetry : dict
            MediaPipe telemetry (angles, rep count, fatigue flag, etc.)
            or a lab-mode dict with just ``{"detected_error": "..."}``.
        smplx_payload : dict | None
            SMPL-X output from ``SMPLXSpotChecker.process_frame()``.
            Only the scalar kinematic features are extracted:
            ``spinal_alignment_score``, ``posture_warning``,
            ``shoulder_symmetry``.

        Returns
        -------
        dict  (CoachingState)
        """
        t0 = time.perf_counter()

        # ── Pre-parse SMPL-X into scalar features (no mesh in state) ──
        smplx = smplx_payload or {}
        spinal_score = float(smplx.get("spinal_alignment_score", 0.0))
        shoulder_sym = float(smplx.get("shoulder_symmetry", 0.0))
        posture_warn = str(smplx.get("posture_warning", "None"))

        # ── Classify risk from scalar features before entering graph ──
        risk_level = self._classify_risk(telemetry, spinal_score, shoulder_sym)

        # ── Detect impossible observations before entering graph ──
        detected_error = str(telemetry.get("detected_error", ""))
        input_refusal = False
        refusal_text = ""

        if self._is_impossible_observation(detected_error):
            input_refusal = True
            refusal_text = (
                "I cannot provide coaching for this observation: "
                "human knees do not bend backward in the way described. "
                "Please verify the sensor output or describe the movement "
                "in anatomically standard terms."
            )

        initial_state: CoachingState = {
            "telemetry_json": json.dumps(telemetry, indent=2),
            "spinal_alignment_score": spinal_score,
            "shoulder_symmetry": shoulder_sym,
            "posture_warning": posture_warn,
            "risk_level": risk_level,
            "detected_error": detected_error,
            "proposed_cue": "",
            "proposed_reasoning": "",
            "form_score": 0.0,
            "retrieved_guidelines": "",
            "refutation": "",
            "judge_verdict": "",
            "final_coaching": refusal_text,
            "judge_explanation": "",
            "is_safe": input_refusal,
            "input_refusal": input_refusal,
            "iteration": 0,
            "latency_ms": 0.0,
        }

        # Short-circuit: refuse before even entering the graph
        if input_refusal:
            initial_state["judge_verdict"] = "REFUSE"
            initial_state["latency_ms"] = round(
                (time.perf_counter() - t0) * 1000, 1
            )
            agent_log(
                "ROUTER", RED,
                "Observation is anatomically impossible — refusing before "
                "graph execution.",
            )
            return initial_state

        result = self._graph.invoke(initial_state)
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        logger.info(
            "SR-RAG complete in %.1f ms  verdict=%s  form=%.2f",
            result["latency_ms"],
            result.get("judge_verdict"),
            result.get("form_score", 0),
        )
        return result

    # ------------------------------------------------------------------
    # Pre-parse helpers (run BEFORE graph, keep state lightweight)
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_risk(
        telemetry: dict,
        spinal_score: float,
        shoulder_sym: float,
    ) -> str:
        """Deterministic risk classification from scalar features."""
        fatigue = bool(telemetry.get("neuromuscular_fatigue", False))

        if spinal_score > 0 and spinal_score < 0.50:
            return "critical"

        high_risk_count = 0
        if spinal_score > 0 and spinal_score < 0.70:
            high_risk_count += 1
        if shoulder_sym > 0 and shoulder_sym < 0.70:
            high_risk_count += 1
        if high_risk_count >= 2:
            return "critical"
        if high_risk_count >= 1:
            return "high"

        if fatigue:
            return "moderate"
        if spinal_score > 0 and spinal_score < 0.85:
            return "moderate"
        if shoulder_sym > 0 and shoulder_sym < 0.85:
            return "moderate"

        return "low"

    @staticmethod
    def _is_impossible_observation(detected_error: str) -> bool:
        """Deterministic guardrail for anatomically impossible input."""
        de = detected_error.lower()
        return "knee" in de and "backward" in de

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(CoachingState)

        graph.add_node("proposer", self._proposer_node)
        graph.add_node("refuter", self._refuter_node)
        graph.add_node("judge", self._judge_node)

        # START → Proposer → Refuter → Judge
        graph.add_edge(START, "proposer")
        graph.add_edge("proposer", "refuter")
        graph.add_edge("refuter", "judge")

        # Judge → END (SAFE) or Judge → Proposer (UNSAFE, with retry cap)
        graph.add_conditional_edges(
            "judge",
            self._route_after_judge,
            {"SAFE": END, "UNSAFE": "proposer"},
        )

        return graph.compile()

    # ------------------------------------------------------------------
    # Node 1: Proposer
    # ------------------------------------------------------------------

    def _proposer_node(self, state: CoachingState) -> dict:
        """Draft a coaching cue from fused kinematic features + telemetry."""
        iteration = state["iteration"] + 1
        agent_log("PROPOSER", CYAN, f"Drafting coaching cue (attempt {iteration})...")

        prior_rejection = ""
        if state["refutation"]:
            prior_rejection = (
                "\n\nYour previous cue was REJECTED by the Judge:\n"
                f"{state['refutation']}\n"
                "Generate a DIFFERENT, safer cue that addresses this concern."
            )

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an elite AI fitness coach.  Given real-time "
             "biomechanical data from a MediaPipe skeleton and SMPL-X "
             "volumetric mesh, generate ONE concise coaching cue "
             "(1-2 sentences).  Focus on the most critical form "
             "correction RIGHT NOW.\n\n"
             "SMPL-X KINEMATIC FEATURES (pre-extracted):\n"
             "  Spinal alignment: {spinal_score}/1.0\n"
             "  Shoulder symmetry: {shoulder_sym}/1.0\n"
             "  Posture warning: {posture_warning}\n"
             "  Risk level: {risk_level}\n"
             "Also estimate a form_score [0.0-1.0]."
             "{prior_rejection}"),
            ("human",
             "TELEMETRY:\n{telemetry}"),
        ])

        chain = prompt | self.proposer_llm
        result: ProposerOutput = chain.invoke({
            "telemetry": state["telemetry_json"],
            "spinal_score": state["spinal_alignment_score"],
            "shoulder_sym": state["shoulder_symmetry"],
            "posture_warning": state["posture_warning"],
            "risk_level": state["risk_level"],
            "prior_rejection": prior_rejection,
        })

        agent_log(
            "PROPOSER", CYAN,
            f"Cue: \"{result.coaching_cue}\"\n"
            f"Reasoning: {result.reasoning}\n"
            f"Form score: {result.form_score:.2f}",
        )

        return {
            "proposed_cue": result.coaching_cue,
            "proposed_reasoning": result.reasoning,
            "form_score": result.form_score,
            "iteration": iteration,
        }

    # ------------------------------------------------------------------
    # Node 2: Refuter
    # ------------------------------------------------------------------

    def _refuter_node(self, state: CoachingState) -> dict:
        """Adversarial clinical review — retrieve NASM guidelines from
        FAISS and check the proposed cue for safety violations."""
        agent_log("REFUTER", YELLOW, "Retrieving NASM guidelines and auditing cue...")

        query = (
            f"{state['proposed_cue']} "
            f"spinal_alignment={state['spinal_alignment_score']} "
            f"{state['telemetry_json'][:200]}"
        )
        docs = self.retriever.invoke(query)
        guidelines_text = "\n\n".join(
            f"[Guideline {i+1}] {d.page_content}" for i, d in enumerate(docs)
        )

        topic = self._search_topic_label(state["telemetry_json"])
        agent_log(
            "REFUTER", YELLOW,
            f"Retrieving NASM guidelines for {topic}...\n"
            f"Retrieved {len(docs)} guideline chunks:\n"
            + "\n".join(f"  • {d.page_content[:80]}…" for d in docs),
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a clinical safety auditor for a fitness AI.  "
             "Your ONLY job is to find problems.  Compare the proposed "
             "coaching cue against the retrieved clinical guidelines, "
             "the SMPL-X kinematic features, and the raw telemetry.\n\n"
             "If the cue could cause injury, contradicts a guideline, or "
             "misses a critical safety warning present in the data, write "
             "a concise objection (2-3 sentences max).\n\n"
             "If the cue is safe and accurate, respond with exactly: "
             "NO_OBJECTION"),
            ("human",
             "PROPOSED CUE:\n{cue}\n\n"
             "SMPL-X FEATURES:\n"
             "  Spinal alignment: {spinal_score}/1.0\n"
             "  Shoulder symmetry: {shoulder_sym}/1.0\n"
             "  Posture warning: {posture_warning}\n"
             "  Risk level: {risk_level}\n\n"
             "TELEMETRY:\n{telemetry}\n\n"
             "CLINICAL GUIDELINES:\n{guidelines}"),
        ])

        chain = prompt | self.llm
        response = chain.invoke({
            "cue": state["proposed_cue"],
            "spinal_score": state["spinal_alignment_score"],
            "shoulder_sym": state["shoulder_symmetry"],
            "posture_warning": state["posture_warning"],
            "risk_level": state["risk_level"],
            "telemetry": state["telemetry_json"],
            "guidelines": guidelines_text,
        })

        refutation = response.content.strip()
        agent_log("REFUTER", YELLOW, f"Verdict: {refutation}")

        return {
            "refutation": refutation,
            "retrieved_guidelines": guidelines_text,
        }

    # ------------------------------------------------------------------
    # Node 3: Judge
    # ------------------------------------------------------------------

    def _judge_node(self, state: CoachingState) -> dict:
        """Final safety arbiter — explicit binary SAFE / UNSAFE verdict."""
        agent_log("JUDGE", GREEN, "Evaluating proposed cue...")

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are the final safety judge for a fitness AI.\n\n"
             "Review the proposed coaching cue, the Refuter's analysis, "
             "and the biomechanical data below.\n\n"
             "YOUR VERDICT must be exactly one of:\n"
             "  SAFE   — The cue is clinically sound and can be delivered. "
             "Put the approved cue in final_coaching_cue.\n"
             "  UNSAFE — The cue violates guidelines or could cause injury. "
             "Set final_coaching_cue to an empty string.\n\n"
             "For deadlift lumbar rounding, prefer cues aligned with NASM: "
             "chest up, brace core, neutral spine, bar close.\n\n"
             "HALLUCINATION GUARDRAILS: Never approve a cue that treats "
             "fantasy injuries as real.  Never approve a cue that "
             "contradicts the Refuter when the Refuter cites a specific "
             "guideline."),
            ("human",
             "PROPOSED CUE:\n{cue}\n\n"
             "REFUTER ANALYSIS:\n{refutation}\n\n"
             "SMPL-X FEATURES:\n"
             "  Spinal alignment: {spinal_score}/1.0\n"
             "  Shoulder symmetry: {shoulder_sym}/1.0\n"
             "  Posture warning: {posture_warning}\n"
             "  Risk level: {risk_level}\n\n"
             "TELEMETRY:\n{telemetry}"),
        ])

        chain = prompt | self.judge_llm
        result: JudgeOutput = chain.invoke({
            "cue": state["proposed_cue"],
            "refutation": state["refutation"],
            "spinal_score": state["spinal_alignment_score"],
            "shoulder_sym": state["shoulder_symmetry"],
            "posture_warning": state["posture_warning"],
            "risk_level": state["risk_level"],
            "telemetry": state["telemetry_json"],
        })

        verdict = result.verdict  # "SAFE" or "UNSAFE"

        if verdict == "SAFE":
            label, color = "SAFE ✓", GREEN
        else:
            label, color = "UNSAFE ✗", RED

        agent_log(
            "JUDGE", color,
            f"Verdict: {label}\n"
            f"Explanation: {result.explanation}\n"
            f"Final cue: \"{result.final_coaching_cue}\"",
        )

        if verdict == "SAFE":
            return {
                "is_safe": True,
                "judge_verdict": "SAFE",
                "final_coaching": result.final_coaching_cue,
                "judge_explanation": result.explanation,
            }

        return {
            "is_safe": False,
            "judge_verdict": "UNSAFE",
            "final_coaching": "",
            "judge_explanation": result.explanation,
        }

    # ------------------------------------------------------------------
    # Routing: Judge → END or Judge → Proposer
    # ------------------------------------------------------------------

    def _route_after_judge(
        self, state: CoachingState,
    ) -> Literal["SAFE", "UNSAFE"]:
        """Binary routing from the Judge node.

        SAFE  → END  (approved cue delivered)
        UNSAFE → Proposer  (retry, up to MAX_RETRIES)

        If retries are exhausted, emit a safe fallback cue and route
        to END so the graph terminates.
        """
        if state["judge_verdict"] == "SAFE":
            return "SAFE"

        # UNSAFE path — check retry budget
        if state["iteration"] >= self.MAX_RETRIES:
            agent_log(
                "ROUTER", RED,
                f"Max retries ({self.MAX_RETRIES}) exhausted — "
                "emitting safe fallback cue.",
            )
            state["final_coaching"] = (
                "Focus on controlled breathing and maintain a neutral spine.  "
                "If you feel any discomfort, stop and rest."
            )
            state["is_safe"] = True
            state["judge_verdict"] = "SAFE"
            return "SAFE"

        agent_log(
            "ROUTER", YELLOW,
            f"UNSAFE — routing back to Proposer "
            f"(attempt {state['iteration'] + 1}/{self.MAX_RETRIES})",
        )
        return "UNSAFE"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _search_topic_label(telemetry_json: str) -> str:
        """Short human-readable label for FAISS retrieval log."""
        try:
            data = json.loads(telemetry_json)
        except json.JSONDecodeError:
            return "general form safety"
        err = data.get("detected_error")
        if isinstance(err, str) and err.strip():
            slug = re.sub(r"[^\w\s-]", "", err.lower())[:48].strip()
            return slug or "reported error"
        exercise = data.get("exercise", "")
        if exercise:
            return f"{exercise} form analysis"
        return "general form safety"
