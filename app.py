"""
app.py — Neuro-Fit: Real-Time AI Fitness Trainer
==================================================

Phase 7 — Full UI Assembly with:
  • Tab 1 "Workout Zone": WebRTC or IP Webcam live video, skeleton overlay,
    telemetry dashboard, live coaching cues, FitCoin earnings per set.
  • Tab 2 "Rewards Shop": Open Food Facts product catalog, FitCoin wallet.
  • Non-blocking TTS via background thread (pyttsx3).
  • SR-RAG coaching pipeline (mock-safe: skipped if GROQ_API_KEY absent).
  • SMPL-X spot-check (mock mode unless CUDA + model files available).

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import logging
import queue
import socket
import threading
import time
from pathlib import Path
from typing import Optional

import av
import cv2
import numpy as np
from dotenv import load_dotenv
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

load_dotenv(Path(__file__).resolve().parent / ".env")

from vision_engine import PoseTracker
from fitcoin_economy import calculate_rewards, fetch_store_rewards
from utils.audio import TTSEngine

logger = logging.getLogger(__name__)

_MAX_INFERENCE_WIDTH = 720   # higher res → better classifier accuracy
_IP_TARGET_WIDTH = 640       # phone stream capped at 640px wide

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Optional heavy imports — graceful degradation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_RAG_AVAILABLE = False
try:
    import os
    if os.getenv("GROQ_API_KEY"):
        from reasoning import SRRAGOrchestrator
        _RAG_AVAILABLE = True
except Exception:
    pass

_SMPLX_AVAILABLE = False
try:
    from perception import SMPLXSpotChecker
    _SMPLX_AVAILABLE = True
except Exception:
    pass

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Page config & global CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(
    page_title="Neuro-Fit · AI Fitness Trainer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2.25rem !important;
        padding-bottom: 1rem;
    }
    [data-testid="stMetric"] {
        background: #0e1117;
        border: 1px solid #262730;
        border-radius: 0.6rem;
        padding: 0.6rem 0.8rem;
    }
    .neurofit-wallet-row {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        min-height: 3.25rem;
        padding-top: 0.35rem;
        padding-bottom: 0.35rem;
        overflow: visible;
    }
    .fitcoin-badge {
        display: inline-block;
        background: linear-gradient(135deg, #f5af19, #f12711);
        color: white;
        padding: 0.45rem 1rem;
        border-radius: 1rem;
        font-weight: 700;
        font-size: 1.05rem;
        line-height: 1.35;
        white-space: nowrap;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
    }
    .product-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 0.8rem;
        padding: 1rem;
        text-align: center;
        height: 100%;
    }
    .product-card img {
        max-height: 120px;
        object-fit: contain;
        margin-bottom: 0.5rem;
    }
    @keyframes fatigue-border-flash {
        0%, 100% {
            box-shadow: 0 0 20px 5px rgba(255,0,0,0.7),
                        inset 0 0 30px rgba(255,0,0,0.08);
            border-color: #ff0000;
        }
        50% {
            box-shadow: 0 0 40px 12px rgba(255,0,0,0.35),
                        inset 0 0 10px rgba(255,0,0,0.03);
            border-color: #cc0000;
        }
    }
    .fatigue-screen-flash {
        border: 4px solid #ff0000 !important;
        border-radius: 0.7rem;
        padding: 1rem;
        animation: fatigue-border-flash 0.6s ease-in-out infinite;
        margin-bottom: 0.75rem;
    }
    .fatigue-banner {
        background: linear-gradient(135deg, #7f1d1d, #b91c1c, #dc2626);
        color: #fff;
        padding: 1rem 1.5rem;
        border-radius: 0.6rem;
        font-weight: 800;
        font-size: 1.15rem;
        text-align: center;
        animation: fatigue-border-flash 0.6s ease-in-out infinite;
        margin-bottom: 0.75rem;
        letter-spacing: 0.02em;
    }
    .exercise-header {
        background: linear-gradient(135deg, #1e3a5f, #0ea5e9);
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 0.6rem;
        font-weight: 700;
        font-size: 1.35rem;
        text-align: center;
        margin-bottom: 0.75rem;
        letter-spacing: 0.03em;
    }
    .exercise-header small {
        font-weight: 400;
        font-size: 0.75rem;
        opacity: 0.8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Session state initialisation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if "tracker" not in st.session_state:
    st.session_state.tracker = PoseTracker()

if "tts" not in st.session_state:
    st.session_state.tts = TTSEngine()

if "fitcoin_balance" not in st.session_state:
    st.session_state.fitcoin_balance = 0.0

if "purchased_items" not in st.session_state:
    st.session_state.purchased_items = []

if "last_coaching" not in st.session_state:
    st.session_state.last_coaching = ""

if "last_form_score" not in st.session_state:
    st.session_state.last_form_score = 0.0

if "coaching_latency_ms" not in st.session_state:
    st.session_state.coaching_latency_ms = 0.0

if "rag_orchestrator" not in st.session_state:
    if _RAG_AVAILABLE:
        try:
            st.session_state.rag_orchestrator = SRRAGOrchestrator()
        except Exception as exc:
            logger.warning("SR-RAG init failed: %s", exc)
            st.session_state.rag_orchestrator = None
    else:
        st.session_state.rag_orchestrator = None

if "smplx_checker" not in st.session_state:
    if _SMPLX_AVAILABLE:
        st.session_state.smplx_checker = SMPLXSpotChecker(MOCK_MODE=True)
    else:
        st.session_state.smplx_checker = None

if "store_catalog" not in st.session_state:
    st.session_state.store_catalog = None

if "prev_rep_count" not in st.session_state:
    st.session_state.prev_rep_count = 0

if "target_muscle_group" not in st.session_state:
    st.session_state.target_muscle_group = "Arms"

if "fatigue_toast_shown" not in st.session_state:
    st.session_state.fatigue_toast_shown = False

if "is_workout_active" not in st.session_state:
    st.session_state.is_workout_active = False

if "target_exercise" not in st.session_state:
    st.session_state.target_exercise = ""

if "fatigue_stopped" not in st.session_state:
    st.session_state.fatigue_stopped = False

tracker: PoseTracker = st.session_state.tracker
tts: TTSEngine = st.session_state.tts

_rag = st.session_state.get("rag_orchestrator")
_smplx = st.session_state.get("smplx_checker")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Thread-safe video state (shared between callback threads & main thread)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _VideoState:
    """Thread-safe mutable state shared between camera threads and
    the Streamlit main thread.  Plain Python — no st.session_state."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frame_i: int = 0
        self.last_bgr: Optional[np.ndarray] = None
        self.prev_rep_count: int = 0
        self.last_form_score: float = 0.0
        self.last_coaching: str = ""
        self.coaching_latency_ms: float = 0.0
        self.coins_earned: float = 0.0
        self.session_gen: int = 0
        self.detected_exercise: str = "unknown"
        self.exercise_confidence: float = 0.0
        self.fatigue_active: bool = False
        self.is_workout_active: bool = False
        self.locked_exercise: str = ""
        self.fatigue_stopped: bool = False
        self.allowed_labels: set = set()
        # IP webcam: background thread writes JPEG bytes; fragment just reads
        self._ip_jpg_bytes: Optional[bytes] = None
        # WebRTC zero-copy pipeline
        self._raw_webrtc_frame: Optional[np.ndarray] = None
        self._display_webrtc_frame: Optional[np.ndarray] = None
        self._webrtc_frame_seq: int = 0


if "_vstate" not in st.session_state:
    st.session_state._vstate = _VideoState()
_vs: _VideoState = st.session_state._vstate
_vs.is_workout_active = st.session_state.get("is_workout_active", False)

if _vs.fatigue_stopped and st.session_state.get("is_workout_active", False):
    st.session_state.is_workout_active = False
    st.session_state.fatigue_stopped = True
    _vs.is_workout_active = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Terminal logging — printed after every rep so API calls are visible
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_C = {
    "cyan":   "\033[96m",
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
}


def _log_rep_to_terminal(
    rep: int,
    telemetry: dict,
    smplx: dict | None,
    result: dict,
) -> None:
    """Print a structured summary of the full Groq SR-RAG exchange to stdout.
    Each block is clearly labelled so it can be screenshotted for a demo.
    """
    b, r = _C["bold"], _C["reset"]
    div = "─" * 64

    print(f"\n{_C['cyan']}{b}{'═'*64}{r}")
    print(f"{_C['cyan']}{b}  REP {rep} — Groq SR-RAG Pipeline Trace{r}")
    print(f"{_C['cyan']}{b}{'═'*64}{r}")

    # ── Input sent to Groq ──────────────────────────────────────────
    print(f"\n{b}[INPUT] MediaPipe Telemetry → Groq:{r}")
    print(json.dumps(telemetry, indent=2))

    if smplx:
        print(f"\n{b}[INPUT] SMPL-X Kinematic Features → Groq:{r}")
        print(json.dumps(smplx, indent=2))

    # ── Proposer ─────────────────────────────────────────────────────
    print(f"\n{_C['yellow']}{b}[PROPOSER] Draft coaching cue:{r}")
    print(f"  Cue       : {result.get('proposed_cue', '—')}")
    print(f"  Reasoning : {result.get('proposed_reasoning', '—')}")
    print(f"  Form score: {result.get('form_score', 0.0):.2f}")

    # ── Refuter ──────────────────────────────────────────────────────
    refutation = result.get("refutation", "—")
    refute_color = _C["red"] if refutation.startswith("OBJECTION") else _C["green"]
    print(f"\n{refute_color}{b}[REFUTER] Adversarial audit:{r}")
    print(f"  {refutation}")

    guidelines = result.get("retrieved_guidelines", "")
    if guidelines:
        print(f"\n{b}  NASM guidelines retrieved:{r}")
        for line in guidelines.strip().split("\n")[:3]:
            print(f"    {line}")

    # ── Judge ────────────────────────────────────────────────────────
    verdict = result.get("judge_verdict", "—")
    verdict_color = _C["green"] if verdict == "SAFE" else _C["red"]
    print(f"\n{verdict_color}{b}[JUDGE] Verdict: {verdict}{r}")
    print(f"  {result.get('judge_explanation', '—')}")

    # ── Final output spoken by TTS ───────────────────────────────────
    print(f"\n{_C['green']}{b}[FINAL COACHING CUE → TTS]:{r}")
    print(f"  \"{result.get('final_coaching', '—')}\"")

    print(f"\n{b}  Iterations : {result.get('iteration', '—')}{r}")
    print(f"{b}  Latency    : {result.get('latency_ms', 0.0):.0f} ms (Groq LPU){r}")
    print(f"{_C['cyan']}{b}{'─'*64}{r}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Background coaching worker — queue-based, NEVER blocks the video pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_rep_q: queue.Queue = queue.Queue(maxsize=2)
_rag_q: queue.Queue = queue.Queue(maxsize=1)

_RAG_EVERY_N_REPS = 3

_SYMMETRY_THRESHOLD = 15.0
_ECCENTRIC_RATIO_MIN = 0.6


def _instant_form_cue(rep_count: int, telemetry: dict) -> str:
    """Analyse joint angles and timing to produce a spoken coaching cue
    in ~0 ms — no API call, pure local math on the telemetry dict."""

    fatigue = telemetry.get("neuromuscular_fatigue", False)
    if fatigue:
        return f"Rep {rep_count}. Fatigue detected. Stop and rest."

    angles = telemetry.get("angles_deg", {})
    concentric = telemetry.get("concentric_sec", 0.0)
    eccentric = telemetry.get("eccentric_sec", 0.0)

    le = angles.get("left_elbow", 0.0)
    re = angles.get("right_elbow", 0.0)
    lk = angles.get("left_knee", 0.0)
    rk = angles.get("right_knee", 0.0)
    ls = angles.get("left_shoulder", 0.0)
    rs = angles.get("right_shoulder", 0.0)
    lh = angles.get("left_hip", 0.0)
    rh = angles.get("right_hip", 0.0)

    tips: list[str] = []

    if abs(le - re) > _SYMMETRY_THRESHOLD:
        weak = "left" if le > re else "right"
        tips.append(f"Keep your {weak} elbow closer to your body")

    if abs(lk - rk) > _SYMMETRY_THRESHOLD:
        tips.append("Keep both knees aligned")

    if abs(ls - rs) > _SYMMETRY_THRESHOLD:
        tips.append("Level your shoulders")

    if abs(lh - rh) > _SYMMETRY_THRESHOLD:
        tips.append("Keep your hips square")

    if concentric > 0.3 and eccentric > 0.1:
        ratio = eccentric / concentric
        if ratio < _ECCENTRIC_RATIO_MIN:
            tips.append("Slow down the lowering phase for better control")
        elif concentric > 3.0:
            tips.append("Try to be more explosive on the way up")

    if not tips:
        positives = [
            "Good form, maintain that range of motion",
            "Solid rep, keep your core tight",
            "Nice tempo, stay controlled",
            "Great rep, keep it up",
        ]
        tips.append(positives[rep_count % len(positives)])

    cue = f"Rep {rep_count}. {tips[0]}."
    return cue


def _quick_cue_worker() -> None:
    """**Fast thread** — speaks an instant form-based coaching cue on every
    rep by analysing joint angles and timing locally (~0 ms, no API call).
    Also awards FitCoins and forwards the rep to the RAG thread."""
    while True:
        rep_count, telemetry, frame = _rep_q.get()
        gen_at_start = _vs.session_gen

        if _vs.session_gen != gen_at_start:
            continue

        # ── Instant form-based spoken cue (no API call) ──────────────
        tts.flush()
        cue = _instant_form_cue(rep_count, telemetry)
        tts.speak(cue)
        _vs.last_coaching = cue

        # ── FitCoins (always) ────────────────────────────────────────
        form = (_vs.last_form_score or 0.85) * 10.0
        avg_time = telemetry.get("avg_rep_sec", 2.5)
        rep_time = telemetry.get("current_rep_sec", 2.5)
        velocity_mult = min(1.5, avg_time / rep_time) if rep_time > 0 else 1.0
        coins = calculate_rewards(1, form, velocity_mult)
        with _vs.lock:
            _vs.coins_earned += coins

        # ── Forward to RAG thread (non-blocking) ─────────────────────
        try:
            _rag_q.get_nowait()
        except queue.Empty:
            pass
        _rag_q.put((rep_count, telemetry, frame, gen_at_start))


def _rag_coaching_worker() -> None:
    """**Slow thread** — runs the full SMPL-X + Groq SR-RAG pipeline in
    the background every ``_RAG_EVERY_N_REPS`` reps.  Updates the UI
    coaching text and form score.  Only speaks the detailed cue if TTS
    is not already busy with the instant cue."""
    last_rag_rep = 0

    while True:
        rep_count, telemetry, frame, gen_at_start = _rag_q.get()

        if _vs.session_gen != gen_at_start:
            continue

        fatigue = telemetry.get("neuromuscular_fatigue", False)
        should_rag = (
            rep_count == 1
            or fatigue
            or rep_count - last_rag_rep >= _RAG_EVERY_N_REPS
        )
        if not should_rag:
            continue

        smplx_payload = None
        if _smplx is not None:
            try:
                smplx_payload = _smplx.process_frame(frame)
            except Exception:
                pass

        if _rag is not None:
            try:
                telemetry_ctx = {
                    **telemetry,
                    "target_muscle_group": st.session_state.get(
                        "target_muscle_group", "",
                    ),
                }
                result = _rag.run(telemetry_ctx, smplx_payload)
                coaching = result.get("final_coaching", "")
                _vs.last_form_score = float(result.get("form_score", 0.0))
                _vs.last_coaching = coaching
                _vs.coaching_latency_ms = float(result.get("latency_ms", 0.0))
                last_rag_rep = rep_count

                _log_rep_to_terminal(rep_count, telemetry, smplx_payload, result)

                if coaching and _vs.session_gen == gen_at_start:
                    tts.speak(coaching)
            except Exception as exc:
                logger.warning("SR-RAG run failed: %s", exc)
        else:
            _vs.last_form_score = 0.85
            _vs.last_coaching = (
                "Good rep! Keep your elbows steady and control the descent."
            )
            last_rag_rep = rep_count
            if _vs.session_gen == gen_at_start:
                tts.speak(_vs.last_coaching)


threading.Thread(target=_quick_cue_worker, daemon=True, name="neurofit-quick-cue").start()
threading.Thread(target=_rag_coaching_worker, daemon=True, name="neurofit-rag").start()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants & drawing helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_AUTO_DETECT = "\U0001f916 Auto-Detect (AI Classifier)"

EXERCISE_MAP: dict[str, list[str]] = {
    "Biceps": [_AUTO_DETECT, "Barbell Bicep Curl", "Dumbbell Bicep Curl", "Hammer Curl"],
    "Triceps": [_AUTO_DETECT, "Tricep Dip", "Overhead Tricep Extension", "Tricep Pushdown"],
    "Arms": [_AUTO_DETECT, "Barbell Bicep Curl", "Hammer Curl", "Tricep Dip"],
    "Forearms": [_AUTO_DETECT, "Wrist Curl", "Reverse Wrist Curl"],
    "Chest": [_AUTO_DETECT, "Bench Press", "Push-Up", "Dumbbell Fly"],
    "Back": [_AUTO_DETECT, "Deadlift", "Bent-Over Row", "Lat Pulldown"],
    "Shoulders": [_AUTO_DETECT, "Shoulder Press", "Lateral Raise", "Front Raise"],
    "Legs": [_AUTO_DETECT, "Barbell Squat", "Lunge", "Leg Press"],
}

_EXERCISE_KEY_MAP: dict[str, str] = {
    "Barbell Bicep Curl": "barbell biceps curl",
    "Dumbbell Bicep Curl": "barbell biceps curl",
    "Hammer Curl": "hammer curl",
    "Tricep Dip": "tricep dips",
    "Overhead Tricep Extension": "tricep Pushdown",
    "Tricep Pushdown": "tricep Pushdown",
    "Bench Press": "bench press",
    "Push-Up": "push-up",
    "Dumbbell Fly": "chest fly machine",
    "Deadlift": "deadlift",
    "Bent-Over Row": "t bar row",
    "Lat Pulldown": "lat pulldown",
    "Shoulder Press": "shoulder press",
    "Lateral Raise": "lateral raise",
    "Front Raise": "lateral raise",
    "Barbell Squat": "squat",
    "Lunge": "squat",
    "Leg Press": "leg extension",
    "Wrist Curl": "barbell biceps curl",
    "Reverse Wrist Curl": "barbell biceps curl",
}

_MUSCLE_GROUP_CLASSIFIER_LABELS: dict[str, set[str]] = {
    "Biceps": {"barbell biceps curl", "hammer curl"},
    "Triceps": {"tricep dips", "tricep Pushdown"},
    "Arms": {"barbell biceps curl", "hammer curl", "tricep dips", "tricep Pushdown"},
    "Forearms": {"barbell biceps curl", "hammer curl"},
    "Chest": {"bench press", "push-up", "chest fly machine",
              "decline bench press", "incline bench press"},
    "Back": {"deadlift", "romanian deadlift", "t bar row",
             "lat pulldown", "pull Up"},
    "Shoulders": {"shoulder press", "lateral raise"},
    "Legs": {"squat", "leg extension", "leg raises", "hip thrust"},
}


def _draw_setup_overlay(frame: np.ndarray) -> None:
    """Burn a centred 'SETUP MODE' banner onto the video frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    band_h = 74
    y1 = (h - band_h) // 2
    cv2.rectangle(overlay, (0, y1), (w, y1 + band_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for text, scale, thick, dy, color in (
        ("SETUP MODE", 0.9, 2, 30, (0, 255, 255)),
        ("Get into position & click Start Set", 0.55, 1, 58, (200, 200, 200)),
    ):
        (tw, _), _ = cv2.getTextSize(text, font, scale, thick)
        cv2.putText(
            frame, text, ((w - tw) // 2, y1 + dy),
            font, scale, color, thick, cv2.LINE_AA,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shared frame-processing logic (used by BOTH WebRTC and IP Webcam)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _process_and_annotate(img: np.ndarray) -> np.ndarray:
    """Run MediaPipe on a BGR frame and return the annotated result.

    Execution modes (driven by ``_vs`` flags):
      • **Setup** (``is_workout_active=False``) — skeleton drawn, classifier
        and rep counter skipped, "SETUP MODE" overlay burned onto frame.
      • **Tracking** (``is_workout_active=True``) — full pipeline with
        locked exercise, frame-skipped classifier, rep counting, fatigue.
        On fatigue the set is **auto-stopped** and ``fatigue_stopped`` set.
    """
    h, w = img.shape[:2]
    active = _vs.is_workout_active
    locked = _vs.locked_exercise if active else ""

    if w > _MAX_INFERENCE_WIDTH:
        scale = _MAX_INFERENCE_WIDTH / float(w)
        small = cv2.resize(
            img,
            (_MAX_INFERENCE_WIDTH, max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = img

    allowed = _vs.allowed_labels if (active and not locked) else None
    annotated_small, telemetry = tracker.process_frame(
        small, tracking_active=active, locked_exercise=locked,
        allowed_labels=allowed or None,
    )

    if annotated_small.shape[:2] != (h, w):
        annotated = cv2.resize(
            annotated_small, (w, h), interpolation=cv2.INTER_LINEAR,
        )
    else:
        annotated = annotated_small

    if not active and not _vs.fatigue_stopped:
        _draw_setup_overlay(annotated)

    _vs.last_bgr = annotated

    if active and telemetry is not None:
        _vs.detected_exercise = telemetry.get("exercise", "unknown")
        _vs.exercise_confidence = telemetry.get("exercise_confidence", 0.0)
        fatigue = bool(telemetry.get("neuromuscular_fatigue", False))
        _vs.fatigue_active = fatigue

        rep_count = telemetry.get("rep_count", 0)
        if rep_count > _vs.prev_rep_count:
            _vs.prev_rep_count = rep_count
            try:
                _rep_q.get_nowait()
            except queue.Empty:
                pass
            _rep_q.put((rep_count, telemetry, annotated.copy()))

        if fatigue:
            _vs.is_workout_active = False
            _vs.fatigue_stopped = True

    return annotated


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WebRTC video callback
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """Ultra-light: stash raw frame for the background pose worker and
    immediately return the latest annotated result.  Zero heavy processing
    here keeps the WebRTC pipeline at native camera frame-rate."""
    img = frame.to_ndarray(format="bgr24")
    _vs._raw_webrtc_frame = img
    _vs._webrtc_frame_seq += 1

    out = _vs._display_webrtc_frame
    if out is not None and out.shape[:2] == img.shape[:2]:
        return av.VideoFrame.from_ndarray(out, format="bgr24")
    return av.VideoFrame.from_ndarray(cv2.flip(img, 1), format="bgr24")


def _webrtc_pose_worker() -> None:
    """Background daemon: grabs the latest raw WebRTC frame and runs the
    full MediaPipe + tracking pipeline *off* the critical WebRTC thread.
    The video callback just swaps frame references so the feed stays at
    the camera's native fps with zero jank."""
    last_seq = 0
    while True:
        seq = _vs._webrtc_frame_seq
        raw = _vs._raw_webrtc_frame
        if seq == last_seq or raw is None:
            time.sleep(0.005)
            continue
        last_seq = seq
        try:
            annotated = _process_and_annotate(raw)
            _vs._display_webrtc_frame = annotated
        except Exception:
            pass


threading.Thread(
    target=_webrtc_pose_worker, daemon=True, name="neurofit-webrtc-pose",
).start()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IP Webcam capture thread
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


_MJPEG_BOUNDARY = b"--neurofit\r\n"


def _find_free_port() -> int:
    """Bind to port 0, let the OS pick a free port, return it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class IPWebcamPoseWorker:
    """Three-thread pipeline for IP Webcam with skeleton overlay.

    Thread 1 (drain):  read phone MJPEG at full speed, keep only newest frame
    Thread 2 (pose):   MediaPipe on latest frame → JPEG encode → jpg_q
    Thread 3 (server): MJPEG HTTP server on localhost — browser <img> tag
                        plays it natively with zero JS, zero Streamlit overhead

    The MJPEG port is dynamically assigned to avoid conflicts.
    """

    _JPG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 85]

    def __init__(self, phone_url: str) -> None:
        self.phone_url = phone_url
        self.error: str = ""
        self.mjpeg_port: int = 0
        self._running = False
        self._raw_q: queue.Queue = queue.Queue(maxsize=1)
        self._jpg_q: queue.Queue = queue.Queue(maxsize=1)
        self._threads: list[threading.Thread] = []

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.error = ""
        self.mjpeg_port = _find_free_port()
        for target, name in [
            (self._drain_loop,  "ip-drain"),
            (self._pose_loop,   "ip-pose"),
            (self._server_loop, "ip-mjpeg"),
        ]:
            t = threading.Thread(target=target, daemon=True, name=name)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._running = False
        for t in self._threads:
            t.join(timeout=3)
        self._threads.clear()
        self.mjpeg_port = 0

    # ── Thread 1: drain phone MJPEG as fast as possible ──────────────

    def _drain_loop(self) -> None:
        cap = cv2.VideoCapture(self.phone_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            self.error = f"Cannot connect to {self.phone_url}"
            self._running = False
            return

        fail = 0
        while self._running:
            ret, frame = cap.read()
            if not ret:
                fail += 1
                if fail > 60:
                    self.error = "Stream lost."
                    self._running = False
                    break
                time.sleep(0.02)
                continue
            fail = 0

            h, w = frame.shape[:2]
            if w > _IP_TARGET_WIDTH:
                scale = _IP_TARGET_WIDTH / float(w)
                frame = cv2.resize(
                    frame,
                    (_IP_TARGET_WIDTH, max(1, int(round(h * scale)))),
                    interpolation=cv2.INTER_AREA,
                )

            try:
                self._raw_q.get_nowait()
            except queue.Empty:
                pass
            self._raw_q.put(frame)

        cap.release()

    # ── Thread 2: MediaPipe + JPEG encode ────────────────────────────

    def _pose_loop(self) -> None:
        while self._running:
            try:
                frame = self._raw_q.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                annotated = _process_and_annotate(frame)
                ok, buf = cv2.imencode(".jpg", annotated, self._JPG_PARAMS)
                if ok:
                    jpg = buf.tobytes()
                    _vs._ip_jpg_bytes = jpg
                    try:
                        self._jpg_q.get_nowait()
                    except queue.Empty:
                        pass
                    self._jpg_q.put(jpg)
            except Exception as exc:
                logger.warning("IP pose loop: %s", exc)

    # ── Thread 3: MJPEG HTTP server ──────────────────────────────────

    def _server_loop(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind(("127.0.0.1", self.mjpeg_port))
        except OSError as exc:
            logger.warning("MJPEG bind failed: %s", exc)
            return
        srv.listen(2)
        srv.settimeout(1.0)

        header = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: multipart/x-mixed-replace; boundary=neurofit\r\n"
            "Cache-Control: no-cache, no-store\r\n"
            "Connection: keep-alive\r\n"
            "Access-Control-Allow-Origin: *\r\n\r\n"
        ).encode()

        while self._running:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            conn.settimeout(2.0)
            try:
                conn.recv(4096)
                conn.sendall(header)
                while self._running:
                    try:
                        jpg = self._jpg_q.get(timeout=1.0)
                    except queue.Empty:
                        continue
                    part = (
                        _MJPEG_BOUNDARY
                        + b"Content-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(jpg)}\r\n\r\n".encode()
                        + jpg
                        + b"\r\n"
                    )
                    conn.sendall(part)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                try:
                    conn.close()
                except OSError:
                    pass
        srv.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Header
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

try:
    header_left, header_right = st.columns([4, 1], vertical_alignment="center")
except TypeError:
    header_left, header_right = st.columns([4, 1])
with header_left:
    st.title("Neuro-Fit")
    st.caption("Real-Time Biomechanical AI Trainer")
with header_right:

    @st.fragment(run_every="1s")
    def wallet_badge() -> None:
        with _vs.lock:
            st.session_state.fitcoin_balance = _vs.coins_earned
        st.markdown(
            '<div class="neurofit-wallet-row">'
            f'<span class="fitcoin-badge">🪙 {st.session_state.fitcoin_balance:.1f} FitCoins</span>'
            "</div>",
            unsafe_allow_html=True,
        )

    wallet_badge()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tabs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

tab_workout, tab_shop = st.tabs(["🏋️ Workout Zone", "🛒 Rewards Shop"])


# ──────────────────────────────────────────────────────────────────────
# TAB 1: Workout Zone
# ──────────────────────────────────────────────────────────────────────

with tab_workout:
    # --- Camera source selector ---
    cam_mode = st.radio(
        "Camera Source",
        ["Browser Webcam (WebRTC)", "IP Webcam (Phone)"],
        horizontal=True,
        key="cam_mode",
    )

    if cam_mode == "IP Webcam (Phone)":
        ip_col1, ip_col2 = st.columns([3, 1])
        with ip_col1:
            ip_url = st.text_input(
                "IP Webcam Stream URL",
                value="http://10.75.90.121:8080/video",
                key="ip_url",
                placeholder="http://<phone-ip>:8080/video",
            )
        with ip_col2:
            st.caption("Open IP Webcam on your phone → Start Server → copy the IP shown.")

    # ── Pre-Workout Setup Controls (cascading dropdowns) ────────────
    _workout_active = st.session_state.get("is_workout_active", False)
    _fatigue_stopped = st.session_state.get("fatigue_stopped", False) or _vs.fatigue_stopped

    setup_c1, setup_c2, setup_c3 = st.columns([2, 2, 1.5])
    with setup_c1:
        _groups = list(EXERCISE_MAP.keys())
        _cur_group = st.session_state.get("target_muscle_group", "Biceps")
        if _cur_group not in _groups:
            _cur_group = "Biceps"
        muscle = st.selectbox(
            "Target Muscle Group",
            _groups,
            index=_groups.index(_cur_group),
            disabled=_workout_active or _fatigue_stopped,
            key="_muscle_sel",
        )
        st.session_state.target_muscle_group = muscle

    with setup_c2:
        _exercises = EXERCISE_MAP.get(muscle, ["Unknown"])
        _cur_ex = st.session_state.get("target_exercise", "")
        _ex_idx = _exercises.index(_cur_ex) if _cur_ex in _exercises else 0
        exercise_choice = st.selectbox(
            "Specific Exercise",
            _exercises,
            index=_ex_idx,
            disabled=_workout_active or _fatigue_stopped,
            key="_exercise_sel",
        )
        st.session_state.target_exercise = exercise_choice

    with setup_c3:
        st.markdown("")  # vertical spacer to align with selectboxes
        if _fatigue_stopped:
            if st.button(
                "\u2705 Acknowledge & Reset",
                type="primary",
                width="stretch",
            ):
                st.session_state.fatigue_stopped = False
                _vs.fatigue_stopped = False
                st.session_state.is_workout_active = False
                _vs.is_workout_active = False
                _vs.fatigue_active = False
                _vs.locked_exercise = ""
                _vs.allowed_labels = set()
                _vs.session_gen += 1
                tts.flush()
                try:
                    _rep_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    _rag_q.get_nowait()
                except queue.Empty:
                    pass
                st.session_state.fatigue_toast_shown = False
                st.rerun()
        elif not _workout_active:
            if st.button(
                "\u25b6\ufe0f START SET",
                type="primary",
                width="stretch",
            ):
                if exercise_choice == _AUTO_DETECT:
                    _key = ""
                    _vs.allowed_labels = _MUSCLE_GROUP_CLASSIFIER_LABELS.get(muscle, set())
                else:
                    _key = _EXERCISE_KEY_MAP.get(exercise_choice, "")
                    _vs.allowed_labels = set()
                st.session_state.is_workout_active = True
                _vs.is_workout_active = True
                _vs.locked_exercise = _key
                tracker.reset()
                _vs.prev_rep_count = 0
                _vs.detected_exercise = _key or "unknown"
                _vs.exercise_confidence = 0.0
                _vs.fatigue_active = False
                _vs.fatigue_stopped = False
                st.session_state.fatigue_stopped = False
                st.session_state.fatigue_toast_shown = False
                st.rerun()
        else:
            if st.button(
                "\u23f9 STOP SET",
                type="secondary",
                width="stretch",
            ):
                st.session_state.is_workout_active = False
                _vs.is_workout_active = False
                _vs.fatigue_active = False
                _vs.locked_exercise = ""
                _vs.allowed_labels = set()
                _vs.session_gen += 1
                tts.flush()
                try:
                    _rep_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    _rag_q.get_nowait()
                except queue.Empty:
                    pass
                st.session_state.fatigue_toast_shown = False
                st.rerun()

    col_video, col_telemetry = st.columns([3, 2], gap="large")

    with col_video:
        # ── Dynamic exercise header ──────────────────────────────────────
        @st.fragment(run_every="0.4s")
        def _exercise_header() -> None:
            if _vs.fatigue_stopped:
                return  # fatigue banner takes over
            if not _vs.is_workout_active:
                st.markdown(
                    '<div class="exercise-header">'
                    "SETUP MODE &mdash; select exercise &amp; click Start Set"
                    "</div>",
                    unsafe_allow_html=True,
                )
                return
            ex_display = st.session_state.get("target_exercise", "")
            if not ex_display or ex_display == _AUTO_DETECT:
                ex = _vs.detected_exercise.replace("_", " ").title()
                conf = _vs.exercise_confidence
                if ex == "Unknown" or conf == 0.0:
                    label = "Detecting exercise&hellip;"
                else:
                    label = f"{ex} <small>({conf:.0%} &middot; AI classifier)</small>"
            else:
                label = f"{ex_display} <small>(locked)</small>"
            st.markdown(
                f'<div class="exercise-header">{label}</div>',
                unsafe_allow_html=True,
            )

        _exercise_header()

        # ── Fatigue alert overlay (cinematic — persistent until ack) ─────
        @st.fragment(run_every="0.5s")
        def _fatigue_overlay() -> None:
            if _vs.fatigue_stopped or _vs.fatigue_active:
                st.markdown(
                    '<div class="fatigue-screen-flash">'
                    '<div class="fatigue-banner">'
                    "\U0001f6a8 CRITICAL FATIGUE DETECTED: "
                    "Form breakdown imminent. "
                    "Set automatically stopped to prevent injury."
                    "</div></div>",
                    unsafe_allow_html=True,
                )
                st.error(
                    "\U0001f6a8 **CRITICAL FATIGUE DETECTED:** "
                    "Velocity dropped by 50%. Form breakdown imminent. "
                    "Set automatically stopped to prevent injury. "
                    "Click **Acknowledge & Reset** above to continue.",
                    icon="\U0001f534",
                )
                if not st.session_state.get("fatigue_toast_shown"):
                    st.toast(
                        "\U0001f6a8 CRITICAL FATIGUE \u2014 "
                        "Set auto-stopped! Rack the weight!",
                        icon="\U0001f534",
                    )
                    st.session_state.fatigue_toast_shown = True
            else:
                st.session_state.fatigue_toast_shown = False

        _fatigue_overlay()

        st.subheader("Live Feed")

        if cam_mode == "Browser Webcam (WebRTC)":
            # Stop IP capture if it was running
            w = st.session_state.get("_ip_worker")
            if w and w.running:
                w.stop()
                _vs.last_bgr = None
                _vs._ip_jpg_bytes = None

            webrtc_streamer(
                key="neurofit",
                mode=WebRtcMode.SENDRECV,
                video_frame_callback=video_frame_callback,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 720, "max": 1280},
                        "height": {"ideal": 540, "max": 720},
                        "frameRate": {"ideal": 24, "max": 30},
                    },
                    "audio": False,
                },
                async_processing=True,
            )

        else:
            # --- IP Webcam mode ---
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                start_pressed = st.button("▶ Start IP Webcam", width="stretch")
            with btn_col2:
                stop_pressed = st.button("⏹ Stop", width="stretch")

            if start_pressed:
                old = st.session_state.get("_ip_worker")
                if old and old.running:
                    old.stop()
                worker = IPWebcamPoseWorker(ip_url)
                worker.start()
                st.session_state._ip_worker = worker
                st.session_state._ip_streaming = True
                time.sleep(0.3)
                st.rerun()

            if stop_pressed:
                _vs.session_gen += 1
                tts.flush()
                try:
                    _rep_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    _rag_q.get_nowait()
                except queue.Empty:
                    pass
                w = st.session_state.get("_ip_worker")
                if w and w.running:
                    w.stop()
                st.session_state._ip_streaming = False
                st.session_state.is_workout_active = False
                _vs.is_workout_active = False
                _vs.locked_exercise = ""
                _vs.allowed_labels = set()
                _vs.fatigue_active = False
                _vs.fatigue_stopped = False
                _vs.last_bgr = None
                _vs._ip_jpg_bytes = None
                st.session_state.fatigue_stopped = False
                st.session_state.fatigue_toast_shown = False
                with _vs.lock:
                    st.session_state.fitcoin_balance = _vs.coins_earned
                st.rerun()

            streaming = st.session_state.get("_ip_streaming", False)
            worker = st.session_state.get("_ip_worker")

            if not streaming:
                st.info("Click **Start IP Webcam** to connect to your phone.")
            elif worker and worker.error:
                st.error(f"**Connection failed:** {worker.error}")
            elif worker and worker.mjpeg_port:
                port = worker.mjpeg_port
                st.markdown(
                    f'<img src="http://localhost:{port}" '
                    f'style="width:100%;border-radius:8px;background:#000;" '
                    f'alt="IP Webcam Feed"/>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("Starting MJPEG server…")

        # Coaching cue display
        if _vs.last_coaching:
            st.info(f"**AI Coach:** {_vs.last_coaching}")

    with col_telemetry:
        st.subheader("Telemetry Dashboard")

        @st.fragment(run_every="0.3s")
        def telemetry_panel() -> None:
            telemetry = tracker.latest_telemetry

            if telemetry is None:
                st.info(
                    "Waiting for pose data — start the camera and step into frame."
                )
                return

            if telemetry.get("neuromuscular_fatigue") or _vs.fatigue_stopped:
                st.error(
                    "\U0001f6a8 **CRITICAL FATIGUE DETECTED** \u2014 "
                    "Rep velocity dropped >50%. Set auto-stopped. "
                    "Click **Acknowledge & Reset** to continue.",
                    icon="\U0001f534",
                )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Reps", telemetry["rep_count"])
            m2.metric("Current Rep", f'{telemetry["current_rep_sec"]:.2f}s')
            baseline = telemetry.get("baseline_avg_sec", 0.0)
            threshold = telemetry.get("fatigue_threshold_sec", 0.0)
            if baseline > 0:
                m3.metric("Baseline Avg", f'{baseline:.2f}s')
            else:
                m3.metric("Avg Rep", f'{telemetry["avg_rep_sec"]:.2f}s')
            m4.metric(
                "Form Score",
                f'{_vs.last_form_score:.0%}',
            )
            if threshold > 0:
                reps_left = max(0, 5 - telemetry["rep_count"])
                if reps_left > 0:
                    st.caption(
                        f"Warmup: {reps_left} reps remaining before fatigue detection activates"
                    )
                else:
                    st.caption(
                        f"Fatigue triggers if rep > **{threshold:.2f}s** "
                        f"(baseline {baseline:.2f}s × 1.5)"
                    )

            ex_name = telemetry.get("exercise", "unknown").replace("_", " ").title()
            ex_conf = telemetry.get("exercise_confidence", 0.0)
            phase = telemetry.get("phase", "UNKNOWN")
            st.info(
                f"Exercise: **{ex_name}** ({ex_conf:.0%}) · Phase: **{phase}**"
            )

            st.markdown("**Joint Angles**")
            angle_cols = st.columns(4)
            for idx, (joint, deg) in enumerate(telemetry["angles_deg"].items()):
                col = angle_cols[idx % 4]
                col.metric(joint.replace("_", " ").title(), f"{deg}°")

            st.markdown("**Phase Timing**")
            t1, t2 = st.columns(2)
            t1.metric("Concentric (Up)", f'{telemetry["concentric_sec"]:.3f}s')
            t2.metric("Eccentric (Down)", f'{telemetry["eccentric_sec"]:.3f}s')

            if _vs.coaching_latency_ms > 0:
                st.caption(f"Coaching pipeline latency: {_vs.coaching_latency_ms:.0f} ms")

        telemetry_panel()

        # Sync coins from camera thread → session state
        with _vs.lock:
            st.session_state.fitcoin_balance = _vs.coins_earned

        st.divider()
        st.markdown(
            f"**FitCoins Earned This Session:** "
            f'<span class="fitcoin-badge">🪙 {st.session_state.fitcoin_balance:.1f}</span>',
            unsafe_allow_html=True,
        )

        st.divider()
        _, clear_btn_col = st.columns([3, 2])
        with clear_btn_col:
            if st.button("🔄 Clear Session", width="stretch", type="secondary"):
                _vs.session_gen += 1
                tts.flush()
                try:
                    _rep_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    _rag_q.get_nowait()
                except queue.Empty:
                    pass
                with _vs.lock:
                    st.session_state.fitcoin_balance = _vs.coins_earned
                tracker.reset()
                st.session_state.is_workout_active = False
                _vs.is_workout_active = False
                _vs.locked_exercise = ""
                _vs.allowed_labels = set()
                _vs.prev_rep_count = 0
                _vs.last_form_score = 0.0
                _vs.last_coaching = ""
                _vs.coaching_latency_ms = 0.0
                _vs.last_bgr = None
                _vs._ip_jpg_bytes = None
                _vs.detected_exercise = "unknown"
                _vs.exercise_confidence = 0.0
                _vs.fatigue_active = False
                _vs.fatigue_stopped = False
                st.session_state.fatigue_stopped = False
                st.session_state.fatigue_toast_shown = False
                st.session_state.prev_rep_count = 0
                st.session_state.last_coaching = ""
                st.session_state.last_form_score = 0.0
                st.session_state.coaching_latency_ms = 0.0
                st.rerun()


# ──────────────────────────────────────────────────────────────────────
# TAB 2: Rewards Shop
# ──────────────────────────────────────────────────────────────────────

with tab_shop:
    st.subheader("FitCoin Rewards Shop")
    st.markdown(
        "Redeem your FitCoins for real-world dietary supplements and fitness products. "
        "Product data sourced from [Open Food Facts](https://openfoodfacts.org)."
    )

    bal_col, refresh_col = st.columns([3, 1])
    with bal_col:
        st.markdown(
            f'**Your Balance:** <span class="fitcoin-badge">🪙 {st.session_state.fitcoin_balance:.1f}</span>',
            unsafe_allow_html=True,
        )
    with refresh_col:
        if st.button("🔄 Refresh Catalog"):
            st.session_state.store_catalog = None

    if st.session_state.store_catalog is None:
        with st.spinner("Loading products from Open Food Facts..."):
            st.session_state.store_catalog = fetch_store_rewards()

    catalog = st.session_state.store_catalog

    if not catalog:
        st.warning("No products available. Try refreshing the catalog.")
    else:
        purchased_names = {p["name"] for p in st.session_state.purchased_items}

        cols = st.columns(4)
        for i, product in enumerate(catalog):
            col = cols[i % 4]
            with col:
                st.image(
                    product["image_url"],
                    width="stretch",
                )
                st.markdown(f"**{product['name'][:50]}**")
                st.caption(f"Category: {product['category']}")
                cost = product["fitcoin_cost"]
                st.markdown(f"🪙 **{cost} FitCoins**")

                already_bought = product["name"] in purchased_names
                can_afford = st.session_state.fitcoin_balance >= cost

                if already_bought:
                    st.success("Purchased ✓", icon="✅")
                elif st.button(
                    f"Buy for {cost} 🪙",
                    key=f"buy_{i}",
                    disabled=not can_afford,
                ):
                    st.session_state.fitcoin_balance -= cost
                    with _vs.lock:
                        _vs.coins_earned = st.session_state.fitcoin_balance
                    st.session_state.purchased_items.append(product)
                    st.rerun()

                if not can_afford and not already_bought:
                    st.caption("Not enough FitCoins")

                st.divider()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sidebar — pipeline status
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with st.sidebar:
    st.header("Pipeline Status")

    def _status_dot(ok: bool, label: str) -> str:
        dot = "🟢" if ok else "🟡"
        return f"{dot} {label}"

    _has_classifier = tracker._classifier is not None
    st.markdown(_status_dot(True, "MediaPipe BlazePose"))
    st.markdown(
        _status_dot(
            _has_classifier,
            "Exercise Classifier (PyTorch)"
            if _has_classifier
            else "Exercise Classifier (not loaded)",
        )
    )
    st.markdown(_status_dot(True, "Generic Rep Counter"))
    st.markdown(_status_dot(True, "VBT Fatigue Detection"))
    st.markdown(
        _status_dot(
            _smplx is not None,
            "SMPL-X Spot-Check (mock)" if _smplx else "SMPL-X (disabled)",
        )
    )
    st.markdown(
        _status_dot(
            _rag is not None,
            "SR-RAG Coaching (Groq)"
            if _rag
            else "SR-RAG (no GROQ_API_KEY)",
        )
    )
    st.markdown(
        _status_dot(tts.available, "Voice Coach (pyttsx3)")
    )
    st.markdown(_status_dot(True, "FitCoin Economy"))

    st.divider()
    if _vs.fatigue_stopped:
        _active_label = "**FATIGUE STOP** \U0001f534"
    elif _vs.is_workout_active:
        _active_label = "**Set Active** \u2705"
    else:
        _active_label = "Setup Mode"
    _mg = st.session_state.get("target_muscle_group", "\u2014")
    _ex = st.session_state.get("target_exercise", "\u2014")
    st.markdown(
        f"**Status:** {_active_label}  \n"
        f"**Target:** {_mg}  \n"
        f"**Exercise:** {_ex}"
    )

    st.divider()
    st.markdown(
        "**How to use:**\n"
        "1. Select your target muscle group\n"
        "2. Choose a camera source and start the feed\n"
        "3. Get into position, then click **Start Set**\n"
        "4. The AI tracks reps, form, and fatigue in real-time"
    )

    if st.session_state.purchased_items:
        st.divider()
        st.markdown("**Purchased Items**")
        for item in st.session_state.purchased_items:
            st.markdown(f"- {item['name']}")
