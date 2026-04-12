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

_MAX_INFERENCE_WIDTH = 640
_IP_TARGET_WIDTH = 480

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


if "_vstate" not in st.session_state:
    st.session_state._vstate = _VideoState()
_vs: _VideoState = st.session_state._vstate


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

_coaching_q: queue.Queue = queue.Queue(maxsize=1)


def _coaching_worker() -> None:
    """Persistent daemon: pulls the latest (rep, telemetry, frame) tuple from
    the queue and runs the full SMPL-X + Groq SR-RAG + TTS + FitCoin pipeline.

    Only one coaching call runs at a time.  If the user does fast reps during
    a Groq call, intermediate reps are discarded but the MOST RECENT rep is
    always queued and processed next — no rep is silently lost.

    A generation counter (_vs.session_gen) is checked before speaking — if the
    user stopped/cleared the session while a Groq call was in-flight, the
    result is silently discarded instead of being spoken aloud.
    """
    while True:
        rep_count, telemetry, frame = _coaching_q.get()
        gen_at_start = _vs.session_gen

        smplx_payload = None
        if _smplx is not None:
            try:
                smplx_payload = _smplx.process_frame(frame)
            except Exception:
                pass

        if _rag is not None:
            try:
                result = _rag.run(telemetry, smplx_payload)
                coaching = result.get("final_coaching", "")
                _vs.last_form_score = float(result.get("form_score", 0.0))
                _vs.last_coaching = coaching
                _vs.coaching_latency_ms = float(result.get("latency_ms", 0.0))

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
            if _vs.session_gen == gen_at_start:
                tts.speak(_vs.last_coaching)

        if _vs.session_gen != gen_at_start:
            continue

        form = (_vs.last_form_score or 0.85) * 10.0
        avg_time = telemetry.get("avg_rep_sec", 2.5)
        rep_time = telemetry.get("current_rep_sec", 2.5)
        velocity_mult = min(1.5, avg_time / rep_time) if rep_time > 0 else 1.0
        coins = calculate_rewards(1, form, velocity_mult)
        with _vs.lock:
            _vs.coins_earned += coins


threading.Thread(target=_coaching_worker, daemon=True, name="neurofit-coaching").start()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shared frame-processing logic (used by BOTH WebRTC and IP Webcam)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _process_and_annotate(img: np.ndarray) -> np.ndarray:
    """Run MediaPipe pose estimation on a BGR frame, draw the skeleton,
    and return the annotated frame.

    This function is intentionally lightweight — it does ONLY pose inference
    and skeleton drawing.  Coaching (Groq API), SMPL-X, TTS, and FitCoin
    calculations are offloaded to the ``_coaching_worker`` background thread
    so the video pipeline is never blocked by network calls.
    """
    h, w = img.shape[:2]

    if w > _MAX_INFERENCE_WIDTH:
        scale = _MAX_INFERENCE_WIDTH / float(w)
        small = cv2.resize(
            img,
            (_MAX_INFERENCE_WIDTH, max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = img

    annotated_small, telemetry = tracker.process_frame(small)

    if annotated_small.shape[:2] != (h, w):
        annotated = cv2.resize(
            annotated_small, (w, h), interpolation=cv2.INTER_LINEAR
        )
    else:
        annotated = annotated_small

    _vs.last_bgr = annotated

    if telemetry is not None:
        rep_count = telemetry.get("rep_count", 0)
        if rep_count > _vs.prev_rep_count:
            _vs.prev_rep_count = rep_count
            try:
                _coaching_q.get_nowait()
            except queue.Empty:
                pass
            _coaching_q.put((rep_count, telemetry, annotated.copy()))

    return annotated


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WebRTC video callback
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """Runs in the WebRTC thread on every webcam frame.
    Now fully non-blocking — MediaPipe is fast (~15-30ms), coaching is async."""
    img = frame.to_ndarray(format="bgr24")
    return av.VideoFrame.from_ndarray(_process_and_annotate(img), format="bgr24")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IP Webcam capture thread
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


_MJPEG_PORT = 8765  # local MJPEG server port (accessible as http://localhost:8765)


class IPWebcamPoseWorker:
    """Three-thread pipeline for zero-lag IP Webcam with skeleton overlay.

    Architecture
    ──────────────────────────────────────────────────────
    Thread 1 (drain):  read phone MJPEG at full speed → keep only newest frame
    Thread 2 (pose):   MediaPipe on every frame → JPEG encode → push to queue
    Thread 3 (server): serve annotated MJPEG on http://localhost:8765

    Coaching (Groq API / TTS / SMPL-X) runs in a SEPARATE fire-and-forget
    thread spawned by _process_and_annotate — it NEVER blocks the video path.
    """

    _BOUNDARY = b"neurofit"

    def __init__(self, phone_url: str) -> None:
        self.phone_url = phone_url
        self.error: str = ""
        self._running = False
        self._raw_q: queue.Queue = queue.Queue(maxsize=1)  # latest raw frame
        self._jpg_q: queue.Queue = queue.Queue(maxsize=1)  # latest annotated JPEG
        self._threads: list[threading.Thread] = []

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.error = ""
        for target, name in [
            (self._drain_loop,   "ip-drain"),        # drains phone buffer → _raw_q
            (self._pose_loop,    "ip-pose"),          # MediaPipe on latest raw frame
            (self._server_loop,  "ip-mjpeg-server"),  # serves annotated JPEG as MJPEG
        ]:
            t = threading.Thread(target=target, daemon=True, name=name)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._running = False
        for t in self._threads:
            t.join(timeout=3)
        self._threads.clear()

    # ── Thread 1: drain phone MJPEG as fast as possible ─────────────────

    def _drain_loop(self) -> None:
        """Reads phone frames at full speed, keeps ONLY the newest in _raw_q.
        Never blocks on pose — lag is impossible because old frames are dropped.
        """
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

            # Overwrite old raw frame — pose thread always gets the freshest one
            try:
                self._raw_q.get_nowait()
            except queue.Empty:
                pass
            self._raw_q.put(frame)

        cap.release()

    # ── Thread 2: pose inference + JPEG encode ────────────────────────

    def _pose_loop(self) -> None:
        """Runs MediaPipe on EVERY frame from _raw_q — no skipping.
        Coaching is offloaded inside _process_and_annotate so this thread
        only waits on MediaPipe (~15-30 ms) and JPEG encoding (~2-5 ms).
        """
        jpg_params = [cv2.IMWRITE_JPEG_QUALITY, 80]

        while self._running:
            try:
                frame = self._raw_q.get(timeout=1.0)
            except queue.Empty:
                continue

            display = _process_and_annotate(frame)

            _, jpg_arr = cv2.imencode(".jpg", display, jpg_params)
            jpg = jpg_arr.tobytes()

            try:
                self._jpg_q.get_nowait()
            except queue.Empty:
                pass
            self._jpg_q.put(jpg)

    # ── MJPEG server thread ───────────────────────────────────────────

    def _server_loop(self) -> None:
        """Tiny single-client MJPEG HTTP server on localhost:_MJPEG_PORT."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind(("127.0.0.1", _MJPEG_PORT))
        except OSError:
            # Port already in use (previous Streamlit run) — that's fine,
            # the capture thread still does pose inference.
            srv.close()
            return
        srv.listen(5)
        srv.settimeout(1.0)

        header = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: multipart/x-mixed-replace; "
            f"boundary={self._BOUNDARY.decode()}\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n\r\n"
        ).encode()

        while self._running:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue

            conn.settimeout(2.0)
            try:
                conn.recv(4096)  # consume HTTP request headers
                conn.sendall(header)
                while self._running:
                    try:
                        jpg = self._jpg_q.get(timeout=1.0)
                    except queue.Empty:
                        continue
                    part = (
                        f"--{self._BOUNDARY.decode()}\r\n"
                        "Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(jpg)}\r\n\r\n"
                    ).encode() + jpg + b"\r\n"
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

    col_video, col_telemetry = st.columns([3, 2], gap="large")

    with col_video:
        st.subheader("Live Feed")

        if cam_mode == "Browser Webcam (WebRTC)":
            # Stop IP capture if it was running
            if "_ip_capture" in st.session_state and st.session_state._ip_capture.running:
                st.session_state._ip_capture.stop()

            webrtc_streamer(
                key="neurofit",
                mode=WebRtcMode.SENDRECV,
                video_frame_callback=video_frame_callback,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 640, "max": 1280},
                        "height": {"ideal": 480, "max": 720},
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
                start_pressed = st.button("▶ Start IP Webcam", use_container_width=True)
            with btn_col2:
                stop_pressed = st.button("⏹ Stop", use_container_width=True)

            if start_pressed:
                # Kill any old worker first
                old = st.session_state.get("_ip_worker")
                if old and old.running:
                    old.stop()
                worker = IPWebcamPoseWorker(ip_url)
                worker.start()
                st.session_state._ip_worker = worker
                st.session_state._ip_streaming = True

            if stop_pressed:
                _vs.session_gen += 1
                tts.flush()
                try:
                    _coaching_q.get_nowait()
                except queue.Empty:
                    pass
                w = st.session_state.get("_ip_worker")
                if w and w.running:
                    w.stop()
                st.session_state._ip_streaming = False
                with _vs.lock:
                    st.session_state.fitcoin_balance = _vs.coins_earned
                st.rerun()

            streaming = st.session_state.get("_ip_streaming", False)
            worker = st.session_state.get("_ip_worker")

            if not streaming:
                st.info("Click **▶ Start IP Webcam** to connect to your phone.")
            elif worker and worker.error:
                st.error(f"**Connection failed:** {worker.error}")
            else:
                # ── Annotated MJPEG served from Python → browser <img> tag ──
                # Python draws the BlazePose skeleton and re-serves frames on
                # localhost:8765.  The browser simply plays that as MJPEG.
                local_url = f"http://localhost:{_MJPEG_PORT}"
                st.markdown(
                    f"""
                    <div style="width:100%; border-radius:8px; overflow:hidden;
                                background:#000; aspect-ratio:4/3;">
                        <img src="{local_url}"
                             style="width:100%; height:100%; object-fit:cover;"
                             onerror="this.parentElement.innerHTML=
                               '<p style=color:white;padding:1rem>❌ Local MJPEG server not ready yet — wait 2s and refresh.</p>'">
                    </div>
                    <p style="font-size:0.75rem; color:#888; margin-top:0.25rem;">
                        🦴 Skeleton overlay via local MJPEG · Telemetry updates on right →
                    </p>
                    """,
                    unsafe_allow_html=True,
                )

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

            if telemetry.get("neuromuscular_fatigue"):
                st.error(
                    "**Neuromuscular Fatigue Detected** — "
                    "Rep velocity dropped >30% below rolling average. "
                    "Consider resting before form breakdown.",
                    icon="🔴",
                )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Reps", telemetry["rep_count"])
            m2.metric("Current Rep", f'{telemetry["current_rep_sec"]:.2f}s')
            m3.metric("Avg Rep", f'{telemetry["avg_rep_sec"]:.2f}s')
            m4.metric(
                "Form Score",
                f'{_vs.last_form_score:.0%}',
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
            if st.button("🔄 Clear Session", use_container_width=True, type="secondary"):
                _vs.session_gen += 1
                tts.flush()
                try:
                    _coaching_q.get_nowait()
                except queue.Empty:
                    pass
                with _vs.lock:
                    st.session_state.fitcoin_balance = _vs.coins_earned
                tracker.reset()
                _vs.prev_rep_count = 0
                _vs.last_form_score = 0.0
                _vs.last_coaching = ""
                _vs.coaching_latency_ms = 0.0
                _vs.last_bgr = None
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
                    use_container_width=True,
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

    st.markdown(_status_dot(True, "MediaPipe BlazePose"))
    st.markdown(_status_dot(True, "Bicep-Curl State Machine"))
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
    st.markdown(
        "**How to use:** Select your camera source, click **START**, "
        "allow camera access, and begin curling."
    )

    if st.session_state.purchased_items:
        st.divider()
        st.markdown("**Purchased Items**")
        for item in st.session_state.purchased_items:
            st.markdown(f"- {item['name']}")
