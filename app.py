"""
app.py — Streamlit + WebRTC Frontend for Neuro-Fit

Video runs inside a WebRTC peer connection (no lag).
Telemetry panel auto-refreshes via @st.fragment(run_every=...).

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import av
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

load_dotenv(Path(__file__).resolve().parent / ".env")
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from vision_engine import PoseTracker

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Neuro-Fit · AI Fitness Trainer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    [data-testid="stMetric"] {
        background: #0e1117;
        border: 1px solid #262730;
        border-radius: 0.6rem;
        padding: 0.8rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Neuro-Fit")
st.caption("Real-Time Biomechanical AI Trainer · Edge Perception Loop")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Controls")
    st.markdown(
        "**Phase 1 / 2** — Edge Perception  \n"
        "MediaPipe BlazePose · Curl State Machine · VBT Fatigue Flag"
    )
    st.divider()
    st.markdown(
        "**How to use:** Click **START** on the video panel, "
        "allow camera access, and begin curling."
    )

# ---------------------------------------------------------------------------
# Shared PoseTracker — persists across reruns AND WebRTC callback thread
# ---------------------------------------------------------------------------

if "tracker" not in st.session_state:
    st.session_state.tracker = PoseTracker()

tracker: PoseTracker = st.session_state.tracker


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """Runs in the WebRTC thread on every webcam frame."""
    img = frame.to_ndarray(format="bgr24")
    annotated, _ = tracker.process_frame(img)
    return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ---------------------------------------------------------------------------
# Two-column layout
# ---------------------------------------------------------------------------

col_vision, col_telemetry = st.columns([3, 2], gap="large")

with col_vision:
    st.subheader("Live Feed")
    webrtc_streamer(
        key="neurofit",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# ---------------------------------------------------------------------------
# Telemetry panel — auto-refreshes every 300 ms via st.fragment
# ---------------------------------------------------------------------------

with col_telemetry:
    st.subheader("Telemetry Dashboard")

    @st.fragment(run_every="0.3s")
    def telemetry_panel() -> None:
        telemetry = tracker.latest_telemetry

        if telemetry is None:
            st.info("Waiting for pose data — start the camera and step into frame.")
            return

        # Fatigue alert
        if telemetry["neuromuscular_fatigue"]:
            st.error(
                "**Neuromuscular Fatigue Detected** — "
                "Rep velocity dropped >30 % below rolling average. "
                "Consider resting before form breakdown.",
                icon="🔴",
            )

        # Top metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Reps", telemetry["rep_count"])
        m2.metric("Current Rep", f'{telemetry["current_rep_sec"]:.2f} s')
        m3.metric("Avg Rep", f'{telemetry["avg_rep_sec"]:.2f} s')

        # Curl position
        st.info(f"Curl Position: **{telemetry['curl_position']}**")

        # Joint angles
        st.markdown("**Joint Angles (degrees)**")
        angle_cols = st.columns(4)
        for idx, (joint, deg) in enumerate(telemetry["angles_deg"].items()):
            col = angle_cols[idx % 4]
            col.metric(joint.replace("_", " ").title(), f"{deg}°")

        # Phase timing
        st.markdown("**Phase Timing**")
        t1, t2 = st.columns(2)
        t1.metric("Concentric (Up)", f'{telemetry["concentric_sec"]:.3f} s')
        t2.metric("Eccentric (Down)", f'{telemetry["eccentric_sec"]:.3f} s')

    telemetry_panel()
