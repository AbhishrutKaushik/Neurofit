"""
app.py — Streamlit Frontend for Neuro-Fit (Phase 1 / Phase 2)

Two-column real-time layout:
  Column 1  → Live webcam feed with MediaPipe skeletal overlay
  Column 2  → Telemetry dashboard (rep count, joint angles, fatigue alert)

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import cv2
import streamlit as st

from urllib.error import URLError

from vision_engine import PoseTracker

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Neuro-Fit · AI Fitness Trainer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Minimal CSS overrides for a cleaner dashboard aesthetic
st.markdown(
    """
    <style>
    /* Tighten spacing around the top header */
    .block-container { padding-top: 1.5rem; }

    /* Metric cards */
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
# Session-state initialisation (survives Streamlit reruns)
# ---------------------------------------------------------------------------

if "tracker" not in st.session_state:
    st.session_state.tracker = None
    st.session_state.running = False

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Controls")
    start = st.button("▶  Start Tracking", use_container_width=True)
    stop = st.button("■  Stop Tracking", use_container_width=True)
    st.divider()
    st.markdown(
        "**Phase 1 / 2** — Edge Perception  \n"
        "MediaPipe BlazePose · Velocity Engine · VBT Fatigue Flag"
    )

if start:
    if st.session_state.tracker is None:
        try:
            st.session_state.tracker = PoseTracker()
            st.session_state.running = True
        except (RuntimeError, OSError, FileNotFoundError, URLError) as exc:
            st.error(str(exc))

if stop:
    if st.session_state.tracker is not None:
        st.session_state.tracker.release()
        st.session_state.tracker = None
    st.session_state.running = False

# ---------------------------------------------------------------------------
# Main two-column layout
# ---------------------------------------------------------------------------

col_vision, col_telemetry = st.columns([3, 2], gap="large")

with col_vision:
    st.subheader("Live Feed")
    frame_slot = st.empty()

with col_telemetry:
    st.subheader("Telemetry Dashboard")
    alert_slot = st.empty()
    metrics_row = st.empty()
    phase_slot = st.empty()
    angles_slot = st.empty()
    timing_slot = st.empty()

# ---------------------------------------------------------------------------
# Real-time processing loop
# ---------------------------------------------------------------------------

if st.session_state.running and st.session_state.tracker is not None:
    tracker: PoseTracker = st.session_state.tracker

    while st.session_state.running:
        frame, telemetry = tracker.process_frame()

        if frame is None:
            frame_slot.warning("Camera feed lost — check your webcam connection.")
            break

        # Convert BGR → RGB for Streamlit's st.image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_slot.image(frame_rgb, channels="RGB", use_container_width=True)

        if telemetry is None:
            continue

        # ---- Fatigue alert ------------------------------------------------
        if telemetry["neuromuscular_fatigue"]:
            alert_slot.error(
                "⚠️  **Neuromuscular Fatigue Detected** — "
                "Rep velocity has dropped >30 % below your rolling average. "
                "Consider resting before form breakdown occurs.",
                icon="🔴",
            )
        else:
            alert_slot.empty()

        # ---- Top-level metrics -------------------------------------------
        with metrics_row.container():
            m1, m2, m3 = st.columns(3)
            m1.metric("Reps", telemetry["rep_count"])
            m2.metric("Current Rep", f'{telemetry["current_rep_sec"]:.2f} s')
            m3.metric("Avg Rep", f'{telemetry["avg_rep_sec"]:.2f} s')

        # ---- Phase indicator ---------------------------------------------
        phase_slot.info(f"Movement Phase: **{telemetry['phase']}**")

        # ---- Joint angles ------------------------------------------------
        with angles_slot.container():
            st.markdown("**Joint Angles (degrees)**")
            angle_cols = st.columns(4)
            angle_items = list(telemetry["angles_deg"].items())
            for idx, (joint, deg) in enumerate(angle_items):
                col = angle_cols[idx % 4]
                label = joint.replace("_", " ").title()
                col.metric(label, f"{deg}°")

        # ---- Phase timing breakdown --------------------------------------
        with timing_slot.container():
            st.markdown("**Phase Timing**")
            t1, t2 = st.columns(2)
            t1.metric("Concentric (Up)", f'{telemetry["concentric_sec"]:.3f} s')
            t2.metric("Eccentric (Down)", f'{telemetry["eccentric_sec"]:.3f} s')
else:
    frame_slot.info("Press **▶ Start Tracking** in the sidebar to begin.")
