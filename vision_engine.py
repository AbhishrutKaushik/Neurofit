"""
vision_engine.py — Edge Perception Layer for Neuro-Fit

Uses **MediaPipe Tasks** `PoseLandmarker` (VIDEO mode).  Recent PyPI wheels no longer
ship the legacy `mediapipe.solutions` API; Tasks is the supported path.

Pipeline:
  • BlazePose via PoseLandmarker — 33-point 3D world landmarks (hip-centered)
  • 3D joint-angle math (dot product + arccos)
  • Velocity Engine — wrist Y displacement, concentric / eccentric timing
  • VBT fatigue — current rep > 1.3 × rolling mean
  • JSON-serializable telemetry dict
"""

from __future__ import annotations

import ssl
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from pathlib import Path
from typing import Optional

import certifi
import cv2
import numpy as np
from mediapipe.tasks.python.core import base_options as base_options_lib
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarksConnections,
    RunningMode,
)
from mediapipe.tasks.python.vision.core import image as mp_image_module
from mediapipe.tasks.python.vision.core.image import ImageFormat

# ---------------------------------------------------------------------------
# BlazePose landmark indices (same numbering as legacy PoseLandmark)
# ---------------------------------------------------------------------------


class LM(IntEnum):
    """Indices into the 33-point landmark list."""

    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


JOINT_TRIPLETS: dict[str, tuple[int, int, int]] = {
    "left_elbow": (LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST),
    "right_elbow": (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST),
    "left_knee": (LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE),
    "right_knee": (LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
    "left_shoulder": (LM.LEFT_HIP, LM.LEFT_SHOULDER, LM.LEFT_ELBOW),
    "right_shoulder": (LM.RIGHT_HIP, LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW),
    "left_hip": (LM.LEFT_SHOULDER, LM.LEFT_HIP, LM.LEFT_KNEE),
    "right_hip": (LM.RIGHT_SHOULDER, LM.RIGHT_HIP, LM.RIGHT_KNEE),
}

# Official task bundle (lite — smaller/faster; swap URL for full/heavy if needed)
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

FATIGUE_MULTIPLIER: float = 1.3
REP_HISTORY_SIZE: int = 8
PHASE_THRESHOLD: float = 0.04


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------


def _download_https(url: str, dest: Path, timeout_sec: int = 120) -> None:
    """Download over HTTPS using Certifi’s CA bundle (fixes macOS “unable to get local issuer”)."""
    ctx = ssl.create_default_context(cafile=certifi.where())
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Neuro-Fit/1.0 (pose model download)"},
    )
    with urllib.request.urlopen(req, context=ctx, timeout=timeout_sec) as resp:
        with dest.open("wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)


def ensure_pose_model_path(cache_dir: Optional[Path] = None) -> Path:
    """Return path to `.task` model file, downloading once if missing."""
    root = cache_dir or (Path(__file__).resolve().parent / "models")
    root.mkdir(parents=True, exist_ok=True)
    dest = root / "pose_landmarker_lite.task"
    if not dest.is_file():
        _download_https(POSE_MODEL_URL, dest)
    return dest


# ---------------------------------------------------------------------------
# Phase / rep state
# ---------------------------------------------------------------------------


class Phase(Enum):
    IDLE = auto()
    CONCENTRIC = auto()
    ECCENTRIC = auto()


@dataclass
class RepState:
    phase: Phase = Phase.IDLE
    rep_count: int = 0
    phase_start_time: float = 0.0
    last_wrist_y: Optional[float] = None
    concentric_duration: float = 0.0
    eccentric_duration: float = 0.0
    current_rep_duration: float = 0.0
    rep_durations: deque = field(default_factory=lambda: deque(maxlen=REP_HISTORY_SIZE))
    neuromuscular_fatigue: bool = False


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def angle_between_3d_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ∠ABC in degrees (3-D), via dot product and arccos."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


# ---------------------------------------------------------------------------
# Drawing (OpenCV wireframe — Tasks API has no solutions.drawing_utils)
# ---------------------------------------------------------------------------


def _draw_pose_skeleton(
    frame_bgr: np.ndarray,
    landmarks_norm: list,
) -> None:
    """Overlay pose connections + joints using normalized image coordinates."""
    h, w = frame_bgr.shape[:2]
    for conn in PoseLandmarksConnections.POSE_LANDMARKS:
        if conn.start >= len(landmarks_norm) or conn.end >= len(landmarks_norm):
            continue
        a = landmarks_norm[conn.start]
        b = landmarks_norm[conn.end]
        if a.x is None or a.y is None or b.x is None or b.y is None:
            continue
        p1 = (int(a.x * w), int(a.y * h))
        p2 = (int(b.x * w), int(b.y * h))
        cv2.line(frame_bgr, p1, p2, (0, 220, 0), 2, lineType=cv2.LINE_AA)

    for lm in landmarks_norm:
        if lm.x is None or lm.y is None:
            continue
        cv2.circle(
            frame_bgr,
            (int(lm.x * w), int(lm.y * h)),
            4,
            (0, 165, 255),
            -1,
            lineType=cv2.LINE_AA,
        )


# ---------------------------------------------------------------------------
# PoseTracker
# ---------------------------------------------------------------------------


class PoseTracker:
    """Pose estimation + velocity-based fatigue using MediaPipe Tasks."""

    def __init__(
        self,
        camera_index: int = 0,
        model_path: Optional[Path | str] = None,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {camera_index}. "
                "Check that your webcam is connected and not in use."
            )

        path = Path(model_path) if model_path else ensure_pose_model_path()
        if not path.is_file():
            raise FileNotFoundError(f"Pose model not found: {path}")

        options = PoseLandmarkerOptions(
            base_options=base_options_lib.BaseOptions(model_asset_path=str(path)),
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._video_ts_ms: int = 0
        self.rep_state = RepState()

    def process_frame(self) -> tuple[Optional[np.ndarray], Optional[dict]]:
        ok, frame = self.cap.read()
        if not ok:
            return None, None

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        mp_image = mp_image_module.Image(image_format=ImageFormat.SRGB, data=rgb)

        self._video_ts_ms += 33
        result = self._landmarker.detect_for_video(mp_image, self._video_ts_ms)

        if not result.pose_landmarks:
            return frame, None

        norm = result.pose_landmarks[0]
        _draw_pose_skeleton(frame, norm)

        if not result.pose_world_landmarks:
            return frame, None

        world = result.pose_world_landmarks[0]
        angles = self._compute_angles(world)
        self._update_velocity_engine(world)

        return frame, self._build_telemetry(angles)

    def release(self) -> None:
        self.cap.release()
        self._landmarker.close()

    def _xyz(self, lm) -> np.ndarray:
        return np.array(
            [float(lm.x or 0.0), float(lm.y or 0.0), float(lm.z or 0.0)],
            dtype=np.float64,
        )

    def _compute_angles(self, world_landmarks: list) -> dict[str, float]:
        angles: dict[str, float] = {}
        for name, (ia, ib, ic) in JOINT_TRIPLETS.items():
            a = self._xyz(world_landmarks[ia])
            b = self._xyz(world_landmarks[ib])
            c = self._xyz(world_landmarks[ic])
            angles[name] = round(angle_between_3d_points(a, b, c), 1)
        return angles

    def _update_velocity_engine(self, world_landmarks: list) -> None:
        wrist_y = (
            float(world_landmarks[LM.LEFT_WRIST].y or 0.0)
            + float(world_landmarks[LM.RIGHT_WRIST].y or 0.0)
        ) / 2.0
        now = time.monotonic()
        state = self.rep_state

        if state.last_wrist_y is None:
            state.last_wrist_y = wrist_y
            state.phase_start_time = now
            return

        delta_y = wrist_y - state.last_wrist_y
        state.last_wrist_y = wrist_y

        if delta_y < -PHASE_THRESHOLD and state.phase != Phase.CONCENTRIC:
            if state.phase == Phase.ECCENTRIC:
                state.eccentric_duration = now - state.phase_start_time
            state.phase = Phase.CONCENTRIC
            state.phase_start_time = now

        elif delta_y > PHASE_THRESHOLD and state.phase != Phase.ECCENTRIC:
            if state.phase == Phase.CONCENTRIC:
                state.concentric_duration = now - state.phase_start_time
            state.phase = Phase.ECCENTRIC
            state.phase_start_time = now

            if state.concentric_duration > 0:
                state.current_rep_duration = (
                    state.concentric_duration + state.eccentric_duration
                )
                state.rep_count += 1
                state.rep_durations.append(state.current_rep_duration)
                self._evaluate_fatigue()

    def _evaluate_fatigue(self) -> None:
        state = self.rep_state
        if len(state.rep_durations) < 2:
            state.neuromuscular_fatigue = False
            return
        avg_duration = float(np.mean(state.rep_durations))
        state.neuromuscular_fatigue = (
            state.current_rep_duration > FATIGUE_MULTIPLIER * avg_duration
        )

    def _build_telemetry(self, angles: dict[str, float]) -> dict:
        state = self.rep_state
        avg_dur = (
            round(float(np.mean(state.rep_durations)), 3)
            if state.rep_durations
            else 0.0
        )
        return {
            "rep_count": state.rep_count,
            "phase": state.phase.name,
            "angles_deg": angles,
            "concentric_sec": round(state.concentric_duration, 3),
            "eccentric_sec": round(state.eccentric_duration, 3),
            "current_rep_sec": round(state.current_rep_duration, 3),
            "avg_rep_sec": avg_dur,
            "neuromuscular_fatigue": state.neuromuscular_fatigue,
        }
