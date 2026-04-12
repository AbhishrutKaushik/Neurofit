"""
vision_engine.py — Edge Perception Layer for Neuro-Fit

Pipeline:
  • BlazePose via PoseLandmarker (Tasks API, VIDEO mode)
  • Bicep-curl rep counting via elbow-angle state machine (UP ↔ DOWN)
  • VBT fatigue flag: current rep > 1.3 × rolling mean duration
  • JSON-serializable telemetry dict
"""

from __future__ import annotations

import json
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
# BlazePose landmark indices
# ---------------------------------------------------------------------------


class LM(IntEnum):
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

# ---------------------------------------------------------------------------
# Curl thresholds — tuned for seated / restricted ROM
# ---------------------------------------------------------------------------

CURL_UP_THRESHOLD: float = 70.0    # elbow ≤ this → arm is curled (UP)
CURL_DOWN_THRESHOLD: float = 110.0  # elbow ≥ this → arm is extended (DOWN)

FATIGUE_MULTIPLIER: float = 1.3
REP_HISTORY_SIZE: int = 8

# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


def _download_https(url: str, dest: Path, timeout_sec: int = 120) -> None:
    ctx = ssl.create_default_context(cafile=certifi.where())
    req = urllib.request.Request(
        url, headers={"User-Agent": "Neuro-Fit/1.0 (pose model download)"},
    )
    with urllib.request.urlopen(req, context=ctx, timeout=timeout_sec) as resp:
        with dest.open("wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)


def ensure_pose_model_path(cache_dir: Optional[Path] = None) -> Path:
    root = cache_dir or (Path(__file__).resolve().parent / "models")
    root.mkdir(parents=True, exist_ok=True)
    dest = root / "pose_landmarker_lite.task"
    if not dest.is_file():
        _download_https(POSE_MODEL_URL, dest)
    return dest


# ---------------------------------------------------------------------------
# Curl state machine
# ---------------------------------------------------------------------------


class CurlPosition(Enum):
    """Binary flag: is the arm UP (curled) or DOWN (extended)?"""
    UNKNOWN = auto()
    UP = auto()
    DOWN = auto()


@dataclass
class RepState:
    curl_position: CurlPosition = CurlPosition.UNKNOWN
    rep_count: int = 0
    phase_start_time: float = 0.0
    concentric_duration: float = 0.0
    eccentric_duration: float = 0.0
    current_rep_duration: float = 0.0
    rep_durations: deque = field(default_factory=lambda: deque(maxlen=REP_HISTORY_SIZE))
    neuromuscular_fatigue: bool = False


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def angle_between_3d_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ∠ABC in degrees (b is the vertex)."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


# ---------------------------------------------------------------------------
# Drawing helper
# ---------------------------------------------------------------------------


def _draw_pose_skeleton(frame_bgr: np.ndarray, landmarks_norm: list) -> None:
    h, w = frame_bgr.shape[:2]
    for conn in PoseLandmarksConnections.POSE_LANDMARKS:
        if conn.start >= len(landmarks_norm) or conn.end >= len(landmarks_norm):
            continue
        a, b = landmarks_norm[conn.start], landmarks_norm[conn.end]
        if a.x is None or a.y is None or b.x is None or b.y is None:
            continue
        cv2.line(
            frame_bgr,
            (int(a.x * w), int(a.y * h)),
            (int(b.x * w), int(b.y * h)),
            (0, 220, 0), 2, lineType=cv2.LINE_AA,
        )
    for lm in landmarks_norm:
        if lm.x is None or lm.y is None:
            continue
        cv2.circle(
            frame_bgr, (int(lm.x * w), int(lm.y * h)),
            4, (0, 165, 255), -1, lineType=cv2.LINE_AA,
        )


# ---------------------------------------------------------------------------
# PoseTracker
# ---------------------------------------------------------------------------


class PoseTracker:
    """Real-time pose tracking with angle-based bicep-curl rep counting."""

    def __init__(
        self,
        model_path: Optional[Path | str] = None,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
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
        self.latest_telemetry: Optional[dict] = None

    # -- public API used by the WebRTC callback ------------------------------

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, Optional[dict]]:
        """Run pose estimation on a BGR frame (already read from camera).

        Returns (annotated_bgr_frame, telemetry_dict_or_None).
        """
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
        self._update_curl_state_machine(angles)

        telemetry = self._build_telemetry(angles)
        self.latest_telemetry = telemetry
        return frame, telemetry

    def reset(self) -> None:
        """Reset rep counter, fatigue state, and telemetry for a new session."""
        self.rep_state = RepState()
        self.latest_telemetry = None

    def close(self) -> None:
        self._landmarker.close()

    # -- internals -----------------------------------------------------------

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

    def _update_curl_state_machine(self, angles: dict[str, float]) -> None:
        """Angle-based binary state machine: DOWN ↔ UP.

        A rep is counted on a full DOWN → UP → DOWN cycle.
        Uses the *average* of left + right elbow angles.
        """
        elbow_angle = (angles.get("left_elbow", 180.0) + angles.get("right_elbow", 180.0)) / 2.0
        now = time.monotonic()
        state = self.rep_state

        if state.curl_position == CurlPosition.UNKNOWN:
            # Initialise based on first reading
            if elbow_angle >= CURL_DOWN_THRESHOLD:
                state.curl_position = CurlPosition.DOWN
            elif elbow_angle <= CURL_UP_THRESHOLD:
                state.curl_position = CurlPosition.UP
            state.phase_start_time = now
            return

        # Transition DOWN → UP  (concentric phase complete)
        if state.curl_position == CurlPosition.DOWN and elbow_angle <= CURL_UP_THRESHOLD:
            state.concentric_duration = now - state.phase_start_time
            state.curl_position = CurlPosition.UP
            state.phase_start_time = now

        # Transition UP → DOWN  (eccentric phase complete → 1 full rep)
        elif state.curl_position == CurlPosition.UP and elbow_angle >= CURL_DOWN_THRESHOLD:
            state.eccentric_duration = now - state.phase_start_time
            state.curl_position = CurlPosition.DOWN
            state.phase_start_time = now

            state.current_rep_duration = state.concentric_duration + state.eccentric_duration
            state.rep_count += 1
            state.rep_durations.append(state.current_rep_duration)
            self._evaluate_fatigue()
            self._log_rep(angles)

    def _log_rep(self, angles: dict[str, float]) -> None:
        """Print a clean JSON payload to the terminal after every completed rep."""
        state = self.rep_state
        payload = {
            "exercise": "bicep_curl",
            "rep": state.rep_count,
            "rep_time": round(state.current_rep_duration, 3),
            "concentric_sec": round(state.concentric_duration, 3),
            "eccentric_sec": round(state.eccentric_duration, 3),
            "avg_rep_time": round(float(np.mean(state.rep_durations)), 3),
            "fatigue": state.neuromuscular_fatigue,
            "elbow_angle_avg": round(
                (angles.get("left_elbow", 0.0) + angles.get("right_elbow", 0.0)) / 2.0, 1
            ),
        }
        print(json.dumps(payload, indent=2), flush=True)
        if state.neuromuscular_fatigue:
            print(">>> neuromuscular_fatigue: True", flush=True)

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
            "curl_position": state.curl_position.name,
            "angles_deg": angles,
            "concentric_sec": round(state.concentric_duration, 3),
            "eccentric_sec": round(state.eccentric_duration, 3),
            "current_rep_sec": round(state.current_rep_duration, 3),
            "avg_rep_sec": avg_dur,
            "neuromuscular_fatigue": state.neuromuscular_fatigue,
        }
