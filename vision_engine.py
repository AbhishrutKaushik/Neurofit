"""
vision_engine.py — Edge Perception Layer for Neuro-Fit
======================================================

Pipeline:
  • BlazePose via PoseLandmarker (Tasks API, VIDEO mode)
  • **PyTorch exercise classifier** — predicts the current exercise from
    99 world-landmark features.  Falls back to ``"unknown"`` when the
    checkpoint is absent (Mac dev mode / first run).
  • **Generic rep counter** — per-exercise angle-threshold config selects
    which joint to track; a two-phase state machine (EXTENDED ↔ CONTRACTED)
    counts reps for any exercise without hardcoded curl logic.
  • VBT fatigue flag: current rep > 1.3 × rolling mean duration
  • JSON-serializable telemetry dict
"""

from __future__ import annotations

import json
import logging
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

logger = logging.getLogger(__name__)

# ── Lazy torch import — populated once by _load_classifier() ─────────
_torch_available = False
try:
    import torch
    import torch.nn as nn
    _torch_available = True
except ImportError:
    pass

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
# Exercise-specific rep counting configs
# ---------------------------------------------------------------------------
# ``primary_angles``: which JOINT_TRIPLETS angles to average for the phase
# detector.  ``contracted``: angle ≤ this → joint is contracted (peak effort
# for curls, bottom of squat, etc.).  ``extended``: angle ≥ this → joint is
# fully extended.  A full rep is EXTENDED → CONTRACTED → EXTENDED.

EXERCISE_REP_CONFIG: dict[str, dict] = {
    "bicep_curl": {
        "primary_angles": ["left_elbow", "right_elbow"],
        "contracted": 70.0,
        "extended": 110.0,
    },
    "squat": {
        "primary_angles": ["left_knee", "right_knee"],
        "contracted": 100.0,
        "extended": 160.0,
    },
    "pushup": {
        "primary_angles": ["left_elbow", "right_elbow"],
        "contracted": 90.0,
        "extended": 150.0,
    },
    "deadlift": {
        "primary_angles": ["left_hip", "right_hip"],
        "contracted": 100.0,
        "extended": 160.0,
    },
    "shoulder_press": {
        "primary_angles": ["left_elbow", "right_elbow"],
        "contracted": 90.0,
        "extended": 155.0,
    },
    "lunge": {
        "primary_angles": ["left_knee", "right_knee"],
        "contracted": 100.0,
        "extended": 155.0,
    },
}

DEFAULT_REP_CONFIG: dict = {
    "primary_angles": ["left_elbow", "right_elbow"],
    "contracted": 70.0,
    "extended": 130.0,
}

FATIGUE_MULTIPLIER: float = 1.3
REP_HISTORY_SIZE: int = 8

# ---------------------------------------------------------------------------
# MediaPipe model download
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
# PyTorch exercise classifier (mirrored architecture from training script)
# ---------------------------------------------------------------------------


if _torch_available:

    class ExerciseClassifier(nn.Module):
        """MLP with BatchNorm + Dropout — identical architecture to the
        training script so ``load_state_dict`` works without remapping."""

        def __init__(
            self,
            n_features: int = 99,
            n_classes: int = 6,
            hidden_dims: list[int] | None = None,
            dropout: float = 0.3,
        ) -> None:
            super().__init__()
            dims = hidden_dims or [256, 128, 64]
            layers: list[nn.Module] = []
            in_dim = n_features
            for h in dims:
                layers.extend([
                    nn.Linear(in_dim, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ])
                in_dim = h
            layers.append(nn.Linear(in_dim, n_classes))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


# ---------------------------------------------------------------------------
# Generic phase detector (replaces old CurlPosition state machine)
# ---------------------------------------------------------------------------


class Phase(Enum):
    UNKNOWN = auto()
    EXTENDED = auto()
    CONTRACTED = auto()


@dataclass
class RepState:
    phase: Phase = Phase.UNKNOWN
    rep_count: int = 0
    phase_start_time: float = 0.0
    concentric_duration: float = 0.0
    eccentric_duration: float = 0.0
    current_rep_duration: float = 0.0
    rep_durations: deque = field(
        default_factory=lambda: deque(maxlen=REP_HISTORY_SIZE)
    )
    neuromuscular_fatigue: bool = False
    detected_exercise: str = "unknown"
    exercise_confidence: float = 0.0


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def angle_between_3d_points(
    a: np.ndarray, b: np.ndarray, c: np.ndarray,
) -> float:
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
    """Real-time pose tracking with PyTorch exercise classification and
    generic angle-based rep counting."""

    def __init__(
        self,
        model_path: Optional[Path | str] = None,
        classifier_path: Optional[Path | str] = None,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        # ── MediaPipe PoseLandmarker ──────────────────────────────────
        path = Path(model_path) if model_path else ensure_pose_model_path()
        if not path.is_file():
            raise FileNotFoundError(f"Pose model not found: {path}")

        options = PoseLandmarkerOptions(
            base_options=base_options_lib.BaseOptions(
                model_asset_path=str(path),
            ),
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._video_ts_ms: int = 0

        # ── PyTorch exercise classifier (optional) ────────────────────
        self._classifier = None
        self._idx_to_label: dict[int, str] = {}
        self._norm_mean: Optional[np.ndarray] = None
        self._norm_std: Optional[np.ndarray] = None
        self._device = "cpu"
        self._load_classifier(classifier_path)

        # ── Rep tracking state ────────────────────────────────────────
        self.rep_state = RepState()
        self.latest_telemetry: Optional[dict] = None

    # -- classifier loading ------------------------------------------------

    def _load_classifier(self, override_path: Optional[Path | str]) -> None:
        """Attempt to load the exercise classifier checkpoint.

        Gracefully no-ops when torch is unavailable or the file is missing
        (Mac dev mode) — exercise defaults to ``"unknown"``.
        """
        if not _torch_available:
            return

        ckpt_path = Path(override_path) if override_path else (
            Path(__file__).resolve().parent / "models" / "exercise_classifier.pth"
        )
        if not ckpt_path.is_file():
            logger.info(
                "Classifier checkpoint not found at %s — "
                "exercise classification disabled.", ckpt_path,
            )
            return

        try:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            ckpt = torch.load(
                ckpt_path, map_location=self._device, weights_only=False,
            )
            model = ExerciseClassifier(
                n_features=ckpt["n_features"],
                n_classes=ckpt["n_classes"],
                hidden_dims=ckpt["hidden_dims"],
                dropout=ckpt["dropout"],
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(self._device)
            model.eval()

            self._classifier = model
            self._idx_to_label = ckpt["idx_to_label"]

            if "norm_mean" in ckpt:
                self._norm_mean = np.array(ckpt["norm_mean"], dtype=np.float32)
                self._norm_std = np.array(ckpt["norm_std"], dtype=np.float32)

            logger.info(
                "Exercise classifier loaded (%d classes, device=%s): %s",
                ckpt["n_classes"], self._device,
                list(self._idx_to_label.values()),
            )
        except Exception as exc:
            logger.warning("Classifier load failed: %s", exc)

    # -- public API --------------------------------------------------------

    def process_frame(
        self, frame: np.ndarray,
    ) -> tuple[np.ndarray, Optional[dict]]:
        """Run pose estimation + exercise classification on a BGR frame.

        Returns ``(annotated_bgr_frame, telemetry_dict_or_None)``.
        """
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        mp_image = mp_image_module.Image(
            image_format=ImageFormat.SRGB, data=rgb,
        )

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

        exercise, confidence = self._classify_exercise(world)
        self.rep_state.detected_exercise = exercise
        self.rep_state.exercise_confidence = confidence

        self._update_rep_state_machine(angles, exercise)

        telemetry = self._build_telemetry(angles)
        self.latest_telemetry = telemetry
        return frame, telemetry

    def reset(self) -> None:
        """Reset rep counter, fatigue state, and telemetry for a new session."""
        self.rep_state = RepState()
        self.latest_telemetry = None

    def close(self) -> None:
        self._landmarker.close()

    # -- exercise classification -------------------------------------------

    def _classify_exercise(
        self, world_landmarks: list,
    ) -> tuple[str, float]:
        """Run the PyTorch MLP on the 99 world-landmark features.

        Returns ``(exercise_name, confidence)`` or ``("unknown", 0.0)`` when
        the classifier is unavailable.
        """
        if self._classifier is None:
            return "bicep_curl", 0.0

        features = []
        for lm in world_landmarks:
            features.extend([
                float(lm.x or 0.0),
                float(lm.y or 0.0),
                float(lm.z or 0.0),
            ])

        feat_np = np.array(features, dtype=np.float32)

        if self._norm_mean is not None and self._norm_std is not None:
            feat_np = (feat_np - self._norm_mean) / self._norm_std

        tensor = torch.tensor(feat_np, dtype=torch.float32).unsqueeze(0).to(
            self._device,
        )

        with torch.no_grad():
            logits = self._classifier(tensor)
            probs = torch.softmax(logits, dim=-1)
            conf, idx = probs.max(dim=-1)

        label = self._idx_to_label.get(idx.item(), "unknown")
        return label, round(conf.item(), 3)

    # -- internals ---------------------------------------------------------

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

    def _update_rep_state_machine(
        self, angles: dict[str, float], exercise: str,
    ) -> None:
        """Generic two-phase rep counter.

        Selects the primary joint angles and thresholds from
        ``EXERCISE_REP_CONFIG`` based on the classified exercise, then runs
        a simple EXTENDED ↔ CONTRACTED state machine.

        A full rep = EXTENDED → CONTRACTED → EXTENDED.
        """
        config = EXERCISE_REP_CONFIG.get(exercise, DEFAULT_REP_CONFIG)
        primary_keys = config["primary_angles"]
        contracted_thresh = config["contracted"]
        extended_thresh = config["extended"]

        avg_angle = float(np.mean(
            [angles.get(k, 180.0) for k in primary_keys]
        ))

        now = time.monotonic()
        state = self.rep_state

        if state.phase == Phase.UNKNOWN:
            if avg_angle >= extended_thresh:
                state.phase = Phase.EXTENDED
            elif avg_angle <= contracted_thresh:
                state.phase = Phase.CONTRACTED
            state.phase_start_time = now
            return

        # EXTENDED → CONTRACTED (concentric)
        if (
            state.phase == Phase.EXTENDED
            and avg_angle <= contracted_thresh
        ):
            state.concentric_duration = now - state.phase_start_time
            state.phase = Phase.CONTRACTED
            state.phase_start_time = now

        # CONTRACTED → EXTENDED (eccentric → 1 full rep)
        elif (
            state.phase == Phase.CONTRACTED
            and avg_angle >= extended_thresh
        ):
            state.eccentric_duration = now - state.phase_start_time
            state.phase = Phase.EXTENDED
            state.phase_start_time = now

            state.current_rep_duration = (
                state.concentric_duration + state.eccentric_duration
            )
            state.rep_count += 1
            state.rep_durations.append(state.current_rep_duration)
            self._evaluate_fatigue()
            self._log_rep(angles)

    def _log_rep(self, angles: dict[str, float]) -> None:
        state = self.rep_state
        config = EXERCISE_REP_CONFIG.get(
            state.detected_exercise, DEFAULT_REP_CONFIG,
        )
        primary_keys = config["primary_angles"]
        primary_avg = round(float(np.mean(
            [angles.get(k, 0.0) for k in primary_keys]
        )), 1)

        payload = {
            "exercise": state.detected_exercise,
            "exercise_confidence": state.exercise_confidence,
            "rep": state.rep_count,
            "rep_time": round(state.current_rep_duration, 3),
            "concentric_sec": round(state.concentric_duration, 3),
            "eccentric_sec": round(state.eccentric_duration, 3),
            "avg_rep_time": round(
                float(np.mean(state.rep_durations)), 3,
            ),
            "fatigue": state.neuromuscular_fatigue,
            "primary_angle_avg": primary_avg,
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
            "exercise": state.detected_exercise,
            "exercise_confidence": round(state.exercise_confidence, 3),
            "phase": state.phase.name,
            "rep_count": state.rep_count,
            "angles_deg": angles,
            "concentric_sec": round(state.concentric_duration, 3),
            "eccentric_sec": round(state.eccentric_duration, 3),
            "current_rep_sec": round(state.current_rep_duration, 3),
            "avg_rep_sec": avg_dur,
            "neuromuscular_fatigue": state.neuromuscular_fatigue,
        }
