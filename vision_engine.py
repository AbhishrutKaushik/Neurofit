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
import os
import ssl
import statistics
import time
import urllib.request
import warnings
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from pathlib import Path
from typing import Optional

os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
warnings.filterwarnings("ignore", category=UserWarning)

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
        "contracted": 85.0,
        "extended": 110.0,
    },
    "squat": {
        "primary_angles": ["left_knee", "right_knee"],
        "contracted": 120.0,
        "extended": 145.0,
    },
    "pushup": {
        "primary_angles": ["left_elbow", "right_elbow"],
        "contracted": 110.0,
        "extended": 145.0,
    },
    "deadlift": {
        "primary_angles": ["left_hip", "right_hip"],
        "contracted": 125.0,
        "extended": 150.0,
    },
    "shoulder_press": {
        "primary_angles": ["left_elbow", "right_elbow"],
        "contracted": 110.0,
        "extended": 145.0,
    },
    "lunge": {
        "primary_angles": ["left_knee", "right_knee"],
        "contracted": 120.0,
        "extended": 148.0,
    },
    "lateral_raise": {
        "primary_angles": ["left_shoulder", "right_shoulder"],
        "contracted": 50.0,
        "extended": 38.0,
        "inverted": True,
    },
    "front_raise": {
        "primary_angles": ["left_shoulder", "right_shoulder"],
        "contracted": 50.0,
        "extended": 38.0,
        "inverted": True,
    },
    "tricep_dip": {
        "primary_angles": ["left_elbow", "right_elbow"],
        "contracted": 105.0,
        "extended": 140.0,
    },
    "bent_over_row": {
        "primary_angles": ["left_elbow", "right_elbow"],
        "contracted": 95.0,
        "extended": 130.0,
    },
    "leg_press": {
        "primary_angles": ["left_knee", "right_knee"],
        "contracted": 115.0,
        "extended": 148.0,
    },
    "russian_twist": {
        "primary_angles": ["left_hip", "right_hip"],
        "contracted": 75.0,
        "extended": 88.0,
    },
}

DEFAULT_REP_CONFIG: dict = {
    "primary_angles": ["left_elbow", "right_elbow"],
    "contracted": 90.0,
    "extended": 125.0,
}

# ---------------------------------------------------------------------------
# Classifier label → rep config key translation
# ---------------------------------------------------------------------------
# The PyTorch classifier outputs labels derived from the dataset folder names
# (e.g. "push-up", "barbell biceps curl").  EXERCISE_REP_CONFIG uses short
# keys (e.g. "pushup", "bicep_curl").  This dict bridges the gap so the rep
# state machine always finds the right angle thresholds.

CLASSIFIER_LABEL_TO_REP_KEY: dict[str, str] = {
    "barbell biceps curl": "bicep_curl",
    "hammer curl": "bicep_curl",
    "bench press": "pushup",
    "decline bench press": "pushup",
    "incline bench press": "pushup",
    "chest fly machine": "pushup",
    "push-up": "pushup",
    "deadlift": "deadlift",
    "romanian deadlift": "deadlift",
    "squat": "squat",
    "hip thrust": "deadlift",
    "lateral raise": "lateral_raise",
    "shoulder press": "shoulder_press",
    "lat pulldown": "bent_over_row",
    "t bar row": "bent_over_row",
    "pull Up": "bent_over_row",
    "tricep dips": "tricep_dip",
    "tricep Pushdown": "tricep_dip",
    "leg extension": "leg_press",
    "leg raises": "russian_twist",
    "russian twist": "russian_twist",
    "plank": "pushup",
}

FATIGUE_MULTIPLIER: float = 1.5
FATIGUE_WARMUP_REPS: int = 5
REP_HISTORY_SIZE: int = 20

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
    baseline_avg_duration: float = 0.0


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


def _print_exercise_lock(exercise: str, *, auto: bool = False) -> None:
    """Bold terminal banner when the active exercise changes."""
    tag = "AUTO-DETECT" if auto else "LOCKED"
    print(
        f"\n\033[1m\033[96m🎯 [{tag}]: "
        f"{exercise.upper().replace('_', ' ')}\033[0m\n",
        flush=True,
    )


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
        self._frame_counter: int = 0
        self._pred_buffer: deque[str] = deque(maxlen=15)
        self._last_locked_exercise: str = ""
        self._active_rep_key: str = ""

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
        self,
        frame: np.ndarray,
        *,
        tracking_active: bool = True,
        locked_exercise: str = "",
        classify_stride: int = 5,
        allowed_labels: Optional[set[str]] = None,
    ) -> tuple[np.ndarray, Optional[dict]]:
        """Run pose estimation on a BGR frame, optionally with full tracking.

        Parameters
        ----------
        tracking_active:
            ``False`` = setup mode — skeleton drawn, but classifier / rep
            counter / fatigue evaluator are all skipped.
        locked_exercise:
            When non-empty, the rep counter uses this ``EXERCISE_REP_CONFIG``
            key exclusively and the PyTorch classifier is skipped entirely.
            This prevents random exercise guesses during an active set.
        classify_stride:
            When ``locked_exercise`` is empty, the PyTorch classifier only
            runs every *N*-th frame.  Intermediate frames reuse the last
            prediction, keeping the WebRTC callback fast (~5 ms vs ~25 ms).

        Returns ``(annotated_bgr_frame, telemetry_dict_or_None)``.
        """
        self._frame_counter += 1

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

        if tracking_active:
            if locked_exercise:
                if locked_exercise != self._last_locked_exercise:
                    self._last_locked_exercise = locked_exercise
                    self._pred_buffer.clear()
                    _print_exercise_lock(locked_exercise)
                self.rep_state.detected_exercise = locked_exercise
                self.rep_state.exercise_confidence = 1.0
            elif self._frame_counter % classify_stride == 0:
                exercise, confidence = self._classify_exercise(
                    world, allowed_labels=allowed_labels,
                )
                self._pred_buffer.append(exercise)
                try:
                    smoothed = statistics.mode(self._pred_buffer)
                except statistics.StatisticsError:
                    smoothed = exercise
                if smoothed != self.rep_state.detected_exercise:
                    _print_exercise_lock(smoothed, auto=True)
                self.rep_state.detected_exercise = smoothed
                self.rep_state.exercise_confidence = confidence

            self._update_rep_state_machine(
                angles, self.rep_state.detected_exercise,
            )

        telemetry = self._build_telemetry(angles)
        self.latest_telemetry = telemetry
        return frame, telemetry

    def reset(self) -> None:
        """Reset rep counter, fatigue state, and telemetry for a new session."""
        self.rep_state = RepState()
        self.latest_telemetry = None
        self._frame_counter = 0
        self._pred_buffer.clear()
        self._last_locked_exercise = ""
        self._active_rep_key = ""

    def close(self) -> None:
        self._landmarker.close()

    # -- exercise classification -------------------------------------------

    def _classify_exercise(
        self, world_landmarks: list,
        *, allowed_labels: Optional[set[str]] = None,
    ) -> tuple[str, float]:
        """Run the PyTorch MLP on the 99 world-landmark features.

        When *allowed_labels* is provided, logits for classes outside the set
        are masked to ``-inf`` before softmax so the prediction is constrained
        to the user's selected muscle group.

        Returns ``(exercise_name, confidence)`` or ``("unknown", 0.0)`` when
        the classifier is unavailable.
        """
        if self._classifier is None:
            return "unknown", 0.0

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
            if allowed_labels:
                for i in range(logits.shape[1]):
                    if self._idx_to_label.get(i, "") not in allowed_labels:
                        logits[0, i] = float("-inf")
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

        Uses ``min()`` (standard) or ``max()`` (inverted) of the primary
        joint angles rather than the mean.  This prevents one poorly-tracked
        side from dragging the value into the dead-zone between thresholds
        where the state machine would stall.

        A full rep = EXTENDED → CONTRACTED → EXTENDED.
        """
        rep_key = CLASSIFIER_LABEL_TO_REP_KEY.get(exercise, exercise)
        config = EXERCISE_REP_CONFIG.get(rep_key, DEFAULT_REP_CONFIG)
        primary_keys = config["primary_angles"]
        contracted_thresh = config["contracted"]
        extended_thresh = config["extended"]
        inverted = config.get("inverted", False)

        raw_angles = [angles.get(k, 180.0) for k in primary_keys]

        if inverted:
            driving_angle = float(max(raw_angles))
        else:
            driving_angle = float(min(raw_angles))

        is_extended = (
            (driving_angle <= extended_thresh) if inverted
            else (driving_angle >= extended_thresh)
        )
        is_contracted = (
            (driving_angle >= contracted_thresh) if inverted
            else (driving_angle <= contracted_thresh)
        )

        now = time.monotonic()
        state = self.rep_state

        if rep_key != getattr(self, "_active_rep_key", ""):
            self._active_rep_key = rep_key
            state.phase = Phase.UNKNOWN

        if self._frame_counter % 90 == 0:
            print(
                f"[STATE] exercise={exercise}  rep_key={rep_key}  "
                f"angle={driving_angle:.1f}  "
                f"thresh=[{contracted_thresh}, {extended_thresh}]  "
                f"phase={state.phase.name}  reps={state.rep_count}",
                flush=True,
            )

        if state.phase == Phase.UNKNOWN:
            if is_extended:
                state.phase = Phase.EXTENDED
            elif is_contracted:
                state.phase = Phase.CONTRACTED
            state.phase_start_time = now
            return

        if state.phase == Phase.EXTENDED and is_contracted:
            state.concentric_duration = now - state.phase_start_time
            state.phase = Phase.CONTRACTED
            state.phase_start_time = now

        elif state.phase == Phase.CONTRACTED and is_extended:
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

        payload = {
            "exercise": state.detected_exercise,
            "exercise_confidence": state.exercise_confidence,
            "rep": state.rep_count,
            "angles_deg": {k: round(v, 1) for k, v in angles.items()},
            "rep_time_sec": round(state.current_rep_duration, 3),
            "concentric_sec": round(state.concentric_duration, 3),
            "eccentric_sec": round(state.eccentric_duration, 3),
            "avg_rep_time_sec": round(
                float(np.mean(state.rep_durations)), 3,
            ),
            "baseline_avg_sec": round(state.baseline_avg_duration, 3),
            "fatigue_threshold_sec": round(
                FATIGUE_MULTIPLIER * state.baseline_avg_duration, 3,
            ) if state.baseline_avg_duration > 0 else None,
            "neuromuscular_fatigue": state.neuromuscular_fatigue,
        }

        b, r = "\033[1m", "\033[0m"
        warmup = state.rep_count <= FATIGUE_WARMUP_REPS
        tag = " (warmup)" if warmup else ""
        print(f"\n{b}── REP {state.rep_count}{tag} ──{r}", flush=True)
        print(json.dumps(payload, indent=2), flush=True)
        if warmup and state.rep_count == FATIGUE_WARMUP_REPS:
            print(
                f"\033[92m{b}✓ Warmup complete — baseline locked at "
                f"{state.baseline_avg_duration:.3f}s  "
                f"(fatigue threshold: "
                f"{FATIGUE_MULTIPLIER * state.baseline_avg_duration:.3f}s){r}",
                flush=True,
            )
        if state.neuromuscular_fatigue:
            print(
                f"\033[91m{b}>>> NEUROMUSCULAR FATIGUE DETECTED{r}",
                flush=True,
            )

    def _evaluate_fatigue(self) -> None:
        """Neuromuscular fatigue detection with a warmup grace period.

        Reps 1-``FATIGUE_WARMUP_REPS`` are *never* flagged.  After the warmup
        the average duration of those first reps is locked as the baseline.
        Subsequent reps are flagged when they exceed
        ``FATIGUE_MULTIPLIER × baseline`` (50 % slower than the warmup pace).
        """
        state = self.rep_state

        if state.rep_count <= FATIGUE_WARMUP_REPS:
            state.neuromuscular_fatigue = False
            if state.rep_count == FATIGUE_WARMUP_REPS:
                state.baseline_avg_duration = float(np.mean(
                    list(state.rep_durations)[:FATIGUE_WARMUP_REPS]
                ))
            return

        if state.baseline_avg_duration <= 0:
            state.neuromuscular_fatigue = False
            return

        state.neuromuscular_fatigue = (
            state.current_rep_duration
            > FATIGUE_MULTIPLIER * state.baseline_avg_duration
        )

    def _build_telemetry(self, angles: dict[str, float]) -> dict:
        state = self.rep_state
        avg_dur = (
            round(float(np.mean(state.rep_durations)), 3)
            if state.rep_durations
            else 0.0
        )
        baseline = round(state.baseline_avg_duration, 3)
        threshold = (
            round(FATIGUE_MULTIPLIER * state.baseline_avg_duration, 3)
            if state.baseline_avg_duration > 0
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
            "baseline_avg_sec": baseline,
            "fatigue_threshold_sec": threshold,
            "neuromuscular_fatigue": state.neuromuscular_fatigue,
        }
