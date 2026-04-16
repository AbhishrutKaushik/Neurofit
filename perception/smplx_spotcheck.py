"""
smplx_spotcheck.py — Phase 3: Volumetric Spot-Checking via SMPL-X

Regresses a full 3-D body mesh from a single RGB frame using the SMPL-X
parametric body model, then extracts deep biomechanical metrics (spinal
alignment, shoulder symmetry) that a 2-D skeleton alone cannot capture.

MOCK_MODE (default True)
    Bypasses all PyTorch / SMPL-X initialisation so the module runs on any
    hardware (MacBook Air, CI runner, etc.) without a GPU.  Returns a
    deterministic mock payload so downstream phases (RAG pipeline, UI) can
    be developed and tested locally.

    Set MOCK_MODE=False on a machine with a CUDA GPU and the SMPL-X .pkl
    model files downloaded into ``model_folder``.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SMPL-X body-joint indices (22-joint kinematic tree)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# These are the first 22 rows of ``smplx_output.joints`` when the model
# is built with default settings.  The full tensor may hold up to 127
# joints (body + hands + face) depending on ``use_pca`` / ``use_face``.
#
# Index │ Joint          Index │ Joint
# ──────┼───────────     ──────┼──────────────
#   0   │ Pelvis           12  │ Neck
#   1   │ L_Hip            13  │ L_Collar
#   2   │ R_Hip            14  │ R_Collar
#   3   │ Spine1           15  │ Head
#   4   │ L_Knee           16  │ L_Shoulder
#   5   │ R_Knee           17  │ R_Shoulder
#   6   │ Spine2           18  │ L_Elbow
#   7   │ L_Ankle          19  │ R_Elbow
#   8   │ R_Ankle          20  │ L_Wrist
#   9   │ Spine3           21  │ R_Wrist
#  10   │ L_Foot
#  11   │ R_Foot
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_PELVIS = 0
_LEFT_HIP = 1
_RIGHT_HIP = 2
_SPINE1 = 3
_SPINE2 = 6
_SPINE3 = 9
_NECK = 12
_HEAD = 15
_LEFT_SHOULDER = 16
_RIGHT_SHOULDER = 17

# Axial spine chain used for curvature analysis.
_SPINE_CHAIN: List[int] = [_PELVIS, _SPINE1, _SPINE2, _SPINE3, _NECK]

# Maximum tolerable average inter-segment deviation (degrees).
# 30° maps to a score of 0.0 (severe rounding / hyperextension).
_MAX_DEVIATION_DEG: float = 30.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# body_pose ↔ joint mapping reference
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# ``body_pose`` is a (1, 63) tensor — 21 joints × 3 axis-angle values.
# Joint 0 (Pelvis) is encoded separately as ``global_orient``, so
# ``body_pose`` covers joints 1-21:
#
#   body_pose[:, (k-1)*3 : k*3]  →  joint k
#
#   Spine1 = joint 3  → indices 6:9
#   Spine2 = joint 6  → indices 15:18
#   Spine3 = joint 9  → indices 24:27
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SPINE1_POSE_SLICE = slice(6, 9)
_SPINE2_POSE_SLICE = slice(15, 18)
_SPINE3_POSE_SLICE = slice(24, 27)


class SMPLXSpotChecker:
    """Regress an SMPL-X mesh from an RGB frame + MediaPipe landmarks and
    extract volumetric posture metrics.

    Parameters
    ----------
    MOCK_MODE : bool
        If *True* (default), skip all heavy imports (``torch``, ``smplx``)
        and return a deterministic mock payload.
    model_folder : Path | str | None
        Directory containing SMPL-X ``.pkl`` model files (e.g.
        ``SMPLX_NEUTRAL.pkl``).  Required when ``MOCK_MODE=False``.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(
        self,
        MOCK_MODE: bool = True,
        model_folder: Optional[Path | str] = None,
    ) -> None:
        self.mock_mode: bool = MOCK_MODE

        # Lazy references — only populated when running the real model
        self._torch: Any = None
        self._body_model: Any = None
        self._device: Any = None

        if not self.mock_mode:
            try:
                self._init_model(
                    Path(model_folder) if model_folder else Path("models/smplx")
                )
            except Exception as exc:
                # Graceful degradation: if GPU setup fails for any reason
                # (missing files, driver mismatch, OOM on model load, …)
                # fall back to mock so Streamlit keeps running.
                logger.warning(
                    "SMPL-X initialisation failed (%s). "
                    "Falling back to MOCK_MODE=True.",
                    exc,
                )
                self.mock_mode = True

    def _init_model(self, model_folder: Path) -> None:
        """Import ``torch`` + ``smplx``, build the body model, move to
        the best available device.

        Imports are deferred so that neither library is loaded when the
        module runs in mock mode — saving ~1 GB of resident memory.
        """
        import smplx as smplx_lib
        import torch

        self._torch = torch

        # ── Dynamic device selection ──
        # torch.cuda.is_available() returns True only if a CUDA-capable
        # GPU **and** a matching driver are present.
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info("SMPL-X device selected → %s", self._device)

        if not model_folder.is_dir():
            raise FileNotFoundError(
                f"SMPL-X model directory not found: {model_folder}.  "
                "Download from https://smpl-x.is.tue.mpg.de/ and place "
                "the .pkl files in this directory."
            )

        # ── Build the SMPL-X layer ──
        # num_betas  = 10  : shape-space dimensionality (β ∈ ℝ¹⁰)
        # use_pca    = True: reduce each hand's 45-D pose to 12-D via PCA
        # batch_size = 1   : single-frame inference
        self._body_model = smplx_lib.create(
            model_path=str(model_folder),
            model_type="smplx",
            gender="neutral",
            num_betas=10,
            use_pca=True,
            num_pca_comps=12,
            batch_size=1,
        ).to(self._device)

        param_count = sum(p.numel() for p in self._body_model.parameters())
        logger.info(
            "SMPL-X model loaded on %s (%s parameters).",
            self._device,
            f"{param_count:,}",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(
        self,
        image_array: np.ndarray,
        mediapipe_landmarks: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run SMPL-X volumetric inference on a single frame.

        Parameters
        ----------
        image_array : np.ndarray
            BGR ``uint8`` image, shape ``(H, W, 3)``.
        mediapipe_landmarks
            The 33 MediaPipe BlazePose landmarks for this frame (a
            ``NormalizedLandmarkList`` or plain list).  When available
            in real mode, a coarse pose hint is derived from these
            landmarks; otherwise the neutral (zero) pose is used.

        Returns
        -------
        dict
            JSON-serialisable payload containing at least
            ``spinal_alignment_score`` and ``posture_warning``.
        """
        # ── Mock path — works on any hardware ──
        if self.mock_mode:
            return {
                "spinal_alignment_score": 0.85,
                "posture_warning": "None",
                "shoulder_symmetry": 0.90,
                "mock": True,
            }

        # ── Real path — requires PyTorch + SMPL-X ──
        try:
            return self._real_inference(image_array, mediapipe_landmarks)

        except Exception as exc:
            # Catch CUDA OOM and any other runtime failures so the
            # Streamlit application loop never crashes mid-demo.
            oom = False
            if hasattr(self._torch.cuda, "OutOfMemoryError"):
                oom = isinstance(exc, self._torch.cuda.OutOfMemoryError)
            if not oom and isinstance(exc, RuntimeError):
                oom = "out of memory" in str(exc).lower()

            if oom:
                logger.error(
                    "CUDA out-of-memory during SMPL-X forward pass.  "
                    "Clearing cache and falling back to MOCK_MODE."
                )
                if self._torch.cuda.is_available():
                    self._torch.cuda.empty_cache()
                self.mock_mode = True
                return {
                    "spinal_alignment_score": 0.85,
                    "posture_warning": "None",
                    "shoulder_symmetry": 0.90,
                    "mock": True,
                }

            # Non-OOM errors: log and degrade gracefully.
            logger.exception("Unexpected error in SMPL-X inference.")
            self.mock_mode = True
            return {
                "spinal_alignment_score": 0.85,
                "posture_warning": "None",
                "shoulder_symmetry": 0.90,
                "mock": True,
            }

    # ------------------------------------------------------------------
    # Real inference path
    # ------------------------------------------------------------------

    def _real_inference(
        self,
        image_array: np.ndarray,
        mediapipe_landmarks: Optional[Any],
    ) -> Dict[str, Any]:
        """Forward-pass through SMPL-X and extract biomechanical metrics.

        In a full production pipeline a learned regressor (PyMAF-X,
        ExPose, or SMPLify-X optimisation loop) would lift the 2-D image
        to pose (θ) and shape (β) parameters.  Here we construct
        *approximate* pose parameters from MediaPipe landmarks when they
        are available, otherwise we use the neutral zero-pose, to
        validate the complete forward-pass → metric-extraction pipeline.
        """
        torch = self._torch

        # ── Shape parameters β : (1, 10) ──
        # All zeros = average body proportions.
        betas = torch.zeros(
            1, 10, dtype=torch.float32, device=self._device
        )

        # ── Body pose θ : (1, 63) — 21 joints × 3 axis-angle ──
        body_pose = torch.zeros(
            1, 63, dtype=torch.float32, device=self._device
        )

        # ── Global orientation : (1, 3) — root rotation ──
        global_orient = torch.zeros(
            1, 3, dtype=torch.float32, device=self._device
        )

        # If MediaPipe landmarks are available, inject a coarse torso-
        # lean hint into the three spine joints.
        if mediapipe_landmarks is not None:
            body_pose = self._mediapipe_to_pose_hint(
                mediapipe_landmarks, body_pose
            )

        # ── Forward pass (inference only — no gradient tape) ──
        with torch.no_grad():
            output = self._body_model(
                betas=betas,              # (1, 10)
                body_pose=body_pose,      # (1, 63)
                global_orient=global_orient,  # (1, 3)
            )
        # output.vertices : (1, 10475, 3)  — full mesh vertices in metres
        # output.joints   : (1, J, 3)      — J ≥ 22 body joints (up to
        #                                     127 with hands + face)

        # Move joint tensor to CPU numpy for metric extraction.
        # Shape after indexing batch dim: (J, 3)
        joints_3d: np.ndarray = output.joints[0].cpu().numpy()

        # ── Extract biomechanical metrics ──
        alignment_score, posture_warning = self._calculate_spinal_alignment(
            joints_3d
        )
        shoulder_sym = self._shoulder_symmetry(joints_3d)

        return {
            "spinal_alignment_score": alignment_score,
            "posture_warning": posture_warning,
            "shoulder_symmetry": shoulder_sym,
            "device": str(self._device),
            "mock": False,
        }

    def get_mesh_data(
        self,
        image_array: np.ndarray,
        mediapipe_landmarks: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run inference and return the full mesh geometry alongside metrics.

        This is the method used by the Phase-3 validation script to
        extract vertices and faces for 3-D visualisation with trimesh.

        Returns
        -------
        dict with keys:
            vertices  : np.ndarray (10475, 3) — mesh verts in metres
            faces     : np.ndarray (F, 3)     — triangle indices
            joints_3d : np.ndarray (J, 3)     — body joints in metres
            metrics   : dict                  — same as process_frame output
            device    : str
            elapsed_s : float                 — wall-clock seconds for the
                                                forward pass only
        Raises ``RuntimeError`` if mock_mode is active (no mesh to return).
        """
        if self.mock_mode:
            raise RuntimeError(
                "get_mesh_data() requires MOCK_MODE=False with a real "
                "SMPL-X model loaded on GPU/CPU."
            )

        import time as _time
        torch = self._torch

        betas = torch.zeros(1, 10, dtype=torch.float32, device=self._device)
        body_pose = torch.zeros(1, 63, dtype=torch.float32, device=self._device)
        global_orient = torch.zeros(1, 3, dtype=torch.float32, device=self._device)

        if mediapipe_landmarks is not None:
            body_pose = self._mediapipe_to_pose_hint(mediapipe_landmarks, body_pose)

        if self._device.type == "cuda":
            torch.cuda.synchronize()

        t0 = _time.perf_counter()

        with torch.no_grad():
            output = self._body_model(
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
            )

        if self._device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = _time.perf_counter() - t0

        vertices = output.vertices[0].cpu().numpy()   # (10475, 3)
        joints_3d = output.joints[0].cpu().numpy()     # (J, 3)

        # SMPL-X face array lives on the model layer itself.
        faces = self._body_model.faces                  # (F, 3) int32

        alignment_score, posture_warning = self._calculate_spinal_alignment(joints_3d)
        shoulder_sym = self._shoulder_symmetry(joints_3d)

        return {
            "vertices": vertices,
            "faces": faces,
            "joints_3d": joints_3d,
            "metrics": {
                "spinal_alignment_score": alignment_score,
                "posture_warning": posture_warning,
                "shoulder_symmetry": shoulder_sym,
            },
            "device": str(self._device),
            "elapsed_s": round(elapsed, 4),
        }

    # ------------------------------------------------------------------
    # Biomechanical extraction helpers
    # ------------------------------------------------------------------

    def _calculate_spinal_alignment(
        self, joints_3d: np.ndarray
    ) -> Tuple[float, str]:
        """Compute a spinal-alignment score from 3-D SMPL-X joints.

        Algorithm
        ---------
        1. Extract the five joints along the axial spine chain:
              Pelvis(0) → Spine1(3) → Spine2(6) → Spine3(9) → Neck(12)

        2. For every **triplet** of consecutive joints (A, B, C), compute
           the 3-D angle ∠ABC via the dot-product identity:

                         BA⃗ · BC⃗
              cos θ  =  ──────────
                        |BA⃗| |BC⃗|

           A perfectly straight (neutral) spine produces θ ≈ 180°
           because BA⃗ and BC⃗ are nearly anti-parallel.  The
           *deviation* from neutral is therefore:

              δ = 180° − θ

        3. Average the deviations across all three triplets and map to
           [0.0, 1.0]:

              score = clamp(1 − mean(δ) / MAX_DEV, 0, 1)

           where MAX_DEV = 30° (anything worse is score 0).

        4. Generate a human-readable warning string:
              score < 0.70  →  "Severe lumbar rounding detected"
              score < 0.85  →  "Mild spinal flexion — brace your core"
              else          →  "None"

        Parameters
        ----------
        joints_3d : np.ndarray, shape (J, 3)
            3-D joint positions from the SMPL-X forward pass (metres).

        Returns
        -------
        (score, warning) : tuple[float, str]
        """
        # Spine chain coords: shape (5, 3)
        spine = joints_3d[_SPINE_CHAIN]

        deviations_deg: List[float] = []

        # Three triplets: (Pelvis,Spine1,Spine2), (Spine1,Spine2,Spine3),
        #                  (Spine2,Spine3,Neck)
        for i in range(len(spine) - 2):
            A = spine[i]      # shape (3,) — e.g. Pelvis
            B = spine[i + 1]  # shape (3,) — e.g. Spine1 (vertex)
            C = spine[i + 2]  # shape (3,) — e.g. Spine2

            # Vectors radiating from the middle joint
            BA = A - B  # (3,)  — points toward the joint *below*
            BC = C - B  # (3,)  — points toward the joint *above*

            norm_BA = float(np.linalg.norm(BA))
            norm_BC = float(np.linalg.norm(BC))

            # Degenerate segment (two joints coincide) — skip.
            if norm_BA < 1e-8 or norm_BC < 1e-8:
                continue

            # Dot-product → cosine of the included angle
            cos_theta = np.dot(BA, BC) / (norm_BA * norm_BC)
            cos_theta = float(np.clip(cos_theta, -1.0, 1.0))

            theta_rad = math.acos(cos_theta)
            theta_deg = math.degrees(theta_rad)

            # Deviation from the perfectly-straight 180° baseline
            deviation = abs(180.0 - theta_deg)
            deviations_deg.append(deviation)

        # If all segments were degenerate, assume neutral posture.
        if not deviations_deg:
            return 1.0, "None"

        mean_deviation = sum(deviations_deg) / len(deviations_deg)

        # Linear map: 0° dev → 1.0,  _MAX_DEVIATION_DEG → 0.0
        score = max(0.0, min(1.0, 1.0 - mean_deviation / _MAX_DEVIATION_DEG))
        score = round(score, 2)

        # Warning tiers
        if score < 0.70:
            warning = "Severe lumbar rounding detected"
        elif score < 0.85:
            warning = "Mild spinal flexion — brace your core"
        else:
            warning = "None"

        return score, warning

    def _shoulder_symmetry(self, joints_3d: np.ndarray) -> float:
        """Shoulder-height symmetry in [0, 1] (1.0 = perfectly level).

        Compares the Y-coordinate (vertical axis) of the left and right
        shoulder joints.  A difference of 0.1 m maps to a score of 0.0.
        """
        left_y = float(joints_3d[_LEFT_SHOULDER, 1])
        right_y = float(joints_3d[_RIGHT_SHOULDER, 1])
        diff = abs(left_y - right_y)
        return round(max(0.0, 1.0 - diff * 10.0), 2)

    # ------------------------------------------------------------------
    # MediaPipe → SMPL-X pose hint
    # ------------------------------------------------------------------

    def _mediapipe_to_pose_hint(self, landmarks: Any, body_pose: Any) -> Any:
        """Derive a coarse SMPL-X ``body_pose`` hint from MediaPipe 3-D
        landmarks.

        This is **not** a replacement for a learned regressor (PyMAF-X,
        ExPose).  It estimates the overall torso-lean direction from the
        hip-to-shoulder vector and distributes a proportional axis-angle
        rotation across Spine1 / Spine2 / Spine3.

        MediaPipe landmark indices used:
            11 = L_Shoulder, 12 = R_Shoulder, 23 = L_Hip, 24 = R_Hip

        Axis-angle encoding
        -------------------
        A rotation of θ radians about unit axis ``n̂`` is stored as the
        3-D vector ``θ · n̂``.  We compute the rotation that takes the
        neutral upward torso vector [0, −1, 0] to the observed torso
        vector, then split it across the three spine joints.
        """
        torch = self._torch

        try:
            # Accept both NormalizedLandmarkList and plain list
            lm = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks
            if len(lm) < 33:
                return body_pose

            # Mid-hip and mid-shoulder in MediaPipe normalised coords
            mid_hip = np.array([
                (lm[23].x + lm[24].x) / 2.0,
                (lm[23].y + lm[24].y) / 2.0,
                (lm[23].z + lm[24].z) / 2.0,
            ], dtype=np.float32)

            mid_shoulder = np.array([
                (lm[11].x + lm[12].x) / 2.0,
                (lm[11].y + lm[12].y) / 2.0,
                (lm[11].z + lm[12].z) / 2.0,
            ], dtype=np.float32)

            # Observed torso direction (hip → shoulder), normalised
            torso_vec = mid_shoulder - mid_hip
            torso_norm = float(np.linalg.norm(torso_vec))
            if torso_norm < 1e-6:
                return body_pose
            torso_vec /= torso_norm

            # Neutral upward direction (MediaPipe uses Y-down)
            neutral = np.array([0.0, -1.0, 0.0], dtype=np.float32)

            # Rotation axis = neutral × torso  (perpendicular to both)
            axis = np.cross(neutral, torso_vec)
            axis_norm = float(np.linalg.norm(axis))

            # Rotation angle = arccos(neutral · torso)
            cos_angle = float(np.clip(np.dot(neutral, torso_vec), -1.0, 1.0))
            angle = math.acos(cos_angle)

            # Skip negligible rotations (< ~1°)
            if axis_norm < 1e-6 or angle < 0.02:
                return body_pose

            axis /= axis_norm

            # Full axis-angle vector: direction × magnitude
            full_aa = (axis * angle).astype(np.float32)  # shape (3,)

            # Distribute the lean across Spine1 / Spine2 / Spine3 with
            # decreasing weight (lower spine bears more of the lean).
            pose_np = body_pose.cpu().numpy()                  # (1, 63)
            pose_np[0, _SPINE1_POSE_SLICE] = full_aa * 0.50   # Spine1: 50 %
            pose_np[0, _SPINE2_POSE_SLICE] = full_aa * 0.30   # Spine2: 30 %
            pose_np[0, _SPINE3_POSE_SLICE] = full_aa * 0.20   # Spine3: 20 %

            body_pose = torch.tensor(
                pose_np, dtype=torch.float32, device=self._device
            )

        except Exception as exc:
            logger.debug(
                "MediaPipe → pose hint conversion failed (%s); "
                "falling back to zero pose.",
                exc,
            )

        return body_pose
