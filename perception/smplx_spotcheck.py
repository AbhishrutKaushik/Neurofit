"""
smplx_spotcheck.py — Phase 3: Volumetric Spot-Checking via SMPL-X

Triggers periodically (every N seconds) or on a fatigue flag to regress a
full 3D body mesh from a single RGB frame.  Extracts deep biomechanical
metrics (spinal alignment, joint torque) that a 2-D skeleton cannot capture.

MOCK_MODE (default True)
  Bypasses all PyTorch / SMPL-X initialisation so the module runs instantly
  on a MacBook Air M3 with 8 GB RAM.  Returns a plausible mock payload so
  downstream phases (RAG pipeline, UI) can be developed and tested locally.

  Set MOCK_MODE=False when running on a machine with a CUDA GPU and the
  SMPL-X model files downloaded to `model_dir`.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np


class SMPLXSpotChecker:
    """Regress SMPL-X mesh from an RGB frame and extract posture metrics.

    Parameters
    ----------
    mock_mode : bool
        If True (default), skip all heavy imports and return mock data.
    model_dir : Path | str | None
        Directory containing the SMPL-X `.npz` model files.
        Only required when ``mock_mode=False``.
    trigger_interval_sec : float
        Minimum seconds between two spot-checks (rate limiter).
    """

    def __init__(
        self,
        mock_mode: bool = True,
        model_dir: Optional[Path | str] = None,
        trigger_interval_sec: float = 5.0,
    ) -> None:
        self.mock_mode = mock_mode
        self.trigger_interval = trigger_interval_sec
        self._last_trigger: float = 0.0

        if not mock_mode:
            self._init_real_model(Path(model_dir) if model_dir else Path("models/smplx"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_trigger(self, fatigue_flag: bool = False) -> bool:
        """Return True if enough time has elapsed OR a fatigue flag is set."""
        now = time.monotonic()
        elapsed = now - self._last_trigger
        if fatigue_flag or elapsed >= self.trigger_interval:
            self._last_trigger = now
            return True
        return False

    def process_frame(self, image_bgr: np.ndarray) -> dict:
        """Run SMPL-X regression on a single BGR frame.

        Returns a JSON-compatible dict with volumetric posture metrics.
        """
        if self.mock_mode:
            return self._mock_inference()
        return self._real_inference(image_bgr)

    # ------------------------------------------------------------------
    # Mock path — runs on any hardware
    # ------------------------------------------------------------------

    def _mock_inference(self) -> dict:
        """Return a realistic-looking mock payload without any GPU work.

        Adds small random jitter so downstream code sees variance across
        successive calls (useful for testing alert thresholds).
        """
        alignment = round(random.uniform(0.72, 0.98), 2)

        if alignment < 0.78:
            warning = "Mild lumbar flexion detected — brace core"
        elif alignment < 0.82:
            warning = "Slight anterior pelvic tilt"
        else:
            warning = "None"

        payload = {
            "spinal_alignment_score": alignment,
            "posture_warning": warning,
            "shoulder_symmetry": round(random.uniform(0.80, 1.00), 2),
            "hip_torque_ratio": round(random.uniform(0.85, 1.05), 2),
            "mock": True,
        }
        return payload

    # ------------------------------------------------------------------
    # Real path — requires CUDA + SMPL-X model files
    # ------------------------------------------------------------------

    def _init_real_model(self, model_dir: Path) -> None:
        """Initialise PyTorch and SMPL-X body model (GPU-only path).

        Import torch / smplx lazily so they never load in mock mode.
        """
        import smplx as smplx_lib
        import torch

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"SMPL-X model directory not found: {model_dir}. "
                "Download from https://smpl-x.is.tue.mpg.de/ "
                "and place .npz files in this directory."
            )

        self.body_model = smplx_lib.create(
            model_path=str(model_dir),
            model_type="smplx",
            gender="neutral",
            num_betas=10,
            use_pca=True,
            num_pca_comps=12,
            batch_size=1,
        ).to(self.device)

        self._torch = torch

    def _real_inference(self, image_bgr: np.ndarray) -> dict:
        """Run actual SMPL-X regression on a frame.

        NOTE: A full production pipeline would use a regression network
        (e.g. ExPose, PIXIE, or SMPLify-X optimisation) to lift the 2-D
        image to SMPL-X parameters.  For now we demonstrate the forward
        pass with default (zero-pose) parameters to validate that the
        model loads and produces a mesh.
        """
        torch = self._torch

        with torch.no_grad():
            output = self.body_model()

        vertices = output.vertices.cpu().numpy()[0]
        joints = output.joints.cpu().numpy()[0]

        spine_indices = [0, 3, 6, 9]
        spine_joints = joints[spine_indices]
        deviations = np.diff(spine_joints, axis=0)
        lateral_drift = float(np.linalg.norm(deviations[:, [0, 2]], axis=1).mean())
        alignment = round(max(0.0, 1.0 - lateral_drift * 5.0), 2)

        if alignment < 0.78:
            warning = "Significant spinal lateral deviation — stop exercise"
        elif alignment < 0.85:
            warning = "Mild lumbar flexion detected — brace core"
        else:
            warning = "None"

        return {
            "spinal_alignment_score": alignment,
            "posture_warning": warning,
            "shoulder_symmetry": round(
                1.0 - abs(float(joints[16, 1] - joints[17, 1])), 2
            ),
            "hip_torque_ratio": round(
                float(np.linalg.norm(joints[1] - joints[2])) * 10, 2
            ),
            "mock": False,
        }
