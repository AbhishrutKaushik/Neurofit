"""
validate_phase3.py — Phase 3 SMPL-X Lab Validation Script
==========================================================

Feeds a single static image through the SMPL-X pipeline, generates the
3-D body mesh, displays it in an interactive trimesh window, and prints
timing + device diagnostics.

Usage (from repo root, venv active):

    python scripts/validate_phase3.py                       # uses default test image
    python scripts/validate_phase3.py --image path/to/squat.jpg
    python scripts/validate_phase3.py --image squat.jpg --model-folder models/smplx

Pass/Fail criteria:
    PASS  — mesh renders, forward-pass < 5 s, device = cuda
    REVIEW — mesh renders but forward-pass 5-30 s (likely CPU fallback)
    FAIL  — crash, mock fallback, or > 30 s
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3: SMPL-X mesh validation on lab GPU",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a test image (.jpg/.png). If omitted, a synthetic "
             "640x480 frame is used (still exercises the full forward pass).",
    )
    parser.add_argument(
        "--model-folder",
        type=str,
        default=str(PROJECT_ROOT / "models" / "smplx"),
        help="Directory containing SMPL-X .pkl model files.",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Skip the interactive 3-D window (useful for headless CI).",
    )
    args = parser.parse_args()

    _divider("STEP 1 — PyTorch & CUDA diagnostics")

    import torch  # noqa: E402

    print(f"  PyTorch version : {torch.__version__}")
    print(f"  CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU name        : {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version    : {torch.version.cuda}")
        mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        print(f"  GPU memory      : {mem:.1f} GB")
    else:
        print("  WARNING: No CUDA GPU detected — will run on CPU (slow).")

    _divider("STEP 2 — Load test image")

    import cv2  # noqa: E402
    import numpy as np  # noqa: E402

    if args.image:
        img_path = Path(args.image)
        if not img_path.is_file():
            print(f"  ERROR: Image not found: {img_path}")
            sys.exit(1)
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  ERROR: cv2.imread failed for {img_path}")
            sys.exit(1)
        print(f"  Loaded {img_path.name}  shape={image.shape}")
    else:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        print("  No --image flag; using synthetic 640x480 black frame.")
        print("  (The mesh will be in neutral T-pose — that is expected.)")

    _divider("STEP 3 — Initialise SMPLXSpotChecker (MOCK_MODE=False)")

    t_init_start = time.perf_counter()

    from perception.smplx_spotcheck import SMPLXSpotChecker  # noqa: E402

    checker = SMPLXSpotChecker(
        MOCK_MODE=False,
        model_folder=args.model_folder,
    )
    t_init = time.perf_counter() - t_init_start

    if checker.mock_mode:
        print("\n  FAIL: Checker fell back to MOCK_MODE=True.")
        print("        Check the model folder and logs above.")
        sys.exit(1)

    print(f"  Model loaded in {t_init:.2f} s on device={checker._device}")

    _divider("STEP 4 — Forward pass → mesh + metrics")

    result = checker.get_mesh_data(image, mediapipe_landmarks=None)

    verts = result["vertices"]
    faces = result["faces"]
    joints = result["joints_3d"]
    elapsed = result["elapsed_s"]
    device = result["device"]
    metrics = result["metrics"]

    print(f"  Device used     : {device}")
    print(f"  Forward pass    : {elapsed:.4f} s")
    print(f"  Vertices        : {verts.shape}")
    print(f"  Faces           : {faces.shape}")
    print(f"  Joints          : {joints.shape}")
    print()
    print("  Biomechanical metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v}")

    _divider("STEP 5 — Verdict")

    if elapsed < 5.0 and "cuda" in device:
        verdict = "PASS"
        msg = f"Mesh generated in {elapsed:.2f}s on CUDA."
    elif elapsed < 5.0:
        verdict = "REVIEW"
        msg = (
            f"Mesh generated in {elapsed:.2f}s but on CPU. "
            "Check PyTorch CUDA installation."
        )
    elif elapsed < 30.0:
        verdict = "REVIEW"
        msg = (
            f"Mesh took {elapsed:.2f}s — likely running on CPU. "
            "Reinstall PyTorch with --index-url .../cu124."
        )
    else:
        verdict = "FAIL"
        msg = f"Forward pass took {elapsed:.2f}s — unacceptably slow."

    colour = {"PASS": "\033[92m", "REVIEW": "\033[93m", "FAIL": "\033[91m"}
    reset = "\033[0m"
    print(f"\n  {colour.get(verdict, '')}{verdict}{reset}: {msg}")

    if args.no_viewer:
        print("\n  --no-viewer set; skipping 3-D visualisation.")
        sys.exit(0 if verdict == "PASS" else 1)

    _divider("STEP 6 — 3-D mesh viewer (trimesh)")

    try:
        import trimesh  # noqa: E402
    except ImportError:
        print("  trimesh not installed.  Run:")
        print("      pip install trimesh[easy]")
        print("  Then re-run this script.")
        sys.exit(1)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    mesh.visual.face_colors = [180, 180, 180, 255]

    joint_spheres = []
    for j in joints[:22]:
        sphere = trimesh.creation.uv_sphere(radius=0.015)
        sphere.apply_translation(j)
        sphere.visual.face_colors = [255, 60, 60, 255]
        joint_spheres.append(sphere)

    scene = trimesh.Scene([mesh] + joint_spheres)

    print("  Opening interactive 3-D window...")
    print("  (Close the window to exit this script.)\n")
    scene.show(caption="Neuro-Fit Phase 3 — SMPL-X Mesh Validation")


def _divider(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
