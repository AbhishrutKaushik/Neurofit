"""
extract_pose_data.py — Video-to-CSV Pose Extractor for Neuro-Fit
=================================================================

Walks ``datasets/videos/<exercise_name>/`` and processes every video
frame-by-frame through MediaPipe BlazePose, extracting the 33 world
landmarks (99 features: x, y, z per landmark).  Results are written
to a single ``pose_dataset.csv`` suitable for PyTorch training.

Expected directory layout
─────────────────────────
    datasets/
    └── videos/
        ├── bicep_curl/
        │   ├── clip_001.mp4
        │   └── clip_002.avi
        ├── squat/
        │   └── clip_001.mp4
        └── deadlift/
            └── ...

Run:
    python scripts/extract_pose_data.py

Output:
    datasets/pose_dataset.csv   (99 float columns + 1 label column)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import mediapipe as mp
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = PROJECT_ROOT / "datasets" / "videos"
OUTPUT_CSV = PROJECT_ROOT / "datasets" / "pose_dataset.csv"

NUM_LANDMARKS = 33
NUM_FEATURES = NUM_LANDMARKS * 3  # x, y, z per landmark

HEADER = [f"lm{i}_{axis}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")] + ["label"]


def collect_videos(root: Path) -> list[tuple[Path, str]]:
    """Return [(video_path, exercise_label), ...] sorted deterministically."""
    pairs: list[tuple[Path, str]] = []
    if not root.is_dir():
        print(f"[ERROR] Video directory not found: {root}", file=sys.stderr)
        sys.exit(1)
    for exercise_dir in sorted(root.iterdir()):
        if not exercise_dir.is_dir():
            continue
        label = exercise_dir.name
        for vf in sorted(exercise_dir.iterdir()):
            if vf.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
                pairs.append((vf, label))
    return pairs


def count_total_frames(videos: list[tuple[Path, str]]) -> int:
    total = 0
    for vpath, _ in videos:
        cap = cv2.VideoCapture(str(vpath))
        total += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    return total


def extract(videos: list[tuple[Path, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = count_total_frames(videos)

    mp_pose = mp.solutions.pose

    with (
        open(out_path, "w", newline="") as fh,
        mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose,
    ):
        writer = csv.writer(fh)
        writer.writerow(HEADER)

        pbar = tqdm(total=total_frames, desc="Extracting poses", unit="frame")
        rows_written = 0
        skipped = 0

        for vpath, label in videos:
            cap = cv2.VideoCapture(str(vpath))
            if not cap.isOpened():
                tqdm.write(f"[WARN] Cannot open {vpath}")
                continue

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                pbar.update(1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)

                if not result.pose_world_landmarks:
                    skipped += 1
                    continue

                row: list[float | str] = []
                for lm in result.pose_world_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])
                row.append(label)

                writer.writerow(row)
                rows_written += 1

            cap.release()

        pbar.close()

    print(f"\nDone.  {rows_written:,} frames extracted, {skipped:,} skipped (no pose).")
    print(f"Output: {out_path}")


def main() -> None:
    print(f"Scanning {VIDEO_DIR} for exercise videos...")
    videos = collect_videos(VIDEO_DIR)
    if not videos:
        print("[ERROR] No video files found.", file=sys.stderr)
        sys.exit(1)

    labels = sorted(set(lbl for _, lbl in videos))
    print(f"Found {len(videos)} videos across {len(labels)} exercises: {labels}")
    extract(videos, OUTPUT_CSV)


if __name__ == "__main__":
    main()
