"""
cv-pipeline/keypoints/extract_keypoints.py
────────────────────────────────────────────────────────────────────────────────
Stage 1 of the CalorieVision CV pipeline: MediaPipe Pose keypoint extraction.

Usage (library):
    from cv_pipeline.keypoints.extract_keypoints import extract_keypoints
    frames = extract_keypoints("workout.mp4")

Usage (CLI):
    python extract_keypoints.py path/to/video.mp4 --out output.json
    python extract_keypoints.py path/to/video.mp4 --out output.json --max-frames 300
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode


# ─── Constants ────────────────────────────────────────────────────────────────

# Names of all 33 MediaPipe Pose landmarks, indexed by their integer ID.
# Useful for human-readable JSON output.
LANDMARK_NAMES: list[str] = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

assert len(LANDMARK_NAMES) == 33, "Landmark name list must have exactly 33 entries"

# ─── MediaPipe model asset ────────────────────────────────────────────────────
# The Tasks API requires a pre-downloaded .task bundle.  We download it once
# to a local cache inside the repo (ignored by .gitignore via cv-pipeline/models/).

import urllib.request
import hashlib

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
_MODEL_DIR  = Path(__file__).resolve().parents[1] / "models"
_MODEL_PATH = _MODEL_DIR / "pose_landmarker_full.task"


def _ensure_model() -> str:
    """Download the MediaPipe Pose task bundle if not already cached."""
    if _MODEL_PATH.exists():
        return str(_MODEL_PATH)
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[CalorieVision] Downloading MediaPipe model -> {_MODEL_PATH}")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    print("[CalorieVision] Model download complete.")
    return str(_MODEL_PATH)



# ─── Core function ────────────────────────────────────────────────────────────

def extract_keypoints(
    video_path: str,
    *,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    model_complexity: int = 1,
    max_frames: Optional[int] = None,
) -> list[dict]:
    """
    Extract MediaPipe Pose landmarks from every frame of a video.

    Parameters
    ----------
    video_path : str
        Path to the input video file (MP4, AVI, MOV, MKV, …).
    min_detection_confidence : float
        Minimum confidence for initial pose detection (0–1). Default 0.5.
    min_tracking_confidence : float
        Minimum confidence for pose tracking between frames (0–1). Default 0.5.
    model_complexity : int
        Ignored in Tasks API (kept for API compatibility). Use 0/1/2 to select
        Lite/Full/Heavy model bundles once multi-model support is added.
    max_frames : int | None
        If set, stop after this many frames. Useful for quick tests.

    Returns
    -------
    list[dict]
        One dict per frame, **always** including frames where no pose was
        detected (landmarks will be an empty list in that case).

        Each dict has the shape::

            {
                "frame_index":   int,      # 0-based frame number
                "timestamp":     float,    # seconds from start of video
                "pose_detected": bool,     # True if MediaPipe found a pose
                "landmarks": [
                    {
                        "index":      int,    # 0–32
                        "name":       str,    # e.g. "left_shoulder"
                        "x":          float,  # normalised [0, 1] (horizontal)
                        "y":          float,  # normalised [0, 1] (vertical)
                        "z":          float,  # depth relative to hip midpoint
                        "visibility": float,  # landmark confidence [0, 1]
                    },
                    … × 33
                ]
            }

    Raises
    ------
    FileNotFoundError
        If `video_path` does not exist.
    RuntimeError
        If OpenCV cannot open the video file.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # ── Build the Tasks-API landmarker ────────────────────────────────────────
    model_path = _ensure_model()
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=VisionTaskRunningMode.VIDEO,
        min_pose_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        num_poses=1,
    )

    results_list: list[dict] = []

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            # Tasks VIDEO mode requires a monotonically increasing timestamp in ms
            timestamp_ms = int(frame_index * 1000 / fps)
            detection = landmarker.detect_for_video(mp_image, timestamp_ms)

            timestamp_sec = frame_index / fps
            pose_detected = bool(
                detection.pose_landmarks and len(detection.pose_landmarks) > 0
            )

            if pose_detected:
                raw_lms = detection.pose_landmarks[0]  # first (only) person
                landmarks = [
                    {
                        "index":      idx,
                        "name":       LANDMARK_NAMES[idx],
                        "x":          float(lm.x),
                        "y":          float(lm.y),
                        "z":          float(lm.z),
                        "visibility": float(lm.visibility),
                    }
                    for idx, lm in enumerate(raw_lms)
                ]
            else:
                landmarks = []

            results_list.append(
                {
                    "frame_index":   frame_index,
                    "timestamp":     round(timestamp_sec, 6),
                    "pose_detected": pose_detected,
                    "landmarks":     landmarks,
                }
            )

            frame_index += 1
            if max_frames is not None and frame_index >= max_frames:
                break

    cap.release()
    return results_list


# ─── CLI entry point ──────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="extract_keypoints",
        description="Extract MediaPipe Pose keypoints from a video and save as JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("video_path", help="Path to the input video file")
    p.add_argument(
        "--out", "-o",
        default=None,
        help=(
            "Output JSON file path. "
            "Defaults to <video_stem>_keypoints.json in the same directory."
        ),
    )
    p.add_argument(
        "--max-frames", type=int, default=None,
        help="Stop after this many frames (useful for quick testing)",
    )
    p.add_argument(
        "--model-complexity", type=int, choices=[0, 1, 2], default=1,
        help="MediaPipe model: 0=Lite 1=Full 2=Heavy",
    )
    p.add_argument(
        "--min-detection", type=float, default=0.5,
        help="Minimum pose detection confidence (0–1)",
    )
    p.add_argument(
        "--min-tracking", type=float, default=0.5,
        help="Minimum pose tracking confidence (0–1)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    video_path = Path(args.video_path)

    # Determine output path
    if args.out is None:
        out_path = video_path.with_name(f"{video_path.stem}_keypoints.json")
    else:
        out_path = Path(args.out)

    print(f"[CalorieVision] Extracting keypoints from: {video_path}")
    print(f"[CalorieVision] Model complexity: {args.model_complexity}")

    frames = extract_keypoints(
        str(video_path),
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection,
        min_tracking_confidence=args.min_tracking,
        max_frames=args.max_frames,
    )

    # Summary stats
    total_frames = len(frames)
    detected_frames = sum(1 for f in frames if f["pose_detected"])

    print(f"[CalorieVision] Processed {total_frames} frames")
    print(f"[CalorieVision] Pose detected in {detected_frames}/{total_frames} frames "
          f"({100 * detected_frames / max(total_frames, 1):.1f}%)")

    # Write JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "video_path":     str(video_path),
                "total_frames":   total_frames,
                "detected_frames": detected_frames,
                "frames":         frames,
            },
            fh,
            indent=2,
        )

    print(f"[CalorieVision] Output saved -> {out_path}")


if __name__ == "__main__":
    main()
