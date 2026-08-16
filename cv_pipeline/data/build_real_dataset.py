"""
cv_pipeline/data/build_real_dataset.py
────────────────────────────────────────────────────────────────────────────────
Builds a 100% REAL human keypoint dataset from downloaded workout videos
across the 8 target classes + rest.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("build_real_dataset")

from cv_pipeline.keypoints.extract_keypoints import extract_keypoints

# Define real human video intervals for the 8 exercise classes + rest
REAL_INTERVALS = [
    # Pushups
    {"video": "IODxDxX7oi4.mp4", "label": "pushup", "start": 30.0, "end": 75.0},
    {"video": "IODxDxX7oi4.mp4", "label": "pushup", "start": 90.0, "end": 135.0},
    {"video": "cbKkB3POqaY.mp4", "label": "pushup", "start": 312.0, "end": 357.0},
    {"video": "cbKkB3POqaY.mp4", "label": "pushup", "start": 807.0, "end": 852.0},
    {"video": "dJlFmxiL11s.mp4", "label": "pushup", "start": 120.0, "end": 165.0},

    # Squats
    {"video": "aclHkVaku9U.mp4", "label": "squat", "start": 20.0, "end": 65.0},
    {"video": "aclHkVaku9U.mp4", "label": "squat", "start": 80.0, "end": 125.0},
    {"video": "cbKkB3POqaY.mp4", "label": "squat", "start": 57.0, "end": 145.0},
    {"video": "cbKkB3POqaY.mp4", "label": "squat", "start": 368.0, "end": 456.0},
    {"video": "cbKkB3POqaY.mp4", "label": "squat", "start": 566.0, "end": 598.0},

    # Planks
    {"video": "gC_L9qAHVJ8.mp4", "label": "plank", "start": 10.0, "end": 60.0},
    {"video": "gC_L9qAHVJ8.mp4", "label": "plank", "start": 75.0, "end": 130.0},
    {"video": "cbKkB3POqaY.mp4", "label": "plank", "start": 255.0, "end": 300.0},
    {"video": "cbKkB3POqaY.mp4", "label": "plank", "start": 510.0, "end": 555.0},
    {"video": "dJlFmxiL11s.mp4", "label": "plank", "start": 300.0, "end": 345.0},

    # Jumping Jacks
    {"video": "cbKkB3POqaY.mp4", "label": "jumping_jack", "start": 28.0, "end": 55.0},
    {"video": "L_xrDAtykMI.mp4", "label": "jumping_jack", "start": 30.0, "end": 90.0},
    {"video": "L_xrDAtykMI.mp4", "label": "jumping_jack", "start": 120.0, "end": 180.0},

    # Lunges
    {"video": "cbKkB3POqaY.mp4", "label": "lunge", "start": 150.0, "end": 210.0},
    {"video": "cbKkB3POqaY.mp4", "label": "lunge", "start": 600.0, "end": 650.0},

    # Situps / Crunches
    {"video": "ml6cT4AZdqI.mp4", "label": "situp", "start": 25.0, "end": 85.0},
    {"video": "ml6cT4AZdqI.mp4", "label": "situp", "start": 110.0, "end": 170.0},
    {"video": "ml6cT4AZdqI.mp4", "label": "situp", "start": 200.0, "end": 260.0},

    # Burpees
    {"video": "cbKkB3POqaY.mp4", "label": "burpee", "start": 467.0, "end": 498.0},

    # Mountain Climbers
    {"video": "cbKkB3POqaY.mp4", "label": "mountain_climber", "start": 665.0, "end": 697.0},

    # Rest / Standing / Pauses
    {"video": "v7AYKMP6rOE.mp4", "label": "rest", "start": 10.0, "end": 60.0},
    {"video": "v7AYKMP6rOE.mp4", "label": "rest", "start": 120.0, "end": 180.0},
    {"video": "cbKkB3POqaY.mp4", "label": "rest", "start": 0.0, "end": 25.0},
    {"video": "cbKkB3POqaY.mp4", "label": "rest", "start": 300.0, "end": 311.0},
    {"video": "cbKkB3POqaY.mp4", "label": "rest", "start": 456.0, "end": 467.0},
    {"video": "cbKkB3POqaY.mp4", "label": "rest", "start": 498.0, "end": 510.0},
]


def build_dataset() -> None:
    logger.info("=== Building Real Human Pose Dataset from Downloaded Video Clips ===")
    test_videos_dir = _REPO_ROOT / "shared" / "test-videos"

    all_frames: List[dict] = []
    all_labels: List[str] = []
    class_frame_counts: Dict[str, int] = {}
    class_seq_counts: Dict[str, int] = {}

    # Track extracted video keypoints cache to avoid re-running MediaPipe
    video_keypoints_cache: Dict[str, List[dict]] = {}

    for item in REAL_INTERVALS:
        vid_filename = item["video"]
        label = item["label"]
        st = item["start"]
        et = item["end"]

        vid_path = test_videos_dir / vid_filename
        if not vid_path.exists():
            logger.warning("Video file not found: %s", vid_path)
            continue

        # Extract or load keypoints
        if vid_filename not in video_keypoints_cache:
            # Check if keypoint json cache exists
            vid_id = vid_filename.split(".")[0]
            kp_cache_file = test_videos_dir / f"{vid_id}_keypoints.json"
            if kp_cache_file.exists():
                try:
                    logger.info("Loading cached keypoints for %s", vid_filename)
                    with open(kp_cache_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    kps = data.get("frames", data) if isinstance(data, dict) else data
                    video_keypoints_cache[vid_filename] = kps
                except Exception:
                    kps = []
            else:
                kps = []

            if not kps:
                logger.info("Extracting keypoints for %s via MediaPipe (2 FPS)...", vid_filename)
                kps = extract_keypoints(str(vid_path), sample_fps=2.0)
                video_keypoints_cache[vid_filename] = kps

        kps = video_keypoints_cache.get(vid_filename, [])
        if not kps:
            logger.warning("No keypoints available for %s", vid_filename)
            continue

        # Filter frames for this time interval
        interval_frames = [
            f for f in kps
            if st <= f.get("timestamp", 0.0) <= et and f.get("landmarks")
        ]

        if not interval_frames:
            continue

        # Add to global frames list
        for f in interval_frames:
            f_copy = dict(f)
            f_copy["frame_index"] = len(all_frames)
            all_frames.append(f_copy)
            all_labels.append(label)

        class_frame_counts[label] = class_frame_counts.get(label, 0) + len(interval_frames)
        # Approximate 15-frame sequences (with stride 7)
        n_seqs = max(1, (len(interval_frames) - 15) // 7 + 1)
        class_seq_counts[label] = class_seq_counts.get(label, 0) + n_seqs

    # Save to dataset/train_data.json and dataset/train_labels.json
    dataset_dir = _REPO_ROOT / "dataset"
    dataset_dir.mkdir(exist_ok=True)

    data_path = dataset_dir / "train_data.json"
    labels_path = dataset_dir / "train_labels.json"

    with open(data_path, "w", encoding="utf-8") as f:
        json.dump({"frames": all_frames}, f)

    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(all_labels, f)

    logger.info("=== Dataset Build Complete! ===")
    logger.info("Total Real Frames Extracted: %d", len(all_frames))
    logger.info("Total Real Labels: %d", len(all_labels))
    logger.info("Data saved to %s and %s", data_path, labels_path)

    print("\n" + "=" * 60)
    print("REAL DATASET COMPOSITION (Source: Real Human Videos)")
    print("=" * 60)
    print(f"{'Class':<20} | {'Frames':<10} | {'Estimated Sequences (15f)':<25}")
    print("-" * 60)
    for c in sorted(class_frame_counts.keys()):
        print(f"{c:<20} | {class_frame_counts[c]:<10} | {class_seq_counts.get(c, 0):<25}")
    print("=" * 60)


if __name__ == "__main__":
    build_dataset()
