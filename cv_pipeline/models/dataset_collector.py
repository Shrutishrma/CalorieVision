"""
cv_pipeline/models/dataset_collector.py
────────────────────────────────────────────────────────────────────────────────
Automated Dataset Collector for CalorieVision LSTM Training.

This script automatically:
  1. Searches YouTube for short clips of each of the 26 exercise classes.
  2. Downloads the top N results per exercise using yt-dlp.
  3. Extracts MediaPipe 33-landmark keypoints from each video.
  4. Generates per-frame labels (since each clip is a single exercise).
  5. Merges all exercises into a combined training dataset:
       - dataset/combined_keypoints.json
       - dataset/combined_labels.json

Usage
-----
    # Full automated run (downloads 3 clips per exercise, ~78 clips total):
    python cv_pipeline/models/dataset_collector.py

    # Download 5 clips per exercise for more data:
    python cv_pipeline/models/dataset_collector.py --clips-per-exercise 5

    # Only download videos (skip keypoint extraction):
    python cv_pipeline/models/dataset_collector.py --download-only

    # Only extract keypoints from already-downloaded videos:
    python cv_pipeline/models/dataset_collector.py --extract-only

Prerequisites
-------------
    pip install yt-dlp mediapipe opencv-python numpy

    ffmpeg must be installed on the system:
      Windows: winget install ffmpeg
      macOS:   brew install ffmpeg
      Linux:   sudo apt install ffmpeg
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ─── 26 Exercise Classes ──────────────────────────────────────────────────────

EXERCISE_CLASSES = [
    "squat", "pushup", "jumping_jack", "lunge", "plank", "burpee",
    "mountain_climber", "high_knees", "situp", "jump_rope",
    "bicycle_crunch", "shoulder_press",
    "deadlift", "pull_up", "bench_press", "tricep_dip",
    "leg_raise", "wall_sit", "box_jump", "russian_twist",
    "hip_thrust", "calf_raise", "lateral_raise", "bicep_curl",
    "kettlebell_swing", "superman_hold",
]

# YouTube search queries optimised for short single-exercise demonstration clips.
# These are designed to find 15-60 second clips showing ONLY that exercise.
SEARCH_QUERIES = {
    "squat":            "bodyweight squat exercise form 30 seconds",
    "pushup":           "push ups exercise form short demo",
    "jumping_jack":     "jumping jacks exercise 30 seconds",
    "lunge":            "forward lunges exercise form demo",
    "plank":            "plank hold exercise 30 seconds",
    "burpee":           "burpees exercise demo short",
    "mountain_climber": "mountain climbers exercise 30 seconds",
    "high_knees":       "high knees exercise 30 seconds",
    "situp":            "sit ups crunches exercise form",
    "jump_rope":        "jump rope skipping exercise short",
    "bicycle_crunch":   "bicycle crunches exercise form demo",
    "shoulder_press":   "shoulder press exercise form demo",
    "deadlift":         "deadlift exercise form short demo",
    "pull_up":          "pull ups exercise form demo",
    "bench_press":      "bench press exercise form short",
    "tricep_dip":       "tricep dips exercise bodyweight demo",
    "leg_raise":        "lying leg raises exercise form",
    "wall_sit":         "wall sit exercise hold 30 seconds",
    "box_jump":         "box jumps plyometric exercise demo",
    "russian_twist":    "russian twists exercise form demo",
    "hip_thrust":       "hip thrust glute bridge exercise demo",
    "calf_raise":       "calf raises exercise form demo",
    "lateral_raise":    "lateral raises exercise form demo",
    "bicep_curl":       "bicep curls exercise form short",
    "kettlebell_swing": "kettlebell swings exercise form demo",
    "superman_hold":    "superman hold exercise form demo",
}


# ─── Step 1: Search YouTube & get video URLs ─────────────────────────────────

def search_youtube(query: str, max_results: int = 5) -> List[str]:
    """
    Search YouTube for a query and return a list of video URLs.
    Uses yt-dlp's built-in search functionality (no API key needed).
    """
    search_url = f"ytsearch{max_results}:{query}"
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--flat-playlist",
        "--print", "url",
        "--no-warnings",
        "--match-filter", "duration < 120",   # Only clips under 2 minutes
        search_url,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60,
        )
        urls = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        return urls[:max_results]
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning(f"Search failed for '{query}': {e}")
        return []


# ─── Step 2: Download a single video ─────────────────────────────────────────

def download_video(url: str, output_dir: Path, exercise: str, index: int) -> Optional[Path]:
    """
    Download a single YouTube video to output_dir/{exercise}_{index}.mp4.
    Returns the path to the downloaded file, or None on failure.
    """
    filename = f"{exercise}_{index:02d}"
    output_template = str(output_dir / f"{filename}.%(ext)s")
    final_path = output_dir / f"{filename}.mp4"

    if final_path.exists():
        logger.info(f"  ✓ Already downloaded: {final_path.name}")
        return final_path

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--format", "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--output", output_template,
        "--no-playlist",
        "--no-warnings",
        "--quiet",
        "--socket-timeout", "30",
        "--retries", "2",
        url,
    ]
    try:
        subprocess.run(cmd, timeout=120, check=True, capture_output=True)
        # yt-dlp may save with a different extension, find the actual file
        candidates = list(output_dir.glob(f"{filename}.*"))
        mp4_candidates = [c for c in candidates if c.suffix == ".mp4"]
        if mp4_candidates:
            logger.info(f"  ✓ Downloaded: {mp4_candidates[0].name}")
            return mp4_candidates[0]
        elif candidates:
            logger.info(f"  ✓ Downloaded: {candidates[0].name}")
            return candidates[0]
        else:
            logger.warning(f"  ✗ Download produced no file for {url}")
            return None
    except subprocess.TimeoutExpired:
        logger.warning(f"  ✗ Download timed out for {url}")
        return None
    except subprocess.CalledProcessError as e:
        logger.warning(f"  ✗ Download failed for {url}: {e.stderr[:200] if e.stderr else 'unknown error'}")
        return None


# ─── Step 3: Extract keypoints ────────────────────────────────────────────────

def extract_keypoints_from_video(video_path: Path, output_path: Path) -> Optional[dict]:
    """
    Extract MediaPipe 33-landmark keypoints from a video file.
    Returns the keypoints dict, or None on failure.
    """
    if output_path.exists():
        logger.info(f"  ✓ Keypoints already extracted: {output_path.name}")
        with open(output_path, encoding="utf-8") as f:
            return json.load(f)

    try:
        from cv_pipeline.keypoints.extract_keypoints import extract_keypoints
        frames = extract_keypoints(str(video_path), max_frames=1800)  # cap at 60s @ 30fps
        data = {
            "video_path": str(video_path),
            "total_frames": len(frames),
            "frames": frames,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        logger.info(f"  ✓ Extracted {len(frames)} frames → {output_path.name}")
        return data
    except Exception as e:
        logger.warning(f"  ✗ Keypoint extraction failed for {video_path.name}: {e}")
        return None


# ─── Step 4: Generate per-frame labels ────────────────────────────────────────

def generate_labels(keypoints_data: dict, exercise: str) -> list[str]:
    """
    Since each clip is a SINGLE exercise, every frame gets the same label.
    """
    num_frames = len(keypoints_data.get("frames", []))
    return [exercise] * num_frames


# ─── Step 5: Merge all exercises into combined dataset ────────────────────────

def merge_dataset(dataset_dir: Path) -> tuple[dict, list[str]]:
    """
    Merge all per-exercise keypoints and labels into a single combined dataset.
    Returns (combined_keypoints_dict, combined_labels_list).
    """
    all_frames = []
    all_labels = []
    stats = {}

    for exercise in EXERCISE_CLASSES:
        exercise_dir = dataset_dir / exercise
        if not exercise_dir.exists():
            logger.warning(f"  ⚠ No data for {exercise}")
            stats[exercise] = 0
            continue

        exercise_frames = 0
        for kp_file in sorted(exercise_dir.glob("*_keypoints.json")):
            label_file = kp_file.with_name(kp_file.stem.replace("_keypoints", "_labels") + ".json")
            if not label_file.exists():
                continue

            with open(kp_file, encoding="utf-8") as f:
                kp_data = json.load(f)
            with open(label_file, encoding="utf-8") as f:
                labels = json.load(f)

            frames = kp_data.get("frames", [])
            if len(frames) != len(labels):
                logger.warning(f"  ⚠ Frame/label mismatch in {kp_file.name}: {len(frames)} vs {len(labels)}")
                min_len = min(len(frames), len(labels))
                frames = frames[:min_len]
                labels = labels[:min_len]

            # Filter out frames where pose detection failed (e.g. animated/cartoon clips or empty frames)
            valid_frames = []
            valid_labels = []
            for frame, lbl in zip(frames, labels):
                if frame.get("pose_detected", False) and frame.get("landmarks"):
                    valid_frames.append(frame)
                    valid_labels.append(lbl)

            if len(valid_frames) < 15:
                logger.warning(f"  ⚠ Skipping {kp_file.name} — too few valid pose detections ({len(valid_frames)}/{len(frames)})")
                continue

            all_frames.extend(valid_frames)
            all_labels.extend(valid_labels)
            exercise_frames += len(valid_frames)

        stats[exercise] = exercise_frames

    # Print stats table
    logger.info("\n" + "=" * 50)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 50)
    total = 0
    for ex, count in stats.items():
        marker = "✓" if count > 0 else "✗"
        logger.info(f"  {marker} {ex:<22s} {count:>6d} frames")
        total += count
    logger.info("-" * 50)
    logger.info(f"  TOTAL{' ' * 18}{total:>6d} frames")
    logger.info("=" * 50)

    combined_kp = {"total_frames": len(all_frames), "frames": all_frames}
    return combined_kp, all_labels


# ─── Main orchestrator ────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Automated Dataset Collector for CalorieVision LSTM Training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (search + download + extract + merge):
  python cv_pipeline/models/dataset_collector.py

  # More clips per exercise:
  python cv_pipeline/models/dataset_collector.py --clips-per-exercise 5

  # Only download (skip keypoint extraction):
  python cv_pipeline/models/dataset_collector.py --download-only

  # Only extract from already-downloaded videos:
  python cv_pipeline/models/dataset_collector.py --extract-only
        """,
    )
    parser.add_argument(
        "--clips-per-exercise", type=int, default=3,
        help="Number of YouTube clips to download per exercise (default: 3)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(_REPO_ROOT / "dataset"),
        help="Directory to save videos, keypoints, and labels",
    )
    parser.add_argument(
        "--download-only", action="store_true",
        help="Only download videos, skip keypoint extraction",
    )
    parser.add_argument(
        "--extract-only", action="store_true",
        help="Only extract keypoints from already-downloaded videos",
    )
    parser.add_argument(
        "--exercises", nargs="+", default=None,
        help="Only process specific exercises (e.g. --exercises squat pushup lunge)",
    )
    args = parser.parse_args(argv)

    dataset_dir = Path(args.output_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    exercises = args.exercises if args.exercises else EXERCISE_CLASSES
    n_clips = args.clips_per_exercise

    logger.info("=" * 60)
    logger.info("CalorieVision Automated Dataset Collector")
    logger.info("=" * 60)
    logger.info(f"  Exercises to process : {len(exercises)}")
    logger.info(f"  Clips per exercise   : {n_clips}")
    logger.info(f"  Output directory     : {dataset_dir}")
    logger.info(f"  Download only        : {args.download_only}")
    logger.info(f"  Extract only         : {args.extract_only}")
    logger.info("=" * 60)

    for i, exercise in enumerate(exercises, 1):
        logger.info(f"\n[{i}/{len(exercises)}] ── {exercise.upper()} ──")
        exercise_dir = dataset_dir / exercise
        exercise_dir.mkdir(parents=True, exist_ok=True)

        # ── Search & Download ─────────────────────────────────────────────
        if not args.extract_only:
            query = SEARCH_QUERIES.get(exercise, f"{exercise} exercise form demo")
            logger.info(f"  Searching YouTube: '{query}'")
            urls = search_youtube(query, max_results=n_clips + 2)  # fetch extras in case some fail

            if not urls:
                logger.warning(f"  ✗ No results found for {exercise}")
                continue

            downloaded = 0
            for j, url in enumerate(urls):
                if downloaded >= n_clips:
                    break
                path = download_video(url, exercise_dir, exercise, downloaded + 1)
                if path:
                    downloaded += 1
                time.sleep(1)  # polite delay between downloads

            logger.info(f"  Downloaded {downloaded}/{n_clips} clips for {exercise}")

        # ── Extract Keypoints & Generate Labels ───────────────────────────
        if not args.download_only:
            video_files = sorted(exercise_dir.glob("*.mp4"))
            if not video_files:
                logger.warning(f"  ✗ No video files found in {exercise_dir}")
                continue

            for vid in video_files:
                kp_path = vid.with_name(vid.stem + "_keypoints.json")
                label_path = vid.with_name(vid.stem + "_labels.json")

                kp_data = extract_keypoints_from_video(vid, kp_path)
                if kp_data:
                    labels = generate_labels(kp_data, exercise)
                    with open(label_path, "w", encoding="utf-8") as f:
                        json.dump(labels, f)
                    logger.info(f"  ✓ Labels: {len(labels)} frames → {label_path.name}")

    # ── Merge into combined dataset ───────────────────────────────────────
    if not args.download_only:
        logger.info("\n\nMerging all exercises into combined dataset...")
        combined_kp, combined_labels = merge_dataset(dataset_dir)

        combined_kp_path = dataset_dir / "combined_keypoints.json"
        combined_labels_path = dataset_dir / "combined_labels.json"

        with open(combined_kp_path, "w", encoding="utf-8") as f:
            json.dump(combined_kp, f)
        with open(combined_labels_path, "w", encoding="utf-8") as f:
            json.dump(combined_labels, f)

        logger.info(f"\n✅ Combined dataset saved:")
        logger.info(f"   Keypoints : {combined_kp_path}")
        logger.info(f"   Labels    : {combined_labels_path}")
        logger.info(f"   Total     : {len(combined_labels)} frames")
        logger.info(f"\nNext step — train the LSTM:")
        logger.info(f"   python cv_pipeline/models/train_lstm.py \\")
        logger.info(f"     --data {combined_kp_path} \\")
        logger.info(f"     --labels {combined_labels_path} \\")
        logger.info(f"     --epochs 50 \\")
        logger.info(f"     --checkpoint cv_pipeline/models/lstm_best.pt")


if __name__ == "__main__":
    main()
