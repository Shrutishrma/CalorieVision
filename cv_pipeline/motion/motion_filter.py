"""
cv_pipeline/motion/motion_filter.py
────────────────────────────────────────────────────────────────────────────────
Active/rest segment filter for CalorieVision.

Purpose
-------
Before spending compute on exercise classification or calorie estimation,
filter out "rest" frames (the person is standing still, talking, adjusting
equipment, or the camera is on a static shot) from "active" frames (the
person is actually moving and exercising).

Method
------
Frame-to-frame motion magnitude:

    motion_score(t) = mean_over_landmarks(|pos_t - pos_{t-1}|)

where pos is the (x, y) normalised position of each MediaPipe landmark.

Frames with motion_score > threshold are labelled ACTIVE.
Frames below threshold are labelled REST.

The threshold is tuned for MediaPipe's normalised coordinate space ([0,1]).
Default 0.01 (1% of frame width/height per landmark per frame) works well for
typical workout footage at 30fps; use a lower value for slower movements.

Usage (library)
---------------
    from cv_pipeline.motion.motion_filter import label_motion, filter_active

    frames = extract_keypoints("workout.mp4")
    labels = label_motion(frames)          # list of "active" / "rest"
    active_only = filter_active(frames)    # only the active frame dicts

Usage (CLI)
-----------
    python cv_pipeline/motion/motion_filter.py keypoints.json
    python cv_pipeline/motion/motion_filter.py keypoints.json --threshold 0.008 --out motion.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

ACTIVE_LABEL = "active"
REST_LABEL   = "rest"

# Default motion threshold in normalised coordinate space.
# Empirically: 0.01 catches most exercise movements without triggering on
# camera micro-wobble.
DEFAULT_THRESHOLD: float = 0.01


# ─── Core functions ───────────────────────────────────────────────────────────

def _landmarks_to_xy(frame: dict) -> np.ndarray | None:
    """
    Convert a frame dict's landmarks to a (33, 2) float32 array of (x, y).

    Returns None if the frame has no detected pose.
    """
    landmarks = frame.get("landmarks", [])
    if not landmarks:
        return None
    try:
        arr = np.array([[lm["x"], lm["y"]] for lm in landmarks], dtype=np.float32)
        return arr  # shape (N, 2)
    except (KeyError, TypeError):
        return None


def compute_motion_scores(frames: List[dict]) -> List[float]:
    """
    Compute a per-frame motion score.

    Parameters
    ----------
    frames : list[dict]
        Output of extract_keypoints() — one dict per frame.

    Returns
    -------
    list[float]
        One score per frame.  Frame 0 always gets score 0.0 (no previous
        frame to diff against).  Frames with no detected pose also get 0.0.
    """
    scores: List[float] = []
    prev_xy: np.ndarray | None = None

    for frame in frames:
        xy = _landmarks_to_xy(frame)
        if xy is None or prev_xy is None:
            scores.append(0.0)
            prev_xy = xy
            continue

        if xy.shape != prev_xy.shape:
            scores.append(0.0)
        else:
            diff = np.abs(xy - prev_xy)          # (N, 2)
            scores.append(float(diff.mean()))     # scalar

        prev_xy = xy

    return scores


def label_motion(
    frames: List[dict],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> List[str]:
    """
    Assign "active" or "rest" to every frame.

    Parameters
    ----------
    frames : list[dict]
        Output of extract_keypoints().
    threshold : float
        Motion score threshold.  Frames with score > threshold → "active".
        Default 0.01.

    Returns
    -------
    list[str]
        One label per frame: ``"active"`` or ``"rest"``.
    """
    scores = compute_motion_scores(frames)
    return [ACTIVE_LABEL if s > threshold else REST_LABEL for s in scores]


def filter_active(
    frames: List[dict],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> Tuple[List[dict], List[int]]:
    """
    Return only the active frames, together with their original indices.

    Parameters
    ----------
    frames : list[dict]
        Output of extract_keypoints().
    threshold : float
        Motion threshold.

    Returns
    -------
    active_frames : list[dict]
    active_indices : list[int]
        Original frame indices of the returned frames (useful for time-aligning
        with the full keypoint file).
    """
    labels = label_motion(frames, threshold=threshold)
    active_frames  = [f for f, lb in zip(frames, labels) if lb == ACTIVE_LABEL]
    active_indices = [i for i, lb in enumerate(labels)   if lb == ACTIVE_LABEL]
    return active_frames, active_indices


def summarise(
    frames: List[dict],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """
    Return a summary dict suitable for JSON output.
    """
    scores = compute_motion_scores(frames)
    labels = [ACTIVE_LABEL if s > threshold else REST_LABEL for s in scores]
    n_active = labels.count(ACTIVE_LABEL)
    n_rest   = labels.count(REST_LABEL)
    total    = len(frames)

    logger.info(
        "[motion_filter] %d / %d frames active (%.1f%%)",
        n_active, total, 100 * n_active / max(total, 1),
    )

    return {
        "threshold": threshold,
        "total_frames": total,
        "active_frames": n_active,
        "rest_frames": n_rest,
        "active_pct": round(100 * n_active / max(total, 1), 2),
        "per_frame": [
            {
                "frame_index": f.get("frame_index", i),
                "timestamp": f.get("timestamp", 0.0),
                "motion_score": round(scores[i], 6),
                "label": labels[i],
            }
            for i, f in enumerate(frames)
        ],
    }
def _is_horizontal_pose(frame: dict) -> bool:
    """
    Return True if the person appears to be horizontal (lying/planking).
    Checks if shoulder-to-ankle vertical distance is < 0.35 of frame height.
    """
    lms = frame.get("landmarks", [])
    if not lms or len(lms) < 28:
        return False
    try:
        l_shoulder = lms[11]
        l_ankle    = lms[27]
        if not (l_shoulder and l_ankle and "y" in l_shoulder and "y" in l_ankle):
            return False
        return abs(float(l_shoulder["y"]) - float(l_ankle["y"])) < 0.35
    except (KeyError, TypeError, IndexError):
        return False


def segment_runs(
    frames: List[dict],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    min_active_secs: float = 2.0,
    min_gap_secs: float = 3.0,
    sample_fps: float = 30.0,
) -> List[dict]:
    """
    Group consecutive frames into active segments (runs).
    Merges active runs separated by gaps shorter than min_gap_secs.
    Filters out active runs shorter than min_active_secs.
    Returns a list of dicts with:
      - start_time: float
      - end_time: float
      - label: "active"
      - frames: list[dict]
    """
    if not frames:
        return []

    # Scale threshold proportionally to time delta between samples.
    # threshold=0.01 was tuned for 30fps (dt=33ms). At lower fps (e.g. 2fps, dt=500ms),
    # physical displacement between samples is ~15x larger, so the threshold must scale UP.
    effective_threshold = threshold * (30.0 / max(sample_fps, 0.1))

    labels = label_motion(frames, threshold=effective_threshold)

    # Force horizontal poses (plank, pushup, mountain climber) to ACTIVE
    # regardless of motion score, since they may have near-zero frame-to-frame motion.
    labels = [
        ACTIVE_LABEL if (_is_horizontal_pose(f) or lb == ACTIVE_LABEL) else lb
        for f, lb in zip(frames, labels)
    ]

    timestamps = [f.get("timestamp", 0.0) for f in frames]

    # Find indices of all active frames
    active_indices = [i for i, lb in enumerate(labels) if lb == ACTIVE_LABEL]
    if not active_indices:
        return []

    # Group into runs, merging gaps <= min_gap_secs
    runs_indices = []
    current_run = [active_indices[0]]

    for idx in active_indices[1:]:
        prev_idx = current_run[-1]
        gap_time = timestamps[idx] - timestamps[prev_idx]

        if gap_time <= min_gap_secs:
            # Continue the run, including the gap
            # Add all indices from prev_idx+1 to idx
            for j in range(prev_idx + 1, idx + 1):
                if j not in current_run:
                    current_run.append(j)
        else:
            runs_indices.append(current_run)
            current_run = [idx]
    
    if current_run:
        runs_indices.append(current_run)

    # Filter out short runs and build result
    results = []
    for run in runs_indices:
        start_t = timestamps[run[0]]
        end_t = timestamps[run[-1]]
        if end_t - start_t >= min_active_secs:
            results.append({
                "start_time": start_t,
                "end_time": end_t,
                "label": "active",
                "frames": [frames[i] for i in run]
            })

    return results


# ─── CLI entry point ──────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="motion_filter",
        description="Label frames as active/rest based on landmark motion magnitude.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("keypoints_json", help="JSON file produced by extract_keypoints.py")
    p.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help="Motion score threshold (landmark movement in normalised coords)",
    )
    p.add_argument(
        "--out", "-o", default=None,
        help="Output JSON path. If omitted, prints to stdout.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    kp_path = Path(args.keypoints_json)
    if not kp_path.exists():
        raise FileNotFoundError(f"Keypoints file not found: {kp_path}")

    with open(kp_path, encoding="utf-8") as fh:
        raw = json.load(fh)

    frames = raw["frames"] if isinstance(raw, dict) else raw

    result = summarise(frames, threshold=args.threshold)
    json_str = json.dumps(result, indent=2)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json_str, encoding="utf-8")
        print(f"[motion_filter] Written → {out}  "
              f"({result['active_frames']}/{result['total_frames']} active frames)")
    else:
        print(json_str)


if __name__ == "__main__":
    main()
