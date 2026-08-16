"""
fusion_pipeline/segmentation/ocr_title_segmenter.py
────────────────────────────────────────────────────────────────────────────────
Primary Video Segmenter for CalorieVision using On-Screen OCR Title Transitions.

Logic:
1. Samples video frames with optimized EasyOCR ROI detection.
2. Normalizes on-screen exercise captions to the 26-class vocabulary.
3. Detects timestamp transitions when exercise titles appear, change, or disappear.
4. Generates clean temporal exercise boundaries (e.g. 30s/45s workout intervals).
5. Inserts rest boundaries for rest intervals / gaps between exercise sets.
6. Provides a kinematic posture-shift fallback for videos without on-screen captions.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.schemas import Segment, Source
from fusion_pipeline.ocr.ocr_detector import detect_ocr_segments
from fusion_pipeline.ocr.ocr_normalise import normalise_ocr_text

logger = logging.getLogger("ocr_title_segmenter")


def segment_by_ocr_titles(
    video_path: str | Path,
    total_duration_secs: float,
    *,
    sample_interval_secs: float = 3.0,
    min_segment_secs: float = 5.0,
    keypoints: Optional[List[dict]] = None,
    raw_ocr_segs: Optional[List[Segment]] = None,
) -> Tuple[List[Segment], str]:
    """
    Segment a workout video into discrete exercise intervals using on-screen title transitions.

    Returns
    -------
    segments : List[Segment]
        Ordered, non-overlapping segments spanning [0.0, total_duration_secs].
    method_used : str
        "ocr_title_transitions" or "kinematic_posture_fallback"
    """
    path = Path(video_path)
    if (not path.exists() and not keypoints) or total_duration_secs <= 0:
        return _build_fallback_single_segment(total_duration_secs), "empty_fallback"

    # 1. Run rapid OCR on sampled keyframes if not pre-computed
    if raw_ocr_segs is None:
        try:
            raw_ocr_segs = detect_ocr_segments(str(path), sample_every_n_seconds=sample_interval_secs)
        except Exception as exc:
            logger.warning("OCR detector failed: %s; using kinematic fallback", exc)
            raw_ocr_segs = []

    # 2. Extract valid normalized exercise labels and their timestamps
    timed_labels: List[Tuple[float, str, float]] = []  # (time_sec, norm_label, confidence)

    for s in raw_ocr_segs:
        norm = normalise_ocr_text(s.label)
        if norm != "unknown":
            timed_labels.append((s.start_time, norm, s.confidence))

    logger.info("Found %d OCR timestamps with recognized exercise titles", len(timed_labels))

    # 3. If we have meaningful OCR title coverage (at least 2 title timestamps)
    if len(timed_labels) >= 2:
        segments = _build_segments_from_timed_labels(
            timed_labels,
            total_duration_secs,
            min_segment_secs=min_segment_secs,
            sample_interval=sample_interval_secs,
        )
        if segments:
            logger.info("OCR Title Segmenter produced %d discrete exercise intervals", len(segments))
            return segments, "ocr_title_transitions"

    # 4. Fallback for videos with NO readable on-screen captions
    logger.info("Few or no OCR titles recognized. Falling back to kinematic posture segmentation.")
    fallback_segs = _segment_by_kinematic_shifts(keypoints, total_duration_secs)
    return fallback_segs, "kinematic_posture_fallback"


def _build_segments_from_timed_labels(
    timed_labels: List[Tuple[float, str, float]],
    total_duration_secs: float,
    min_segment_secs: float = 5.0,
    sample_interval: float = 3.0,
) -> List[Segment]:
    """Cluster consecutive matching OCR titles into discrete segments."""
    timed_labels = sorted(timed_labels, key=lambda x: x[0])
    raw_blocks: List[dict] = []

    cur_label = timed_labels[0][1]
    cur_start = timed_labels[0][0]
    cur_end = cur_start + sample_interval
    confs = [timed_labels[0][2]]

    for t, label, conf in timed_labels[1:]:
        # If same exercise and within reasonable continuation window (gap < 20s)
        if label == cur_label and (t - cur_end) <= 20.0:
            cur_end = t + sample_interval
            confs.append(conf)
        else:
            # New exercise transition detected!
            avg_conf = sum(confs) / max(len(confs), 1)
            raw_blocks.append({
                "start": cur_start,
                "end": cur_end,
                "label": cur_label,
                "confidence": round(avg_conf, 3),
            })
            cur_label = label
            cur_start = t
            cur_end = t + sample_interval
            confs = [conf]

    # Append last block
    avg_conf = sum(confs) / max(len(confs), 1)
    raw_blocks.append({
        "start": cur_start,
        "end": cur_end,
        "label": cur_label,
        "confidence": round(avg_conf, 3),
    })

    # Filter very short noise blocks (< min_segment_secs)
    valid_blocks = [b for b in raw_blocks if (b["end"] - b["start"]) >= min_segment_secs]
    if not valid_blocks:
        valid_blocks = raw_blocks

    # 4. Fill gaps and build continuous non-overlapping timeline
    final_segments: List[Segment] = []
    seg_idx = 0
    timeline_cursor = 0.0

    for b in valid_blocks:
        b_start = max(b["start"], timeline_cursor)
        b_end = min(b["end"], total_duration_secs)

        if b_end <= b_start:
            continue

        # If there is an introductory or rest gap before this exercise
        if b_start - timeline_cursor >= 3.0:
            final_segments.append(
                Segment(
                    segment_id=f"seg_{seg_idx:04d}",
                    start_time=round(timeline_cursor, 1),
                    end_time=round(b_start, 1),
                    label="rest",
                    confidence=1.0,
                    source=Source.ocr,
                )
            )
            seg_idx += 1

        final_segments.append(
            Segment(
                segment_id=f"seg_{seg_idx:04d}",
                start_time=round(b_start, 1),
                end_time=round(b_end, 1),
                label=b["label"],
                confidence=b["confidence"],
                source=Source.ocr,
            )
        )
        seg_idx += 1
        timeline_cursor = b_end

    # Tail gap to video end
    if total_duration_secs - timeline_cursor >= 3.0:
        final_segments.append(
            Segment(
                segment_id=f"seg_{seg_idx:04d}",
                start_time=round(timeline_cursor, 1),
                end_time=round(total_duration_secs, 1),
                label="rest",
                confidence=1.0,
                source=Source.ocr,
            )
        )

    return final_segments


def _get_window_motion(frames: List[dict]) -> float:
    """Compute mean motion energy across frames."""
    if not frames or len(frames) < 2:
        return 0.0
    from cv_pipeline.motion.motion_filter import compute_motion_scores
    scores = compute_motion_scores(frames)
    return float(np.mean(scores)) if scores else 0.0


def _segment_by_kinematic_shifts(
    keypoints: Optional[List[dict]],
    total_duration_secs: float,
) -> List[Segment]:
    """
    Fallback segmenter for videos with NO captions.
    Segments by active motion vs stationary rest, labeling moving unknown actions as unclassified_exercise.
    """
    if not keypoints:
        return _build_fallback_single_segment(total_duration_secs)

    from cv_pipeline.motion.motion_filter import segment_runs

    active_runs = segment_runs(
        keypoints,
        threshold=0.008,
        min_active_secs=3.0,
        min_gap_secs=4.0,
        sample_fps=2.0,
    )

    if not active_runs:
        # Check overall motion across entire video
        mean_mot = _get_window_motion(keypoints)
        label = "unclassified_exercise" if mean_mot >= 0.005 else "rest"
        return [
            Segment(
                segment_id="seg_0000",
                start_time=0.0,
                end_time=round(total_duration_secs, 1),
                label=label,
                confidence=0.70 if label != "rest" else 1.0,
                source=Source.pose,
            )
        ]

    segments: List[Segment] = []
    seg_idx = 0
    last_end = 0.0

    for run in active_runs:
        st = run["start_time"]
        et = run["end_time"]

        # Check gap between last_end and st
        if st - last_end >= 2.5:
            gap_frames = [f for f in keypoints if last_end <= f.get("timestamp", 0.0) <= st]
            gap_motion = _get_window_motion(gap_frames)
            # Only classify as rest if the person is truly motionless
            gap_label = "rest" if gap_motion < 0.005 else "unclassified_exercise"
            segments.append(
                Segment(
                    segment_id=f"seg_{seg_idx:04d}",
                    start_time=round(last_end, 1),
                    end_time=round(st, 1),
                    label=gap_label,
                    confidence=1.0 if gap_label == "rest" else 0.65,
                    source=Source.pose,
                )
            )
            seg_idx += 1

        # Classify active run posture
        run_frames = run.get("frames", [])
        posture_label = _classify_run_posture(run_frames)
        segments.append(
            Segment(
                segment_id=f"seg_{seg_idx:04d}",
                start_time=round(st, 1),
                end_time=round(et, 1),
                label=posture_label,
                confidence=0.70,
                source=Source.pose,
            )
        )
        seg_idx += 1
        last_end = et

    if total_duration_secs - last_end >= 2.5:
        tail_frames = [f for f in keypoints if last_end <= f.get("timestamp", 0.0) <= total_duration_secs]
        tail_motion = _get_window_motion(tail_frames)
        tail_label = "rest" if tail_motion < 0.005 else "unclassified_exercise"
        segments.append(
            Segment(
                segment_id=f"seg_{seg_idx:04d}",
                start_time=round(last_end, 1),
                end_time=round(total_duration_secs, 1),
                label=tail_label,
                confidence=1.0 if tail_label == "rest" else 0.65,
                source=Source.pose,
            )
        )

    return segments


def _classify_run_posture(frames: List[dict]) -> str:
    """Classify coarse body posture (prone horizontal vs active movement) from frames."""
    if not frames:
        return "unclassified_exercise"

    horizontal_count = 0
    for f in frames:
        lms = f.get("landmarks", [])
        if lms and len(lms) >= 28:
            try:
                sh_y = float(lms[11]["y"])
                ank_y = float(lms[27]["y"])
                if abs(sh_y - ank_y) < 0.35:
                    horizontal_count += 1
            except (KeyError, TypeError, IndexError):
                pass

    if horizontal_count / max(len(frames), 1) > 0.60:
        return "plank"
    return "unclassified_exercise"


def _build_fallback_single_segment(total_duration_secs: float) -> List[Segment]:
    return [
        Segment(
            segment_id="seg_0000",
            start_time=0.0,
            end_time=round(max(total_duration_secs, 1.0), 1),
            label="full_body",
            confidence=0.5,
            source=Source.pose,
        )
    ]
