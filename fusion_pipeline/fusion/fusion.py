"""
fusion_pipeline/fusion/fusion.py
────────────────────────────────────────────────────────────────────────────────
Fusion layer: combines pose-classifier segments with OCR segments.

Decision logic
--------------
For each time window that overlaps a pose-predicted segment:

  Pose confident + OCR agrees  → AGREE: source=fused, high confidence
  Pose confident + OCR absent  → POSE-ONLY: source=pose, use pose confidence
  OCR confident + pose absent  → OCR-ONLY: source=ocr, use OCR confidence
  Both present + DISAGREE      → DISAGREE: log it, keep pose, conf = 0.5
  Both absent / low confidence → UNKNOWN: source=fused, conf=0.0

Disagreements and every low-confidence prediction (< LOW_CONF_THRESHOLD) are
automatically logged to `pipeline_run.log` in the project root.

Usage (library)
---------------
    from fusion_pipeline.fusion.fusion import fuse_segments, configure_logger

    fused = fuse_segments(pose_segments, ocr_segments, video_duration=120.0)
    # fused: list[Segment] with source=fused, source=pose, or source=ocr

Usage (CLI)
-----------
    python fusion_pipeline/fusion/fusion.py \\
        pose_segments.json ocr_segments.json \\
        --duration 120.0 \\
        --out fused.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.schemas import Segment, Source

# ─── Logging configuration ────────────────────────────────────────────────────

# ─── Logging configuration ────────────────────────────────────────────────────

_default_log_path = Path(__file__).resolve().parents[2] / "eval" / "failure_log.jsonl"
LOG_FILE = Path(os.getenv("PIPELINE_LOG_PATH", str(_default_log_path)))


def configure_logger(log_path: Path | None = None) -> logging.Logger:
    """
    Return a logger writing console messages to stderr and setting up log directory.
    If log_path is None, checks PIPELINE_LOG_PATH env var or defaults to eval/failure_log.jsonl.
    """
    if log_path is None:
        log_path = Path(os.getenv("PIPELINE_LOG_PATH", str(_default_log_path)))

    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger_name = f"pipeline_run.{hash(str(log_path)) & 0xFFFFFF:06x}"
    logger = logging.getLogger(logger_name)

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # File handler (append) for human-readable run logs
        fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
        logger.addHandler(fh)

        # Console handler (INFO only)
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
        logger.addHandler(ch)

    return logger


def _log_failure_jsonl(log_path: Path, event_dict: dict) -> None:
    """Append a single JSONL event entry to log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event_dict) + "\n")


# ─── Constants ────────────────────────────────────────────────────────────────

AGREE_CONF_BOOST  = 0.10   # bonus added when pose and OCR agree
LOW_CONF_THRESHOLD = 0.50  # predictions below this are logged as LOW_CONF
DISAGREE_CONF     = 0.50   # fixed confidence when pose and OCR disagree
OVERLAP_MIN_SECS  = 0.1    # minimum overlap to count OCR as "covering" a pose window


# ─── Overlap utilities ────────────────────────────────────────────────────────

def _overlap(a: Segment, b: Segment) -> float:
    """Return the overlap in seconds between two segments."""
    start = max(a.start_time, b.start_time)
    end   = min(a.end_time,   b.end_time)
    return max(0.0, end - start)


def _find_overlapping_ocr(
    pose_seg: Segment,
    ocr_segs: List[Segment],
) -> List[Segment]:
    """Return all OCR segments that overlap a pose segment by at least OVERLAP_MIN_SECS."""
    return [s for s in ocr_segs if _overlap(pose_seg, s) >= OVERLAP_MIN_SECS]


# ─── Core fusion function ─────────────────────────────────────────────────────

def fuse_segments(
    pose_segments: List[Segment],
    ocr_segments:  List[Segment],
    *,
    video_duration: float = 0.0,
    log_path:       Path  = LOG_FILE,
) -> List[Segment]:
    """
    Fuse pose-classifier output with OCR segments.

    Parameters
    ----------
    pose_segments : list[Segment]  - output of the LSTM/baseline classifier
    ocr_segments  : list[Segment]  - output of ocr_detector.py
    video_duration: float          - total video length in seconds (informational)
    log_path      : Path           - path for the pipeline disagreement log

    Returns
    -------
    list[Segment]
        Fused segments ordered by start_time.
        All have their segment_id re-assigned as "fused_{N:04d}".

    Side Effects
    ------------
    Disagreements and low-confidence predictions are written to `log_path`.
    """
    logger = configure_logger(log_path)
    run_ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    logger.info("─── Fusion run start (%s)  pose=%d  ocr=%d ───",
                run_ts, len(pose_segments), len(ocr_segments))

    fused: List[Segment] = []
    seg_counter = 0

    for pose_seg in sorted(pose_segments, key=lambda s: s.start_time):
        overlapping_ocr = _find_overlapping_ocr(pose_seg, ocr_segments)
        ocr_majority = None
        best_ocr = None

        # ── Determine outcome ─────────────────────────────────────────────────
        if not overlapping_ocr:
            # No OCR signal -> keep pose as-is but keep source=pose
            outcome_source    = Source.pose
            outcome_label     = pose_seg.label
            outcome_conf      = pose_seg.confidence
            outcome_tag       = "POSE_ONLY"

        else:
            # Majority-vote on OCR labels that cover this segment
            ocr_labels = [s.label for s in overlapping_ocr]
            from collections import Counter
            ocr_majority, ocr_count = Counter(ocr_labels).most_common(1)[0]
            best_ocr = max(overlapping_ocr, key=lambda s: s.confidence)

            if ocr_majority.lower() == pose_seg.label.lower():
                # AGREE
                outcome_source = Source.fused
                outcome_label  = pose_seg.label
                outcome_conf   = min(
                    1.0,
                    max(pose_seg.confidence, best_ocr.confidence) + AGREE_CONF_BOOST
                )
                outcome_tag    = "AGREE"
            else:
                # DISAGREE — log, keep pose, source=disagreement
                outcome_source = Source.disagreement
                outcome_label  = pose_seg.label
                outcome_conf   = DISAGREE_CONF
                outcome_tag    = "DISAGREE"

                logger.warning(
                    "DISAGREE  t=%.1f-%.1fs  pose=%s(%.2f)  ocr=%s(%.2f)",
                    pose_seg.start_time, pose_seg.end_time,
                    pose_seg.label, pose_seg.confidence,
                    ocr_majority, best_ocr.confidence,
                )

                _log_failure_jsonl(log_path, {
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                    "event_type": "DISAGREE",
                    "segment_id": pose_seg.segment_id,
                    "start_time": pose_seg.start_time,
                    "end_time": pose_seg.end_time,
                    "pose_label": pose_seg.label,
                    "pose_confidence": pose_seg.confidence,
                    "ocr_label": ocr_majority,
                    "ocr_confidence": best_ocr.confidence if best_ocr else None,
                    "fused_label": outcome_label,
                    "fused_confidence": outcome_conf,
                    "source": outcome_source.value,
                    "outcome_tag": outcome_tag,
                })

        # ── Low-confidence logging ────────────────────────────────────────────
        if outcome_conf < LOW_CONF_THRESHOLD:
            logger.warning(
                "LOW_CONF  t=%.1f-%.1fs  label=%s  conf=%.2f  source=%s  tag=%s",
                pose_seg.start_time, pose_seg.end_time,
                outcome_label, outcome_conf,
                outcome_source.value, outcome_tag,
            )

            if outcome_tag != "DISAGREE":
                _log_failure_jsonl(log_path, {
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                    "event_type": "LOW_CONF",
                    "segment_id": pose_seg.segment_id,
                    "start_time": pose_seg.start_time,
                    "end_time": pose_seg.end_time,
                    "pose_label": pose_seg.label,
                    "pose_confidence": pose_seg.confidence,
                    "ocr_label": ocr_majority,
                    "ocr_confidence": best_ocr.confidence if best_ocr else None,
                    "fused_label": outcome_label,
                    "fused_confidence": outcome_conf,
                    "source": outcome_source.value,
                    "outcome_tag": outcome_tag,
                })

        # ── Build fused segment ───────────────────────────────────────────────
        fused_seg = Segment(
            segment_id = f"fused_{seg_counter:04d}",
            start_time = pose_seg.start_time,
            end_time   = pose_seg.end_time,
            label      = outcome_label,
            confidence = round(outcome_conf, 4),
            source     = outcome_source,
        )
        fused.append(fused_seg)
        seg_counter += 1

    logger.info("─── Fusion run complete  %d segments produced ───", len(fused))
    return fused


# ─── Load helpers ─────────────────────────────────────────────────────────────

def _load_segments(path: Path) -> List[Segment]:
    """Load a JSON file produced by any stage (scene_detect, ocr_detector, etc.)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Support both {"segments": [...]} and a bare list
    items = raw.get("segments", raw) if isinstance(raw, dict) else raw
    return [Segment.from_dict(d) for d in items]


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fuse pose-classifier and OCR segments into a single timeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pose_json", help="JSON file of pose-predicted segments")
    parser.add_argument("ocr_json",  help="JSON file of OCR segments")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Total video duration in seconds (informational)")
    parser.add_argument("--out", "-o", default=None,
                        help="Output JSON path. If omitted, prints to stdout.")
    parser.add_argument("--log", default=str(LOG_FILE),
                        help="Path for the pipeline disagreement log.")
    args = parser.parse_args(argv)

    pose_segs = _load_segments(Path(args.pose_json))
    ocr_segs  = _load_segments(Path(args.ocr_json))

    fused = fuse_segments(
        pose_segs, ocr_segs,
        video_duration=args.duration,
        log_path=Path(args.log),
    )

    out_data = {
        "total_fused_segments": len(fused),
        "segments": [s.to_dict() for s in fused],
    }
    json_str = json.dumps(out_data, indent=2)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json_str, encoding="utf-8")
        print(f"[fusion] Written -> {out}  ({len(fused)} segments)")
    else:
        print(json_str)


if __name__ == "__main__":
    main()
