"""
fusion_pipeline/ocr/ocr_detector.py
────────────────────────────────────────────────────────────────────────────────
Stage: EasyOCR-based on-screen text detection for workout videos.

Purpose
-------
Read on-screen text (rep counters, exercise captions, timers) from video
frames and return them as `Segment` objects with source=Source.ocr.

The detector samples frames at a configurable rate (default every 2 seconds)
to avoid running the expensive OCR model on every frame.  For each sampled
frame that contains readable text, a Segment spanning the sample interval is
emitted.

Output
------
A list of `Segment` objects where:
  • source       = Source.ocr
  • label        = the OCR-detected text (stripped, lowercased)
  • confidence   = highest per-word confidence from EasyOCR for that frame
  • segment_id   = "ocr_NNNN" (zero-padded sequential index)
  • start_time   = timestamp of the sampled frame (seconds)
  • end_time     = start_time + sample_interval

Library usage
-------------
    from fusion_pipeline.ocr.ocr_detector import detect_ocr_segments
    segments = detect_ocr_segments("workout.mp4", sample_every_n_seconds=2.0)

CLI usage
---------
    python fusion-pipeline/ocr/ocr_detector.py path/to/video.mp4
    python fusion-pipeline/ocr/ocr_detector.py path/to/video.mp4 --sample-rate 1.0 --out ocr_out.json

Notes
-----
• EasyOCR is GPU-aware but falls back to CPU automatically.
• First call downloads ~200 MB of language model weights; subsequent calls
  use the local cache (~/.EasyOCR).
• Frames where OCR returns no text are silently skipped (not emitted).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Ensure repo root is importable ───────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]  # parents[2] = repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.schemas import Segment, Source


# ─── Core function ────────────────────────────────────────────────────────────

def detect_ocr_segments(
    video_path: str,
    *,
    sample_every_n_seconds: float = 2.0,
    min_confidence: float = 0.4,
    languages: List[str] | None = None,
    gpu: bool = False,
) -> List[Segment]:
    """
    Sample a video at regular intervals and run EasyOCR on each frame.

    Parameters
    ----------
    video_path : str
        Path to a local MP4/AVI/MKV/MOV file.
    sample_every_n_seconds : float
        How often to run OCR (in seconds).  Lower = more thorough, slower.
        Default 2.0 s.
    min_confidence : float
        Minimum per-word confidence from EasyOCR to include the result.
        Default 0.4.  Words below this threshold are filtered out.
    languages : list[str] | None
        EasyOCR language codes.  Defaults to ["en"].
    gpu : bool
        Pass True to enable GPU acceleration in EasyOCR.  Default False
        (CPU-only, avoids CUDA dependency in CI).

    Returns
    -------
    list[Segment]
        One Segment per frame where OCR found text above `min_confidence`.
        Ordered by start_time.  Frames with no readable text are omitted.

    Raises
    ------
    FileNotFoundError
        If video_path does not exist.
    RuntimeError
        If OpenCV cannot open the video.
    ImportError
        If easyocr is not installed.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    try:
        import easyocr  # noqa: F401  (import check only; Reader built below)
    except ImportError as exc:
        raise ImportError(
            "easyocr is required. Install with: pip install easyocr"
        ) from exc

    if languages is None:
        languages = ["en"]

    # Build EasyOCR Reader (expensive — downloads model on first call)
    logger.info("[ocr_detector] Initialising EasyOCR reader (gpu=%s) …", gpu)
    import easyocr as _easyocr
    reader = _easyocr.Reader(languages, gpu=gpu, verbose=False)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration_secs = (total_frame_count / fps) if (total_frame_count > 0 and fps > 0) else 300.0

    # Adaptive sample interval: for long videos (>10 min), sample every 10-12s
    # Max samples capped at 120 so even a 30-min video runs OCR in < 10 seconds
    effective_interval = max(sample_every_n_seconds, duration_secs / 120.0) if duration_secs > 120 else sample_every_n_seconds

    sample_timestamps = []
    t = 0.5
    while t < duration_secs:
        sample_timestamps.append(t)
        t += effective_interval

    logger.info(
        "[ocr_detector] Sampling %d keyframes (interval=%.1fs, duration=%.1fs) with ROI banner cropping in %s",
        len(sample_timestamps), effective_interval, duration_secs, video_path,
    )

    segments: List[Segment] = []
    seg_counter = 0

    try:
        for timestamp_sec in sample_timestamps:
            # Direct seek to millisecond position (skips sequential frame decoding)
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000.0)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            h, w = frame.shape[:2]
            if h < 20 or w < 20:
                continue

            # ROI Optimization: Workout titles/timers appear in the top 25% or bottom 25%
            # Crop top and bottom banners and stack them vertically (drops middle 50% body clutter)
            top_h = int(h * 0.28)
            bot_h = int(h * 0.28)
            top_banner = frame[0:top_h, 0:w]
            bot_banner = frame[h - bot_h:h, 0:w]
            stacked_banner = np.vstack([top_banner, bot_banner])

            # Downscale banner width to max 640px for 5x faster CRAFT inference
            if w > 640:
                scale = 640.0 / w
                target_w = 640
                target_h = max(20, int(stacked_banner.shape[0] * scale))
                stacked_banner = cv2.resize(stacked_banner, (target_w, target_h), interpolation=cv2.INTER_AREA)

            rgb = cv2.cvtColor(stacked_banner, cv2.COLOR_BGR2RGB)
            results = reader.readtext(rgb, detail=1, paragraph=False)

            high_conf = [
                (text.strip(), conf)
                for (_, text, conf) in results
                if conf >= min_confidence and text.strip()
            ]

            if high_conf:
                label = " ".join(text for text, _ in high_conf).lower()
                confidence = float(max(conf for _, conf in high_conf))
                end_sec = timestamp_sec + effective_interval
                segments.append(
                    Segment(
                        segment_id=f"ocr_{seg_counter:04d}",
                        start_time=round(timestamp_sec, 6),
                        end_time=round(end_sec, 6),
                        label=label,
                        confidence=min(confidence, 1.0),
                        source=Source.ocr,
                    )
                )
                seg_counter += 1

    finally:
        cap.release()

    logger.info("[ocr_detector] Produced %d OCR segment(s) in rapid mode.", len(segments))
    return segments


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ocr_detector",
        description="Detect on-screen text in workout videos using EasyOCR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("video_path", help="Path to the local video file.")
    p.add_argument(
        "--sample-rate", type=float, default=2.0,
        help="Sample OCR every N seconds.",
    )
    p.add_argument(
        "--min-confidence", type=float, default=0.4,
        help="Minimum EasyOCR word confidence to include.",
    )
    p.add_argument(
        "--languages", nargs="+", default=["en"],
        help="EasyOCR language codes.",
    )
    p.add_argument(
        "--gpu", action="store_true", default=False,
        help="Enable GPU acceleration.",
    )
    p.add_argument(
        "--out", "-o", default=None,
        help="Output JSON path.  If omitted, prints to stdout.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    segments = detect_ocr_segments(
        args.video_path,
        sample_every_n_seconds=args.sample_rate,
        min_confidence=args.min_confidence,
        languages=args.languages,
        gpu=args.gpu,
    )

    output = {
        "video_path": args.video_path,
        "total_ocr_segments": len(segments),
        "segments": [s.to_dict() for s in segments],
    }

    json_str = json.dumps(output, indent=2)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json_str, encoding="utf-8")
        print(f"[ocr_detector] {len(segments)} OCR segment(s) written -> {out_path}")
    else:
        print(json_str)


if __name__ == "__main__":
    main()
