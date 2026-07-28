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
    sample_interval_frames = max(1, int(fps * sample_every_n_seconds))

    logger.info(
        "[ocr_detector] Sampling every %d frames (%.1f s) in %s",
        sample_interval_frames, sample_every_n_seconds, video_path,
    )

    segments: List[Segment] = []
    seg_counter = 0
    frame_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % sample_interval_frames == 0:
                timestamp_sec = frame_index / fps
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = reader.readtext(rgb, detail=1, paragraph=False)

                # Filter by confidence and concatenate all words in this frame
                high_conf = [
                    (text.strip(), conf)
                    for (_, text, conf) in results
                    if conf >= min_confidence and text.strip()
                ]

                if high_conf:
                    # Combine all detected words into one label string
                    label = " ".join(text for text, _ in high_conf).lower()
                    confidence = float(max(conf for _, conf in high_conf))

                    end_sec = timestamp_sec + sample_every_n_seconds
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
                    logger.debug(
                        "[ocr_detector] Frame %d (%.2f s): %r (conf=%.2f)",
                        frame_index, timestamp_sec, label, confidence,
                    )
                    seg_counter += 1

            frame_index += 1

    finally:
        cap.release()

    logger.info("[ocr_detector] Produced %d OCR segment(s).", len(segments))
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
        print(f"[ocr_detector] {len(segments)} OCR segment(s) written → {out_path}")
    else:
        print(json_str)


if __name__ == "__main__":
    main()
