"""
fusion_pipeline/segmentation/scene_detect.py
────────────────────────────────────────────────────────────────────────────────
Stage: Scene-boundary detection using PySceneDetect.

Purpose
-------
Produce a list of candidate time-windows from a local video file by detecting
hard scene cuts (content changes between adjacent frames).  These raw
boundaries are *not* exercise labels — they are a coarse segmentation hint
passed to the fusion layer, which will later decide whether a boundary aligns
with an actual exercise transition.

Output
------
A list of `Segment` objects (from shared/schemas.py) where:
  • source     = Source.scene_cut   (the new enum value added for this stage)
  • label      = "unknown"          (no exercise label assigned at this stage)
  • confidence = 1.0                (scene-cut detector is deterministic)
  • segment_id = "scene_NNNN"       (zero-padded sequential index)
  • start_time / end_time           (in seconds from video start)

CLI usage
---------
    python fusion-pipeline/segmentation/scene_detect.py path/to/video.mp4
    python fusion-pipeline/segmentation/scene_detect.py path/to/video.mp4 --threshold 27 --out cuts.json

Library usage
-------------
    from fusion_pipeline.segmentation.scene_detect import detect_scenes
    segments = detect_scenes("workout.mp4")
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Ensure repo root is on sys.path for shared imports ───────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]  # parents[2] = repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.schemas import Segment, Source


# ─── Core function ────────────────────────────────────────────────────────────

def detect_scenes(
    video_path: str,
    *,
    threshold: float = 27.0,
    min_scene_len_frames: int = 15,
) -> List[Segment]:
    """
    Detect scene cuts in a local video file using PySceneDetect's
    ContentDetector algorithm.

    Parameters
    ----------
    video_path : str
        Path to the local MP4/AVI/MKV/MOV file.
    threshold : float
        ContentDetector threshold.  Lower = more sensitive to colour changes.
        Default 27.0 matches PySceneDetect's documented default.
    min_scene_len_frames : int
        Minimum scene length in frames.  Scenes shorter than this are merged
        with the preceding one.  Default 15 frames (~0.5 s at 30 fps).

    Returns
    -------
    list[Segment]
        Ordered list of scene segments.  Each Segment has source=scene_cut,
        label="unknown", confidence=1.0.  Returns an empty list if no cuts
        are found (single-scene video).

    Raises
    ------
    FileNotFoundError
        If video_path does not exist.
    RuntimeError
        If PySceneDetect cannot open the video.
    ImportError
        If scenedetect package is not installed.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector
    except ImportError as exc:
        raise ImportError(
            "scenedetect is required. Install with: pip install scenedetect[opencv]"
        ) from exc

    logger.info("[scene_detect] Opening video: %s", video_path)

    try:
        video = open_video(str(path))
    except Exception as exc:
        raise RuntimeError(f"PySceneDetect could not open video: {video_path}") from exc

    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(
            threshold=threshold,
            min_scene_len=min_scene_len_frames,
        )
    )

    scene_manager.detect_scenes(video, show_progress=False)
    scene_list = scene_manager.get_scene_list()

    logger.info("[scene_detect] Detected %d scene(s) in %s", len(scene_list), video_path)

    segments: List[Segment] = []
    for idx, (start_tc, end_tc) in enumerate(scene_list):
        segment_id = f"scene_{idx:04d}"
        start_sec = start_tc.seconds
        end_sec = end_tc.seconds

        # Guard: end must be strictly after start
        if end_sec <= start_sec:
            logger.warning(
                "[scene_detect] Skipping zero-length scene %s (%.3f -> %.3f)",
                segment_id, start_sec, end_sec,
            )
            continue

        segments.append(
            Segment(
                segment_id=segment_id,
                start_time=round(start_sec, 6),
                end_time=round(end_sec, 6),
                label="unknown",
                confidence=1.0,
                source=Source.scene_cut,
            )
        )

    return segments


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scene_detect",
        description="Detect scene cuts in a local video file (PySceneDetect).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("video_path", help="Path to the local video file.")
    p.add_argument(
        "--threshold", type=float, default=27.0,
        help="ContentDetector threshold (lower = more sensitive).",
    )
    p.add_argument(
        "--min-scene-len", type=int, default=15,
        help="Minimum scene length in frames.",
    )
    p.add_argument(
        "--out", "-o", default=None,
        help="Output JSON file. If omitted, prints to stdout.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    segments = detect_scenes(
        args.video_path,
        threshold=args.threshold,
        min_scene_len_frames=args.min_scene_len,
    )

    output = {
        "video_path": args.video_path,
        "total_scenes": len(segments),
        "segments": [s.to_dict() for s in segments],
    }

    json_str = json.dumps(output, indent=2)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json_str, encoding="utf-8")
        print(f"[scene_detect] {len(segments)} scene(s) written -> {out_path}")
    else:
        print(json_str)


if __name__ == "__main__":
    main()
