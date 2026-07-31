"""
shared/cache.py
─────────────────────────────────────────────────────────────────────────────
File-based caching utilities for expensive pipeline stages (keypoint extraction,
scene detection, OCR, and full pipeline results).

Cache files are saved in `shared/test-videos/` alongside downloaded videos:
  • {video_id}_keypoints.json
  • {video_id}_scenes.json
  • {video_id}_ocr.json
  • {video_id}_result.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.schemas import Segment

logger = logging.getLogger("shared.cache")

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = _REPO_ROOT / "shared" / "test-videos"


def get_cache_dir() -> Path:
    DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_CACHE_DIR


# ─── Keypoints Cache ──────────────────────────────────────────────────────────

def get_cached_keypoints(video_id: str, cache_dir: Optional[Path] = None) -> Optional[List[dict]]:
    cdir = cache_dir or get_cache_dir()
    path = cdir / f"{video_id}_keypoints.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        frames = data.get("frames", data if isinstance(data, list) else [])
        logger.info("Cache hit: Keypoints for video_id=%s (%d frames)", video_id, len(frames))
        return frames
    except Exception as exc:
        logger.warning("Failed reading cached keypoints for %s: %s", video_id, exc)
        return None


def cache_keypoints(video_id: str, frames: List[dict], video_path: str = "", cache_dir: Optional[Path] = None) -> Path:
    cdir = cache_dir or get_cache_dir()
    path = cdir / f"{video_id}_keypoints.json"
    data = {
        "video_id": video_id,
        "video_path": video_path,
        "total_frames": len(frames),
        "detected_frames": sum(1 for f in frames if f.get("pose_detected")),
        "frames": frames,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Cached keypoints for video_id=%s -> %s", video_id, path)
    return path


# ─── Scene Detection Cache ───────────────────────────────────────────────────

def get_cached_scenes(video_id: str, cache_dir: Optional[Path] = None) -> Optional[List[Segment]]:
    cdir = cache_dir or get_cache_dir()
    path = cdir / f"{video_id}_scenes.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        raw_segs = data.get("segments", data if isinstance(data, list) else [])
        segments = [Segment.from_dict(d) for d in raw_segs]
        logger.info("Cache hit: Scenes for video_id=%s (%d scenes)", video_id, len(segments))
        return segments
    except Exception as exc:
        logger.warning("Failed reading cached scenes for %s: %s", video_id, exc)
        return None


def cache_scenes(video_id: str, segments: List[Segment], cache_dir: Optional[Path] = None) -> Path:
    cdir = cache_dir or get_cache_dir()
    path = cdir / f"{video_id}_scenes.json"
    data = {
        "video_id": video_id,
        "total_scenes": len(segments),
        "segments": [s.to_dict() for s in segments],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Cached scenes for video_id=%s -> %s", video_id, path)
    return path


# ─── OCR Cache ────────────────────────────────────────────────────────────────

def get_cached_ocr(video_id: str, cache_dir: Optional[Path] = None) -> Optional[List[Segment]]:
    cdir = cache_dir or get_cache_dir()
    path = cdir / f"{video_id}_ocr.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        raw_segs = data.get("segments", data if isinstance(data, list) else [])
        segments = [Segment.from_dict(d) for d in raw_segs]
        logger.info("Cache hit: OCR for video_id=%s (%d segments)", video_id, len(segments))
        return segments
    except Exception as exc:
        logger.warning("Failed reading cached OCR for %s: %s", video_id, exc)
        return None


def cache_ocr(video_id: str, segments: List[Segment], cache_dir: Optional[Path] = None) -> Path:
    cdir = cache_dir or get_cache_dir()
    path = cdir / f"{video_id}_ocr.json"
    data = {
        "video_id": video_id,
        "total_ocr_segments": len(segments),
        "segments": [s.to_dict() for s in segments],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Cached OCR for video_id=%s -> %s", video_id, path)
    return path


# ─── Full Pipeline Result Cache ──────────────────────────────────────────────

def get_cached_pipeline_result(video_id: str, cache_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    cdir = cache_dir or get_cache_dir()
    path = cdir / f"{video_id}_result.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        logger.info("Cache hit: Full pipeline result for video_id=%s", video_id)
        return data
    except Exception as exc:
        logger.warning("Failed reading cached pipeline result for %s: %s", video_id, exc)
        return None


def cache_pipeline_result(video_id: str, result_dict: Dict[str, Any], cache_dir: Optional[Path] = None) -> Path:
    cdir = cache_dir or get_cache_dir()
    path = cdir / f"{video_id}_result.json"
    path.write_text(json.dumps(result_dict, indent=2), encoding="utf-8")
    logger.info("Cached full pipeline result for video_id=%s -> %s", video_id, path)
    return path
