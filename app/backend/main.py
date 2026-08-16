"""
app/backend/main.py
────────────────────────────────────────────────────────────────────────────────
CalorieVision FastAPI application — full pipeline API.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger("calorievision.api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="CalorieVision API",
    description=(
        "Multi-stage CV pipeline: MediaPipe Pose → PyTorch LSTM → EasyOCR → MET Calorie Estimation. "
        "Accepts YouTube video IDs or URLs and returns exercise segments with per-rep calorie estimates."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-memory job store ───────────────────────────────────────────────────────

_JOBS: Dict[str, Dict[str, Any]] = {}

# ─── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = REPO_ROOT / "shared" / "test_videos_manifest.json"

# ─── MET table ────────────────────────────────────────────────────────────────

MET_TABLE: Dict[str, float] = {
    # Original 12
    "squat":            5.0,
    "pushup":           5.0,
    "jumping_jack":     8.0,
    "lunge":            4.0,
    "plank":            3.5,
    "burpee":           8.0,
    "mountain_climber": 7.0,
    "high_knees":       8.0,
    "situp":            4.5,
    "jump_rope":        10.0,
    "bicycle_crunch":   4.0,
    "shoulder_press":   5.0,
    # Expanded 14
    "deadlift":         6.0,
    "pull_up":          5.5,
    "bench_press":      5.0,
    "tricep_dip":       4.5,
    "leg_raise":        3.5,
    "wall_sit":         3.0,
    "box_jump":         8.0,
    "russian_twist":    4.0,
    "hip_thrust":       4.5,
    "calf_raise":       3.5,
    "lateral_raise":    4.0,
    "bicep_curl":       4.5,
    "kettlebell_swing": 9.0,
    "superman_hold":    3.5,
    # Aliases and generic workout categories
    "glute_bridge":     4.5,
    "crunch":           4.5,
    "rest":             1.0,
    "unknown":          3.0,
    "yoga":             3.0,
    "full_body":        6.0,
    "hiit":             8.0,
}

TIER_MET_SCALE = {"beginner": 0.85, "intermediate": 1.0, "advanced": 1.2}


def _calories(exercise: str, duration_secs: float, weight_kg: float, tier: str) -> float:
    met = MET_TABLE.get(exercise, 3.5)
    scale = TIER_MET_SCALE.get(tier, 1.0)
    return round(met * scale * weight_kg * (duration_secs / 3600), 2)


# ─── Schemas ──────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    video_id: Optional[str] = None
    video_url: Optional[str] = None
    weight_kg: float = 70.0
    user_tier: str = "intermediate"
    video_duration_mins: Optional[float] = None
    force_recompute: bool = False


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["infra"], summary="Liveness probe")
async def health() -> dict:
    return {"status": "ok", "service": "CalorieVision API", "version": "1.0.0"}


@app.get("/manifest", tags=["catalogue"], summary="Workout video catalogue")
async def manifest() -> list:
    if not MANIFEST_PATH.exists():
        return []
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        return data
    except Exception as exc:
        logger.warning("Manifest parse error: %s", exc)
        return []


@app.post("/validate-segments", tags=["validation"], summary="Validate segments payload")
async def validate_segments(segments: list[dict]) -> dict:
    from dataclasses import asdict
    from shared.schemas import Segment
    validated = [Segment(**s) if isinstance(s, dict) else s for s in segments]
    return {
        "total": len(validated),
        "segments": [
            {
                "segment_id": s.segment_id,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "label": s.label,
                "confidence": s.confidence,
                "source": s.source.value if hasattr(s.source, "value") else str(s.source),
            }
            for s in validated
        ],
    }


@app.post("/analyze", tags=["pipeline"], summary="Analyse a workout video")
async def analyze(req: AnalyzeRequest, sync: bool = False):
    if not req.video_id and not req.video_url:
        raise HTTPException(status_code=422, detail="Provide video_id or video_url")

    job_id = str(uuid.uuid4())
    _JOBS[job_id] = {
        "status": "processing",
        "progress": 0.03,
        "stage": "Initializing",
        "result": None,
        "error": None,
        "cancelled": False,
    }

    def _run_pipeline():
        try:
            _update_job(job_id, 0.08, "Stage 0: Video Acquisition")
            # Check for cancellation before starting heavy work
            if _JOBS.get(job_id, {}).get("cancelled"):
                _JOBS[job_id].update({"status": "cancelled", "progress": 1.0, "stage": "Cancelled"})
                return
            result = _real_pipeline(req, job_id)
            # Final cancellation check before writing result
            if _JOBS.get(job_id, {}).get("cancelled"):
                _JOBS[job_id].update({"status": "cancelled", "progress": 1.0, "stage": "Cancelled"})
                return
            _JOBS[job_id].update({"status": "completed", "progress": 1.0, "stage": "Complete", "result": result})
        except Exception as exc:
            if _JOBS.get(job_id, {}).get("cancelled"):
                _JOBS[job_id].update({"status": "cancelled", "progress": 1.0, "stage": "Cancelled"})
                return
            logger.exception("Pipeline error for job %s", job_id)
            _JOBS[job_id].update({"status": "failed", "progress": 1.0, "stage": "Error", "error": str(exc)})

    thread = threading.Thread(target=_run_pipeline, daemon=True)
    thread.start()

    if sync:
        thread.join(timeout=300)
        job = _JOBS.get(job_id, {})
        if job.get("status") == "completed":
            return job["result"]
        return {"job_id": job_id, "status": "processing"}

    return {"job_id": job_id, "status": "processing", "progress": 0.03}


@app.post("/cancel/{job_id}", tags=["pipeline"], summary="Cancel a running job")
async def cancel_job(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") not in ("processing",):
        return {"job_id": job_id, "status": job.get("status"), "message": "Job not running"}
    job["cancelled"] = True
    job["status"] = "cancelled"
    job["stage"] = "Cancelled by user"
    logger.info("Job %s cancelled by user", job_id)
    return {"job_id": job_id, "status": "cancelled", "message": "Cancellation requested"}


@app.get("/status/{job_id}", tags=["pipeline"], summary="Poll async job status")
async def job_status(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# ─── Pipeline logic ───────────────────────────────────────────────────────────

def _update_job(job_id: str, progress: float, stage: str):
    if job_id in _JOBS:
        _JOBS[job_id]["progress"] = progress
        _JOBS[job_id]["stage"] = stage
    logger.info("[%s] %.0f%% — %s", job_id[:8], progress * 100, stage)


# ─── Real pipeline runner ─────────────────────────────────────────────────────

def _real_pipeline(req: AnalyzeRequest, job_id: str) -> Dict[str, Any]:
    """Execute end-to-end multi-modal pipeline with concurrent worker threads."""
    import sys
    import numpy as np
    sys.path.insert(0, str(REPO_ROOT))

    video_id = req.video_id or _extract_video_id(req.video_url or "")
    video_url = req.video_url or f"https://www.youtube.com/watch?v={video_id}"

    _update_job(job_id, 0.15, "Stage 0: Video Acquisition & Metadata (yt-dlp)")
    video_path, real_duration, title = _download_video_and_meta(video_url, video_id)

    if real_duration <= 0:
        real_duration = 300.0

    _update_job(job_id, 0.25, "Stage 1: Pose & OCR Extraction (Parallel Workers)")

    def on_pose_progress(cur: int, total: int):
        ratio = min(1.0, max(0.0, cur / max(total, 1)))
        p = 0.25 + 0.40 * ratio
        pct = int(ratio * 100)
        _update_job(job_id, p, f"Stage 1 & 2: Pose & OCR Processing ({pct}%)")

    # Adaptive sample rate for long videos
    target_sample_fps = 1.0 if real_duration > 720.0 else 2.0
    ocr_interval = max(3.0, real_duration / 50.0) if real_duration > 120.0 else 3.0

    from concurrent.futures import ThreadPoolExecutor
    from fusion_pipeline.ocr.ocr_detector import detect_ocr_segments
    from fusion_pipeline.segmentation.ocr_title_segmenter import segment_by_ocr_titles
    from cv_pipeline.motion.motion_filter import compute_motion_scores

    raw_ocr_segs = []
    keypoints = []
    fps = target_sample_fps

    with ThreadPoolExecutor(max_workers=2) as executor:
        f_pose = executor.submit(
            _extract_keypoints, video_path, video_id, real_duration, req.force_recompute, on_pose_progress
        )
        f_ocr = executor.submit(
            detect_ocr_segments, str(video_path) if video_path else "", sample_every_n_seconds=ocr_interval
        )
        try:
            keypoints, fps = f_pose.result()
        except Exception as exc:
            logger.warning("Parallel pose extraction failed: %s", exc)
            keypoints, fps = [], target_sample_fps

        try:
            raw_ocr_segs = f_ocr.result()
        except Exception as exc:
            logger.warning("Parallel OCR extraction failed: %s", exc)
            raw_ocr_segs = []

    _update_job(job_id, 0.70, "Stage 2: Multi-Modal Title & Kinematic Fusion")
    raw_segments, seg_method = segment_by_ocr_titles(
        video_path=str(video_path) if video_path else "",
        total_duration_secs=real_duration,
        keypoints=keypoints,
        raw_ocr_segs=raw_ocr_segs,
    )

    _update_job(job_id, 0.85, "Stage 3 & 4: Biomechanical Kinematics & LSTM Classification")
    from cv_pipeline.models.lstm_classifier import load_lstm_model, classify_run
    lstm_model = None
    try:
        lstm_model, model_type = load_lstm_model()
        logger.info("Live pipeline loaded model: %s (%s)", lstm_model.__class__.__name__, model_type)
    except Exception as exc:
        logger.warning("Could not load LSTM model: %s", exc)

    # Process each segment: motion energy audit + LSTM verification
    final_segments = []
    for s in raw_segments:
        dur = round(s.end_time - s.start_time, 1)
        if dur <= 0:
            continue

        st = s.start_time
        et = s.end_time
        label = s.label
        conf = s.confidence

        # Get keypoint frames in this window
        window_frames = [
            f for f in keypoints
            if st <= f.get("timestamp", 0.0) <= et
        ] if keypoints else []

        # Compute real frame-to-frame motion energy
        mean_motion = 0.0
        if len(window_frames) >= 2:
            scores = compute_motion_scores(window_frames)
            mean_motion = float(np.mean(scores)) if scores else 0.0

        # Classify as rest ONLY if the person is not moving at all (< 0.005 motion score)
        if label == "rest":
            if mean_motion >= 0.005:
                label = "unclassified_exercise"
                conf = 0.65
        elif label in ("unknown", "full_body"):
            label = "unclassified_exercise"
            conf = 0.60

        # If active exercise window and LSTM model is loaded, refine classification
        if keypoints and lstm_model and label not in ("rest",):
            if len(window_frames) >= 4:
                pose_label, pose_conf = classify_run(window_frames, lstm_model)
                if seg_method == "kinematic_posture_fallback":
                    # Require 60% confidence to trust any classification.
                    # jumping_jack in particular needs 70% — it tends to dominate at
                    # lower confidence because it's the most common class in training.
                    min_conf = 0.70 if pose_label == "jumping_jack" else 0.60
                    if pose_label not in ("unknown", "unclassified_exercise") and pose_conf >= min_conf:
                        label = pose_label
                        conf = pose_conf
                    else:
                        label = "unclassified_exercise"
                        conf = 0.60
                elif pose_label == label:
                    conf = min(1.0, max(conf, pose_conf) + 0.10)

        final_segments.append({
            "start_secs": round(st, 1),
            "end_secs": round(et, 1),
            "duration_secs": dur,
            "exercise": label,
            "confidence": round(conf, 2),
            "calories": _calories(label, dur, req.weight_kg, req.user_tier),
        })

    _update_job(job_id, 0.95, "Stage 5: MET Calorie Estimation")
    time.sleep(0.1)

    total_calories = sum(s.get("calories", 0) for s in final_segments)
    # Active duration includes all exercise including unclassified exercises (excludes only motionless rest)
    active_secs = sum(s.get("duration_secs", 0) for s in final_segments if s.get("exercise") != "rest")
    actual_total_secs = round(real_duration)

    return {
        "video_id": video_id,
        "video_title": title or video_id,
        "classifier_used": "ocr_title_fusion" if seg_method == "ocr_title_transitions" else "pose_fallback",
        "segmentation_method": seg_method,
        "total_calories": round(total_calories, 1),
        "active_duration_secs": round(active_secs),
        "total_duration_secs": actual_total_secs,
        "duration_secs": round(real_duration),
        "weight_kg": req.weight_kg,
        "user_tier": req.user_tier,
        "segments": final_segments,
        "failure_events": [],
    }


def _extract_video_id(url: str) -> str:
    if "v=" in url:
        return url.split("v=")[1].split("&")[0].split("?")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    if "shorts/" in url:
        return url.split("shorts/")[1].split("?")[0]
    return url.strip()


def _get_video_path(out_dir: Path, video_id: str) -> Optional[Path]:
    """Find valid, non-fragment video file for video_id."""
    candidates = [
        f for f in out_dir.glob(f"{video_id}.*")
        if f.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov", ".avi")
        and not re.search(r"\.f\d+\.", f.name)
        and "temp" not in f.name.lower()
        and "part" not in f.name.lower()
    ]
    if not candidates:
        return None
    # Prioritize MP4
    mp4s = [f for f in candidates if f.suffix.lower() == ".mp4"]
    return mp4s[0] if mp4s else candidates[0]


def _download_video_and_meta(url: str, video_id: str) -> tuple[Optional[Path], float, str]:
    """Download via yt-dlp and extract real duration and title."""
    out_dir = REPO_ROOT / "shared" / "test-videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean up any leftover temporary locks or fragment files
    for f in out_dir.glob(f"{video_id}*.temp*"):
        try:
            f.unlink()
        except Exception:
            pass
    for f in out_dir.glob(f"{video_id}*.part*"):
        try:
            f.unlink()
        except Exception:
            pass

    video_path = _get_video_path(out_dir, video_id)
    duration = 0.0
    title = ""

    try:
        import yt_dlp  # type: ignore
        out_template = str(out_dir / f"{video_id}.%(ext)s")
        ydl_opts = {
            "format": "best[ext=mp4][height<=480]/best[height<=480]/best",
            "outtmpl": out_template,
            "quiet": True,
            "no_warnings": True,
            "overwrites": True,
            "nocheckcertificate": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=(video_path is None))
            if info:
                duration = float(info.get("duration", 0.0) or 0.0)
                title = info.get("title", "") or ""

        if not video_path:
            video_path = _get_video_path(out_dir, video_id)
    except Exception as exc:
        logger.warning("yt-dlp error for %s: %s", url, exc)
        video_path = _get_video_path(out_dir, video_id)

    # Fallback to OpenCV if duration not found via yt-dlp
    if duration <= 0 and video_path and video_path.exists():
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                v_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                if frame_count > 0 and v_fps > 0:
                    duration = frame_count / v_fps
            cap.release()
        except Exception:
            pass

    return video_path, duration, title


def _extract_keypoints(
    video_path: Optional[Path],
    video_id: str,
    real_duration_secs: float,
    force_recompute: bool = False,
    progress_callback: Optional[Any] = None,
):
    """Run MediaPipe keypoint extraction with cache coverage validation."""
    cached_path = REPO_ROOT / "shared" / "test-videos" / f"{video_id}_keypoints.json"
    if force_recompute and cached_path.exists():
        try:
            cached_path.unlink()
        except Exception:
            pass

    if cached_path.exists():
        try:
            data = json.loads(cached_path.read_text(encoding="utf-8"))
            kps = data.get("frames", data) if isinstance(data, dict) else data
            if kps and len(kps) > 0:
                last_t = kps[-1].get("timestamp", 0.0)
                is_stale = (last_t < 60.0 and real_duration_secs > 120.0)
                # If cached frames cover at least 70% of video duration or duration is unknown
                if not is_stale and (real_duration_secs <= 0 or last_t >= real_duration_secs * 0.7):
                    logger.info("Using cached keypoints for %s (%d frames, covers %.1fs of %.1fs)", video_id, len(kps), last_t, real_duration_secs)
                    if progress_callback:
                        progress_callback(100, 100)
                    return kps, 2.0
                else:
                    logger.info("Cached keypoints for %s are partial (covers %.1fs of %.1fs). Re-extracting full video.", video_id, last_t, real_duration_secs)
        except Exception as exc:
            logger.warning("Cache load failed for %s: %s", video_id, exc)

    # Run extraction across full video at 2.0 fps
    try:
        from cv_pipeline.keypoints.extract_keypoints import extract_keypoints  # type: ignore
        if video_path and video_path.exists():
            logger.info("Extracting MediaPipe keypoints for full video %s (duration: %.1fs)...", video_id, real_duration_secs)
            kps = extract_keypoints(str(video_path), sample_fps=2.0, progress_callback=progress_callback)
            if kps:
                try:
                    cached_path.write_text(json.dumps(kps), encoding="utf-8")
                except Exception:
                    pass
                return kps, 2.0
    except Exception as exc:
        logger.warning("Keypoint extraction failed: %s", exc)

    return [], 2.0


def _run_lstm_pipeline(keypoints: list, real_duration: float, weight_kg: float, tier: str) -> list:
    """Run motion segmentation followed by PyTorch LSTM classification (biomechanical features, run-resampling)."""
    try:
        from cv_pipeline.motion.motion_filter import segment_runs  # type: ignore
        from cv_pipeline.models.lstm_classifier import LSTMClassifier, classify_run  # type: ignore

        if not keypoints:
            return []

        # 1. Segment active runs — pass sample_fps so threshold scales correctly
        active_runs = segment_runs(
            keypoints,
            threshold=0.01,
            min_active_secs=2.0,
            min_gap_secs=3.0,
            sample_fps=2.0,  # we sample at 2fps in _extract_keypoints
        )
        
        # 2. Load the LSTM model
        model_path = REPO_ROOT / "cv_pipeline" / "models" / "lstm_best.pt"
        if not model_path.exists():
            logger.error("LSTM model not found at %s", model_path)
            return []
            
        model = LSTMClassifier.from_checkpoint(model_path)
        model.eval()
        
        # 3. Classify each active run with classify_run() — uses resampling, no zero-padding
        segs = []
        last_end = 0.0
        
        for run in active_runs:
            run_frames = run["frames"]
            st = run["start_time"]
            et = run["end_time"]
            
            run_label, run_conf = classify_run(run_frames, model)
                
            # Fill gap with rest if any
            if st - last_end > 0.5:
                dur = round(st - last_end, 1)
                segs.append({
                    "start_secs": round(last_end, 1),
                    "end_secs": round(st, 1),
                    "duration_secs": dur,
                    "exercise": "rest",
                    "confidence": 1.0,
                    "calories": _calories("rest", dur, weight_kg, tier),
                })
                
            dur = round(et - st, 1)
            segs.append({
                "start_secs": round(st, 1),
                "end_secs": round(et, 1),
                "duration_secs": dur,
                "exercise": run_label,
                "confidence": round(run_conf, 2),
                "calories": _calories(run_label, dur, weight_kg, tier),
            })
            last_end = et
            
        # End padding with rest
        if real_duration > last_end + 1.0:
            tail_dur = round(real_duration - last_end, 1)
            segs.append({
                "start_secs": round(last_end, 1),
                "end_secs": round(real_duration, 1),
                "duration_secs": tail_dur,
                "exercise": "rest",
                "confidence": 1.0,
                "calories": _calories("rest", tail_dur, weight_kg, tier),
            })
            
        return segs

    except Exception as exc:
        logger.warning("LSTM inference error: %s", exc)
        return []
