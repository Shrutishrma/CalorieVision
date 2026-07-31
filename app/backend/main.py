"""
app/backend/main.py
────────────────────────────────────────────────────────────────────────────
CalorieVision FastAPI application entry point.

Serves:
  • GET  /health              - Liveness probe
  • GET  /manifest            - Pre-configured test workout catalogue (15 videos)
  • POST /validate-segments  - Schema validation for Segment arrays
  • POST /analyze            - Start end-to-end workout video analysis pipeline
  • GET  /status/{job_id}     - Poll analysis job status and retrieve result
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.schemas import Segment, Source, SegmentListResponse
from fusion_pipeline.pipeline import run_pipeline
from fusion_pipeline.fusion.calorie import MET_TABLE
from app.backend.job_store import job_store, Job

logger = logging.getLogger("calorie_vision_backend")

# ─── App Setup ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="CalorieVision API",
    description=(
        "AI-based exercise recognition and calorie estimation. "
        "Accepts workout videos or YouTube URLs and returns an exercise segment timeline "
        "with 3-tier MET calorie calculation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Schemas ───────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    video_id: Optional[str] = Field(None, description="YouTube ID from manifest catalogue")
    video_url: Optional[str] = Field(None, description="Direct YouTube URL")
    weight_kg: float = Field(70.0, ge=30.0, le=250.0, description="User body weight in kg")
    user_tier: str = Field("intermediate", description="Target difficulty tier: beginner, intermediate, advanced")
    video_duration_mins: Optional[float] = Field(None, ge=0.1, le=300.0, description="Video duration in minutes")
    force_recompute: bool = Field(False, description="If True, bypass cached intermediate outputs")


class AnalyzeResponse(BaseModel):
    job_id: str
    status: str
    video_id: str
    video_title: str
    description: str
    tags: Dict[str, Any]
    duration_secs: float
    weight_kg: float
    user_tier: str
    classifier_used: str
    total_kcal: Dict[str, float]
    segments: List[dict]
    exercise_summary: Dict[str, dict]
    failure_events: List[dict]
    stage_timings: Dict[str, float]


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    stage: str
    progress: float
    created_at: str
    updated_at: str
    result: Optional[AnalyzeResponse] = None
    error: Optional[str] = None


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    tags=["infra"],
    summary="Health check probe",
)
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "CalorieVision API", "version": "1.0.0"}


@app.get(
    "/manifest",
    tags=["catalogue"],
    summary="List 15 tagged test workout videos",
)
async def get_manifest() -> List[dict]:
    manifest_path = _REPO_ROOT / "shared" / "test_videos_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Manifest file missing")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return data


@app.post(
    "/validate-segments",
    response_model=SegmentListResponse,
    tags=["validation"],
    summary="Validate Segment array schema",
)
async def validate_segments(segments: list[Segment]) -> SegmentListResponse:
    return SegmentListResponse(segments=segments, total=len(segments))


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    tags=["analysis"],
    summary="Run full exercise recognition and calorie estimation pipeline",
)
async def analyze_workout(
    req: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    sync: bool = Query(True, description="Run synchronously for instant response if set to True")
) -> Any:
    manifest_path = _REPO_ROOT / "shared" / "test_videos_manifest.json"
    manifest_items = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []

    target_id = req.video_id or ""
    if req.video_url and "v=" in req.video_url:
        target_id = req.video_url.split("v=")[-1].split("&")[0]
    elif req.video_url and "youtu.be/" in req.video_url:
        target_id = req.video_url.split("youtu.be/")[-1].split("?")[0]

    video_item = None
    for item in manifest_items:
        if item.get("youtube_id") == target_id:
            video_item = item
            break

    if not video_item:
        video_item = manifest_items[0] if manifest_items else {
            "youtube_id": target_id or "jNQXAC9IVRw",
            "url": req.video_url or "https://youtu.be/jNQXAC9IVRw",
            "label": "workout_session",
            "description": "Custom Workout Video Analysis",
            "tags": {"camera_angle": "single", "caption_present": True, "num_subjects": "single"}
        }

    video_source = req.video_url or target_id or video_item.get("url", "")
    job = job_store.create_job(req.model_dump())

    if sync:
        # Synchronous execution
        res_dict = _execute_pipeline_job(job.job_id, video_source, req, video_item)
        return res_dict
    else:
        # Async background execution
        background_tasks.add_task(_execute_pipeline_job, job.job_id, video_source, req, video_item)
        return {
            "job_id": job.job_id,
            "status": "processing",
            "video_id": video_item.get("youtube_id", "demo"),
            "video_title": video_item.get("description", video_item.get("label", "Workout Session")),
            "description": video_item.get("description", ""),
            "tags": video_item.get("tags", {}),
            "duration_secs": float(video_item.get("duration_secs", 30)),
            "weight_kg": req.weight_kg,
            "user_tier": req.user_tier,
            "classifier_used": "pending",
            "total_kcal": {"beginner": 0.0, "intermediate": 0.0, "advanced": 0.0},
            "segments": [],
            "exercise_summary": {},
            "failure_events": [],
            "stage_timings": {},
        }


@app.get(
    "/status/{job_id}",
    response_model=JobStatusResponse,
    tags=["analysis"],
    summary="Check status and progress of an analysis job",
)
async def get_job_status(job_id: str) -> JobStatusResponse:
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    res_obj = None
    if job.result:
        res_obj = AnalyzeResponse(**job.result)

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        stage=job.stage,
        progress=job.progress,
        created_at=job.created_at,
        updated_at=job.updated_at,
        result=res_obj,
        error=job.error,
    )


# ─── Internal Job Executor ────────────────────────────────────────────────────

def _execute_pipeline_job(
    job_id: str,
    video_source: str,
    req: AnalyzeRequest,
    video_item: dict,
) -> Dict[str, Any]:
    try:
        def _prog_cb(stage: str, pct: float):
            job_store.update_progress(job_id, stage, pct)

        pipeline_res = run_pipeline(
            video_source=video_source,
            weight_kg=req.weight_kg,
            user_tier=req.user_tier,
            video_duration_mins=req.video_duration_mins,
            force_recompute=req.force_recompute,
            progress_callback=_prog_cb,
        )

        title = video_item.get("description", video_item.get("label", "Workout Session"))
        tags = video_item.get("tags", {})

        # Compute calorie report and summary
        summary = _build_exercise_summary(pipeline_res.fused_segments, pipeline_res.duration_secs, req.weight_kg, req.user_tier)

        total_kcal = {
            "beginner": round(sum(s.duration * _get_met(s.label, "beginner") * req.weight_kg / 3600.0 for s in pipeline_res.fused_segments), 2),
            "intermediate": round(sum(s.duration * _get_met(s.label, "intermediate") * req.weight_kg / 3600.0 for s in pipeline_res.fused_segments), 2),
            "advanced": round(sum(s.duration * _get_met(s.label, "advanced") * req.weight_kg / 3600.0 for s in pipeline_res.fused_segments), 2),
        }

        response_dict = {
            "job_id": job_id,
            "status": "completed",
            "video_id": pipeline_res.video_id,
            "video_title": title,
            "description": video_item.get("description", ""),
            "tags": tags,
            "duration_secs": round(pipeline_res.duration_secs, 1),
            "weight_kg": req.weight_kg,
            "user_tier": req.user_tier,
            "classifier_used": pipeline_res.classifier_used,
            "total_kcal": total_kcal,
            "segments": [s.to_dict() for s in pipeline_res.fused_segments],
            "exercise_summary": summary,
            "failure_events": pipeline_res.failure_events,
            "stage_timings": pipeline_res.stage_timings,
        }

        job_store.complete_job(job_id, response_dict)
        return response_dict

    except Exception as exc:
        logger.error("Job %s failed: %s", job_id, exc, exc_info=True)
        job_store.fail_job(job_id, str(exc))
        raise exc


def _get_met(label: str, tier: str) -> float:
    row = MET_TABLE.get(label.lower(), MET_TABLE["unknown"])
    return row.get(tier, 5.0)


def _build_exercise_summary(segments: List[Segment], duration_secs: float, weight_kg: float, tier: str) -> dict:
    summary = {}
    for s in segments:
        label = s.label
        if label not in summary:
            summary[label] = {"total_duration_secs": 0.0, "total_kcal": 0.0, "count": 0}
        dur = s.duration
        met = _get_met(label, tier)
        kcal = dur * met * weight_kg / 3600.0
        summary[label]["total_duration_secs"] = round(summary[label]["total_duration_secs"] + dur, 1)
        summary[label]["total_kcal"] = round(summary[label]["total_kcal"] + kcal, 2)
        summary[label]["count"] += 1
    return summary
