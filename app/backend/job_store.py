"""
app/backend/job_store.py
─────────────────────────────────────────────────────────────────────────────
In-memory Job Store for tracking async pipeline analysis jobs.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class Job:
    job_id: str
    status: str  # "queued" | "processing" | "completed" | "failed"
    stage: str
    progress: float  # 0.0 to 1.0
    created_at: str
    updated_at: str
    request_params: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "request_params": self.request_params,
            "result": self.result,
            "error": self.error,
        }


class JobStore:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}

    def create_job(self, request_params: Dict[str, Any]) -> Job:
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()
        job = Job(
            job_id=job_id,
            status="queued",
            stage="Queued",
            progress=0.0,
            created_at=now,
            updated_at=now,
            request_params=request_params,
        )
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def update_progress(self, job_id: str, stage: str, progress: float):
        job = self._jobs.get(job_id)
        if job:
            job.status = "processing"
            job.stage = stage
            job.progress = round(progress, 2)
            job.updated_at = datetime.now(timezone.utc).isoformat()

    def complete_job(self, job_id: str, result: Dict[str, Any]):
        job = self._jobs.get(job_id)
        if job:
            job.status = "completed"
            job.stage = "Completed"
            job.progress = 1.0
            job.result = result
            job.updated_at = datetime.now(timezone.utc).isoformat()

    def fail_job(self, job_id: str, error_msg: str):
        job = self._jobs.get(job_id)
        if job:
            job.status = "failed"
            job.stage = "Failed"
            job.error = error_msg
            job.updated_at = datetime.now(timezone.utc).isoformat()


# Global singleton instance
job_store = JobStore()
