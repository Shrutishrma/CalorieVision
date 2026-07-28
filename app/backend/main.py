"""
app/backend/main.py
────────────────────────────────────────────────────────────────────────────
CalorieVision FastAPI application entry point.

Run locally:
    uvicorn app.backend.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CalorieVision API",
    description=(
        "AI-based exercise recognition and calorie estimation. "
        "Accepts video uploads or YouTube links and returns a timeline of "
        "recognised exercise segments with calorie estimates."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
# Allow the Vite dev server (localhost:5173) to call the API during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite default
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from shared.schemas import Segment, SegmentListResponse

@app.get(
    "/health",
    tags=["infra"],
    summary="Health check",
    response_description="Service status",
)
async def health() -> dict[str, str]:
    """
    Simple liveness probe.

    Returns `{"status": "ok"}` when the service is up.
    Suitable for use with Docker HEALTHCHECK, Kubernetes probes, or CI smoke tests.
    """
    return {"status": "ok"}


@app.post(
    "/validate-segments",
    response_model=SegmentListResponse,
    tags=["validation"],
    summary="Validate segments schema directly via FastAPI",
)
async def validate_segments(segments: list[Segment]) -> SegmentListResponse:
    """
    Accepts a list of Segment objects directly in request body, validating them
    using Pydantic dataclass validation, and returns a SegmentListResponse.
    """
    return SegmentListResponse(segments=segments, total=len(segments))
