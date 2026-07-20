"""
shared/schemas.py
─────────────────────────────────────────────────────────────────────────────
Defines the canonical data contract passed between every stage of the
CalorieVision pipeline.

Two representations are provided:
  • Segment        — plain Python dataclass (use inside pipeline code)
  • SegmentModel   — Pydantic BaseModel (use in FastAPI request/response bodies)

Both expose the same fields so conversion is trivial:
    model = SegmentModel(**asdict(segment))
    segment = Segment(**model.dict())
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import List


# ─── Source enum ─────────────────────────────────────────────────────────────

class Source(str, Enum):
    """Which pipeline stage produced this segment."""
    pose  = "pose"   # MediaPipe pose estimator
    ocr   = "ocr"    # EasyOCR text detector
    fused = "fused"  # Fusion layer (merged / arbitrated)


# ─── Dataclass (pipeline-internal) ───────────────────────────────────────────

@dataclass
class Segment:
    """
    One recognised exercise segment within a video.

    Attributes
    ----------
    segment_id : str
        Unique identifier (e.g. "seg_0042").
    start_time : float
        Segment start in seconds from the beginning of the video.
    end_time : float
        Segment end in seconds.
    label : str
        Human-readable exercise label, e.g. "squat", "pushup".
    confidence : float
        Model confidence in [0.0, 1.0].
    source : Source
        Which pipeline stage produced this segment.
    """
    segment_id: str
    start_time: float
    end_time:   float
    label:      str
    confidence: float
    source:     Source

    # ── helpers ──────────────────────────────────────────────────────────────

    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return self.end_time - self.start_time

    def to_dict(self) -> dict:
        """Serialise to a plain dict (JSON-safe)."""
        d = asdict(self)
        d["source"] = self.source.value  # enum → string
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Segment":
        """Deserialise from a plain dict."""
        data = dict(data)
        data["source"] = Source(data["source"])
        return cls(**data)


# ─── Pydantic model (FastAPI serialisation) ───────────────────────────────────

try:
    from pydantic import BaseModel, Field, field_validator

    class SegmentModel(BaseModel):
        """
        Pydantic mirror of Segment — use as FastAPI request/response type.

        Example JSON
        ------------
        {
            "segment_id": "seg_0001",
            "start_time": 3.14,
            "end_time":   7.92,
            "label":      "squat",
            "confidence": 0.93,
            "source":     "pose"
        }
        """
        segment_id: str   = Field(..., description="Unique segment identifier")
        start_time: float = Field(..., ge=0.0, description="Start time in seconds")
        end_time:   float = Field(..., ge=0.0, description="End time in seconds")
        label:      str   = Field(..., description="Exercise label")
        confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence")
        source:     Source = Field(..., description="Originating pipeline stage")

        @field_validator("end_time")
        @classmethod
        def end_after_start(cls, v: float, info) -> float:
            start = info.data.get("start_time", 0.0)
            if v <= start:
                raise ValueError("end_time must be greater than start_time")
            return v

        @property
        def duration(self) -> float:
            return self.end_time - self.start_time

        def to_segment(self) -> Segment:
            """Convert to pipeline-internal Segment dataclass."""
            return Segment(**self.model_dump())

        model_config = {"use_enum_values": False}


    class SegmentListResponse(BaseModel):
        """Wrapper returned by endpoints that emit multiple segments."""
        segments: List[SegmentModel]
        total:    int = Field(..., description="Total number of segments")

except ImportError:  # pydantic not installed (pipeline-only environment)
    SegmentModel = None          # type: ignore[assignment,misc]
    SegmentListResponse = None   # type: ignore[assignment,misc]
