"""
shared/schemas.py
─────────────────────────────────────────────────────────────────────────────
Defines the canonical data contract passed between every stage of the
CalorieVision pipeline and FastAPI serialization.

Using `pydantic.dataclasses.dataclass`, `Segment` acts as both a fast, normal
Python dataclass for pipeline internals and a fully validated Pydantic type
for FastAPI request/response bodies.
"""

from __future__ import annotations

from dataclasses import asdict
from enum import Enum
from typing import List
from pydantic import Field, BaseModel
from pydantic.dataclasses import dataclass


# ─── Source enum ─────────────────────────────────────────────────────────────

class Source(str, Enum):
    """Which pipeline stage produced this segment."""
    pose         = "pose"        # MediaPipe pose estimator
    ocr          = "ocr"         # EasyOCR text detector
    fused        = "fused"       # Fusion layer (merged / arbitrated)
    scene_cut    = "scene_cut"   # PySceneDetect raw scene boundary
    disagreement = "disagreement"# Pose and OCR predictions disagreed in fusion layer


# ─── Unified Segment Schema (pydantic.dataclass) ─────────────────────────────

@dataclass(config={"use_enum_values": False})
class Segment:
    """
    One recognised exercise segment within a video. Serves as both internal
    pipeline data structure and validated FastAPI request/response schema.

    Attributes
    ----------
    segment_id : str
        Unique identifier (e.g. "seg_0042").
    start_time : float
        Segment start in seconds from the beginning of the video.
    end_time : float
        Segment end in seconds (must be strictly greater than start_time).
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

    def __post_init__(self) -> None:
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")

    # ── helpers ──────────────────────────────────────────────────────────────

    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return self.end_time - self.start_time

    def to_dict(self) -> dict:
        """Serialise to a plain dict (JSON-safe)."""
        d = asdict(self)
        d["source"] = self.source.value if isinstance(self.source, Source) else self.source
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Segment":
        """Deserialise from a plain dict."""
        data = dict(data)
        data["source"] = Source(data["source"])
        return cls(**data)


# ─── Response Wrappers ───────────────────────────────────────────────────────

class SegmentListResponse(BaseModel):
    """Wrapper returned by endpoints that emit multiple segments."""
    segments: List[Segment]
    total:    int = Field(..., description="Total number of segments")
