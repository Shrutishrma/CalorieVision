"""
tests/test_placeholder.py
─────────────────────────────────────────────────────────────────────────────
Placeholder test suite for CalorieVision CI bootstrap.

Includes:
  • A trivial sanity-check that always passes (CI smoke test)
  • Schema contract tests for shared/schemas.py
"""

import sys
import os

# Make sure the repo root is on sys.path so `shared` is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.schemas import Segment, Source


# ─── Sanity ───────────────────────────────────────────────────────────────────

def test_placeholder_always_passes():
    """CI bootstrap smoke test — always passes."""
    assert True


# ─── Schema: Segment dataclass ────────────────────────────────────────────────

def test_segment_creation():
    seg = Segment(
        segment_id="seg_0001",
        start_time=0.0,
        end_time=5.5,
        label="squat",
        confidence=0.92,
        source=Source.pose,
    )
    assert seg.segment_id == "seg_0001"
    assert seg.duration == 5.5
    assert seg.source == Source.pose


def test_segment_source_enum_values():
    assert Source.pose.value == "pose"
    assert Source.ocr.value == "ocr"
    assert Source.fused.value == "fused"


def test_segment_serialise_roundtrip():
    original = Segment(
        segment_id="seg_0002",
        start_time=1.0,
        end_time=4.0,
        label="pushup",
        confidence=0.87,
        source=Source.fused,
    )
    restored = Segment.from_dict(original.to_dict())
    assert restored == original


# ─── Schema: Pydantic SegmentModel ────────────────────────────────────────────

def test_pydantic_model_available():
    from shared.schemas import SegmentModel
    assert SegmentModel is not None


def test_pydantic_model_creation():
    from shared.schemas import SegmentModel
    model = SegmentModel(
        segment_id="seg_0003",
        start_time=2.0,
        end_time=6.0,
        label="lunge",
        confidence=0.78,
        source=Source.ocr,
    )
    assert model.label == "lunge"
    assert model.duration == 4.0


def test_pydantic_model_end_before_start_raises():
    from shared.schemas import SegmentModel
    import pytest
    with pytest.raises(Exception):
        SegmentModel(
            segment_id="seg_bad",
            start_time=10.0,
            end_time=5.0,   # end < start → should raise
            label="invalid",
            confidence=0.5,
            source=Source.pose,
        )


# ─── FastAPI health endpoint ───────────────────────────────────────────────────

def test_health_endpoint():
    """Smoke-test the /health endpoint without a live server."""
    from fastapi.testclient import TestClient
    from app.backend.main import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
