"""
tests/test_placeholder.py
─────────────────────────────────────────────────────────────────────────────
Test suite for CalorieVision schema contracts and FastAPI backend routes.
"""

import sys
import os
import pytest

# Make sure the repo root is on sys.path so `shared` is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.schemas import Segment, SegmentListResponse, Source


# ─── Sanity ───────────────────────────────────────────────────────────────────

def test_placeholder_always_passes():
    """CI bootstrap smoke test — always passes."""
    assert True


# ─── Schema: Unified Segment (pydantic.dataclasses) ──────────────────────────

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
    assert Source.scene_cut.value == "scene_cut"


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


def test_segment_end_before_start_raises():
    with pytest.raises(ValueError, match="end_time must be greater than start_time"):
        Segment(
            segment_id="seg_bad",
            start_time=10.0,
            end_time=5.0,   # end < start → should raise ValueError
            label="invalid",
            confidence=0.5,
            source=Source.pose,
        )


def test_segment_confidence_bounds():
    with pytest.raises(Exception):
        Segment(
            segment_id="seg_bad_conf",
            start_time=0.0,
            end_time=1.0,
            label="squat",
            confidence=1.5,  # out of bounds [0.0, 1.0]
            source=Source.pose,
        )


# ─── FastAPI endpoints ─────────────────────────────────────────────────────────

def test_health_endpoint():
    """Smoke-test the /health endpoint without a live server."""
    from fastapi.testclient import TestClient
    from app.backend.main import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_validate_segments_endpoint():
    """Verify FastAPI request/response validation directly using Segment class."""
    from fastapi.testclient import TestClient
    from app.backend.main import app

    client = TestClient(app)
    payload = [
        {
            "segment_id": "seg_001",
            "start_time": 0.0,
            "end_time": 10.0,
            "label": "squat",
            "confidence": 0.95,
            "source": "pose"
        }
    ]
    response = client.post("/validate-segments", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["segments"][0]["label"] == "squat"
    assert data["segments"][0]["source"] == "pose"
