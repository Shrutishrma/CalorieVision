"""
tests/test_pipeline_orchestrator.py
────────────────────────────────────────────────────────────────────────────────
Integration unit tests for the central pipeline orchestrator.
"""

import pytest
from shared.schemas import PipelineResult
from fusion_pipeline.pipeline import run_pipeline


def test_pipeline_orchestrator_synthetic():
    res = run_pipeline(
        video_source="jNQXAC9IVRw",
        weight_kg=70.0,
        user_tier="intermediate",
        video_duration_mins=0.5,
        force_recompute=False,
    )

    assert isinstance(res, PipelineResult)
    assert res.video_id == "jNQXAC9IVRw"
    assert res.duration_secs > 0
    assert len(res.pose_segments) > 0
    assert len(res.fused_segments) > 0
    assert res.classifier_used in ("lstm", "knn", "majority_class")
    assert "classification" in res.stage_timings
    assert "fusion" in res.stage_timings
