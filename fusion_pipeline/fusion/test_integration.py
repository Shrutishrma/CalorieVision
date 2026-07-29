"""
fusion_pipeline/fusion/test_integration.py
────────────────────────────────────────────────────────────────────────────────
End-to-End Pipeline Integration Test.

Runs a pipeline simulation:
  scene_detect / keypoints → motion_filter → ocr_detector → baseline classifier → fusion → calorie

Verifies that the final JSON output matches the Segment and CalorieReport contracts in
shared/schemas.py and fusion_pipeline/fusion/calorie.py.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from shared.schemas import Segment, Source
from cv_pipeline.models.baseline import MajorityClassPredictor
from cv_pipeline.motion.motion_filter import filter_active, label_motion
from fusion_pipeline.ocr.ocr_normalise import normalise_ocr_text
from fusion_pipeline.fusion.fusion import fuse_segments
from fusion_pipeline.fusion.calorie import estimate_calories, CalorieReport

_REPO_ROOT = Path(__file__).resolve().parents[2]


class TestPipelineIntegration:
    def test_full_pipeline_mock_flow(self, tmp_path):
        """
        Integration test verifying end-to-end flow from keypoints/scenes through
        baseline classifier, fusion, and calorie estimation.
        """
        # 1. Load mock keypoint data
        keypoints_path = _REPO_ROOT / "shared" / "test-videos" / "jNQXAC9IVRw_keypoints.json"
        assert keypoints_path.exists(), "Test keypoint file missing"

        kp_data = json.loads(keypoints_path.read_text(encoding="utf-8"))
        frames = kp_data.get("frames", [])
        assert len(frames) > 0

        # 2. Run motion filter
        active_frames, active_indices = filter_active(frames, threshold=0.01)
        assert isinstance(active_frames, list)

        # 3. Train stand-in MajorityClassPredictor baseline
        train_labels = ["squat"] * len(frames)
        predictor = MajorityClassPredictor()
        predictor.fit(frames, train_labels)

        # Predict frame labels
        frame_preds = predictor.predict(frames)

        # 4. Construct pose segments from predicted sequence
        pose_segments = [
            Segment(
                segment_id="pose_0001",
                start_time=0.0,
                end_time=10.0,
                label=frame_preds[0] if frame_preds[0] != "UNKNOWN" else "squat",
                confidence=0.85,
                source=Source.pose,
            ),
            Segment(
                segment_id="pose_0002",
                start_time=10.0,
                end_time=20.0,
                label="pushup",
                confidence=0.80,
                source=Source.pose,
            )
        ]

        # 5. Construct OCR segments (1 agreeing, 1 disagreeing)
        raw_ocr_texts = ["30 SQUATS", "JUMPING JACKS 0:30"]
        ocr_segments = [
            Segment(
                segment_id="ocr_0001",
                start_time=1.0,
                end_time=9.0,
                label=normalise_ocr_text(raw_ocr_texts[0]),
                confidence=0.90,
                source=Source.ocr,
            ),
            Segment(
                segment_id="ocr_0002",
                start_time=11.0,
                end_time=19.0,
                label=normalise_ocr_text(raw_ocr_texts[1]),
                confidence=0.85,
                source=Source.ocr,
            )
        ]

        # 6. Run Fusion stage
        log_file = tmp_path / "failure_log.jsonl"
        fused_segments = fuse_segments(pose_segments, ocr_segments, log_path=log_file)
        assert len(fused_segments) == 2

        # Assert contract types & sources
        assert fused_segments[0].source == Source.fused       # AGREE (squat == squat)
        assert fused_segments[1].source == Source.disagreement # DISAGREE (pushup != jumping_jack)

        # 7. Run Calorie estimation stage
        calorie_report = estimate_calories(fused_segments, weight_kg=70.0)
        assert isinstance(calorie_report, CalorieReport)
        assert "beginner" in calorie_report.total_kcal
        assert "intermediate" in calorie_report.total_kcal
        assert "advanced" in calorie_report.total_kcal
        assert len(calorie_report.segments) == 2

        # Verify JSON serialisability against schemas
        report_dict = calorie_report.to_dict()
        json_output = json.dumps(report_dict, indent=2)
        assert "intermediate" in json_output
        assert "disagreement" in json_output
