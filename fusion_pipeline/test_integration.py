"""
fusion_pipeline/test_integration.py
────────────────────────────────────────────────────────────────────────────────
End-to-end integration test for the CalorieVision pipeline:
  scene_detect → motion_filter → ocr_detector → classifier → fusion → calorie

Verifies that the entire sequence executes cleanly and produces output matching
the canonical Segment and CalorieReport contracts in shared/schemas.py and
fusion_pipeline/fusion/calorie.py.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.schemas import Segment, Source
from fusion_pipeline.segmentation.scene_detect import detect_scenes
from fusion_pipeline.fusion.fusion import fuse_segments
from fusion_pipeline.fusion.calorie import estimate_calories, CalorieReport
from cv_pipeline.models.baseline import MajorityClassPredictor
from cv_pipeline.motion.motion_filter import filter_active


# ─── Integration test helper ──────────────────────────────────────────────────

def _create_synthetic_video(path: Path, duration_sec: float = 2.0, fps: int = 30) -> None:
    """Create a temporary MP4 video with a hard scene cut halfway through."""
    import cv2
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (320, 240))

    total_frames = int(duration_sec * fps)
    halfway = total_frames // 2

    for f in range(total_frames):
        # Black frame for first half, White frame for second half (hard cut)
        val = 0 if f < halfway else 255
        frame = np.full((240, 320, 3), val, dtype=np.uint8)
        writer.write(frame)

    writer.release()


# ─── End-to-End Pipeline Test ─────────────────────────────────────────────────

class TestPipelineIntegration:
    @pytest.fixture
    def test_video(self, tmp_path: Path) -> Path:
        video_path = tmp_path / "integration_test_sample.mp4"
        _create_synthetic_video(video_path, duration_sec=3.0, fps=30)
        return video_path

    def test_full_pipeline_flow(self, test_video: Path, tmp_path: Path):
        """
        Run synthetic video through:
        1. scene_detect ( PySceneDetect hard cuts )
        2. motion_filter ( active/rest filtering )
        3. classifier stand-in ( MajorityClassPredictor )
        4. fusion ( arbitration + ocr_normalise )
        5. calorie ( MET-based calculation )
        """
        # Step 1: Scene Detection
        scenes = detect_scenes(str(test_video), threshold=20.0, min_scene_len_frames=10)
        assert isinstance(scenes, list)
        # Should detect at least 1 scene boundary
        assert len(scenes) >= 1

        # Step 2: Motion Filter Stand-in (Synthetic frame scores)
        synthetic_frames = [
            {"frame_index": i, "landmarks": [{"x": 0.5 + (i * 0.01), "y": 0.5, "z": 0.0}] * 33}
            for i in range(90)
        ]
        active_frames, active_indices = filter_active(synthetic_frames, threshold=0.0001)
        assert isinstance(active_frames, list)
        assert isinstance(active_indices, list)

        # Step 3: Classifier Stand-in (MajorityClassPredictor)
        clf = MajorityClassPredictor()
        clf.fit([{"landmarks": [{"x": 0.5, "y": 0.5, "z": 0.0}] * 33}], ["squat"])

        pose_segments = []
        for idx, scene in enumerate(scenes):
            pred_label = clf.predict([synthetic_frames[0]])[0]
            pose_segments.append(
                Segment(
                    segment_id=f"pose_{idx:04d}",
                    start_time=scene.start_time,
                    end_time=scene.end_time,
                    label=pred_label,
                    confidence=0.85,
                    source=Source.pose,
                )
            )

        # Step 4: OCR Stand-in (Simulated OCR segment with raw text)
        ocr_segments = [
            Segment(
                segment_id="ocr_0001",
                start_time=0.0,
                end_time=1.5,
                label="30 SQUATS",  # raw string requiring ocr_normalise
                confidence=0.90,
                source=Source.ocr,
            )
        ]

        # Step 5: Fusion Layer
        log_path = tmp_path / "eval" / "failure_log.jsonl"
        fused_segments = fuse_segments(pose_segments, ocr_segments, log_path=log_path)

        assert len(fused_segments) == len(pose_segments)
        for s in fused_segments:
            assert isinstance(s, Segment)
            assert s.segment_id.startswith("fused_")
            assert s.confidence >= 0.0 and s.confidence <= 1.0
            assert isinstance(s.source, Source)

        # Step 6: Calorie Estimation
        calorie_report = estimate_calories(fused_segments, weight_kg=70.0)
        assert isinstance(calorie_report, CalorieReport)
        assert calorie_report.weight_kg == 70.0
        assert "beginner" in calorie_report.total_kcal
        assert "intermediate" in calorie_report.total_kcal
        assert "advanced" in calorie_report.total_kcal
        assert len(calorie_report.segments) == len(fused_segments)

        # Step 7: Output JSON Contract Verification
        report_dict = calorie_report.to_dict()
        report_json = json.dumps(report_dict)
        parsed = json.loads(report_json)

        assert "weight_kg" in parsed
        assert "total_kcal" in parsed
        assert "total_segments" in parsed
        assert "segments" in parsed
        assert parsed["total_segments"] == len(fused_segments)
