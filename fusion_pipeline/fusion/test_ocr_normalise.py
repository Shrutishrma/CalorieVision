"""
fusion_pipeline/fusion/test_ocr_normalise.py
────────────────────────────────────────────────────────────────────────────────
Unit tests for ocr_normalise.py.
"""

from __future__ import annotations

import pytest
from fusion_pipeline.fusion.ocr_normalise import ocr_normalise, VOCABULARY


class TestOCRNormalise:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("30 SQUATS", "squat"),
            ("SQUAT", "squat"),
            ("10 Push Ups", "pushup"),
            ("PUSH-UP", "pushup"),
            ("50 PUSHUPS", "pushup"),
            ("JUMPING JACKS", "jumping_jack"),
            ("Jumping Jack 30s", "jumping_jack"),
            ("LUNGES", "lunge"),
            ("PLANK 60 SEC", "plank"),
            ("BURPEES", "burpee"),
            ("Mountain Climbers", "mountain_climber"),
            ("High Knees 45s", "high_knees"),
            ("Sit-ups", "situp"),
            ("CRUNCHES", "situp"),
            ("Jump Rope", "jump_rope"),
            ("Bicycle Crunches", "bicycle_crunch"),
            ("Shoulder Press", "shoulder_press"),
            ("Overhead Press", "shoulder_press"),
        ],
    )
    def test_vocabulary_mapping(self, raw: str, expected: str):
        assert ocr_normalise(raw) == expected

    def test_empty_returns_unknown(self):
        assert ocr_normalise("") == "unknown"
        assert ocr_normalise(None) == "unknown"

    def test_unmatched_returns_lowercased(self):
        assert ocr_normalise("random text 123") == "random text 123"
