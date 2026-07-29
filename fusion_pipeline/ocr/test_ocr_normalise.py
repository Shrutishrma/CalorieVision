"""
fusion_pipeline/ocr/test_ocr_normalise.py
────────────────────────────────────────────────────────────────────────────────
Unit tests for ocr_normalise.py.
"""

import pytest
from fusion_pipeline.ocr.ocr_normalise import normalise_ocr_text, EXERCISE_VOCABULARY


@pytest.mark.parametrize("input_text,expected", [
    ("30 SQUATS", "squat"),
    ("squats", "squat"),
    ("PUSH-UPS x15", "pushup"),
    ("Push Ups", "pushup"),
    ("Jumping Jacks 0:45", "jumping_jack"),
    ("planks", "plank"),
    ("Plank Hold", "plank"),
    ("Burpees 10 reps", "burpee"),
    ("Mountain Climbers", "mountain_climber"),
    ("High Knees", "high_knees"),
    ("Sit-Ups", "situp"),
    ("crunches", "situp"),
    ("Jump Rope 1 min", "jump_rope"),
    ("skipping rope", "jump_rope"),
    ("Bicycle Crunches", "bicycle_crunch"),
    ("Shoulder Press", "shoulder_press"),
    ("Random Text 123", "unknown"),
    ("", "unknown"),
])
def test_normalise_ocr_text(input_text, expected):
    assert normalise_ocr_text(input_text) == expected
