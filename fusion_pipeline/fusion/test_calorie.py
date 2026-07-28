"""
fusion_pipeline/fusion/test_calorie.py
────────────────────────────────────────────────────────────────────────────────
Unit tests for calorie.py.

All tests use synthetic Segment objects — no video or ML needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.schemas import Segment, Source
from fusion_pipeline.fusion.calorie import (
    MET_TABLE,
    TIERS,
    DEFAULT_WEIGHT_KG,
    CalorieReport,
    SegmentCalorie,
    estimate_calories,
    kcal_for_segment,
    main,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _seg(label: str, start: float, end: float, conf: float = 0.9) -> Segment:
    return Segment(
        segment_id = f"fused_{int(start*10):04d}",
        start_time = start,
        end_time   = end,
        label      = label,
        confidence = conf,
        source     = Source.fused,
    )


# ─── MET table ────────────────────────────────────────────────────────────────

class TestMETTable:
    def test_all_12_classes_present(self):
        expected = {
            "squat", "pushup", "jumping_jack", "lunge", "plank",
            "burpee", "mountain_climber", "high_knees", "situp",
            "jump_rope", "bicycle_crunch", "shoulder_press",
        }
        for cls in expected:
            assert cls in MET_TABLE, f"Missing MET entry: {cls}"

    def test_all_tiers_present_per_class(self):
        for cls, row in MET_TABLE.items():
            if cls == "unknown":
                continue
            for tier in TIERS:
                assert tier in row, f"{cls} missing tier {tier}"

    def test_met_values_positive(self):
        for cls, row in MET_TABLE.items():
            for tier, val in row.items():
                assert val > 0, f"{cls}/{tier} has non-positive MET={val}"

    def test_advanced_gte_beginner(self):
        """Advanced intensity should never be less than Beginner."""
        for cls, row in MET_TABLE.items():
            assert row["advanced"] >= row["beginner"], (
                f"{cls}: advanced MET ({row['advanced']}) < beginner MET ({row['beginner']})"
            )


# ─── kcal_for_segment ─────────────────────────────────────────────────────────

class TestKcalForSegment:
    def test_formula_correct(self):
        # MET=5.0, weight=60, 1 hour → exactly 300 kcal
        result = kcal_for_segment("squat", 3600.0, 60.0, "intermediate")
        expected = MET_TABLE["squat"]["intermediate"] * 60.0 * 1.0
        assert result == pytest.approx(expected, rel=1e-4)

    def test_zero_duration_returns_zero(self):
        assert kcal_for_segment("squat", 0.0, 70.0, "beginner") == pytest.approx(0.0)

    def test_unknown_label_uses_fallback(self):
        result = kcal_for_segment("unknown", 3600.0, 70.0, "intermediate")
        assert result > 0.0

    def test_unrecognised_label_uses_unknown_fallback(self):
        result = kcal_for_segment("random_exercise_xyz", 3600.0, 70.0, "intermediate")
        expected = kcal_for_segment("unknown", 3600.0, 70.0, "intermediate")
        assert result == pytest.approx(expected)

    def test_invalid_tier_raises(self):
        with pytest.raises(ValueError):
            kcal_for_segment("squat", 60.0, 70.0, "elite")

    def test_advanced_burns_more_than_beginner(self):
        beg = kcal_for_segment("burpee", 600.0, 70.0, "beginner")
        adv = kcal_for_segment("burpee", 600.0, 70.0, "advanced")
        assert adv > beg

    def test_heavier_person_burns_more(self):
        light = kcal_for_segment("squat", 600.0, 60.0,  "intermediate")
        heavy = kcal_for_segment("squat", 600.0, 100.0, "intermediate")
        assert heavy > light

    @pytest.mark.parametrize("cls", [
        "squat", "pushup", "jumping_jack", "lunge", "plank",
        "burpee", "mountain_climber", "high_knees", "situp",
        "jump_rope", "bicycle_crunch", "shoulder_press",
    ])
    def test_all_classes_return_positive_kcal(self, cls):
        result = kcal_for_segment(cls, 300.0, 70.0, "intermediate")
        assert result > 0.0


# ─── estimate_calories ────────────────────────────────────────────────────────

class TestEstimateCalories:
    def test_returns_calorie_report(self):
        segs = [_seg("squat", 0, 60)]
        report = estimate_calories(segs, weight_kg=70.0)
        assert isinstance(report, CalorieReport)

    def test_total_kcal_has_all_tiers(self):
        segs = [_seg("squat", 0, 60)]
        report = estimate_calories(segs)
        assert set(report.total_kcal.keys()) == set(TIERS)

    def test_segment_count_matches_input(self):
        segs = [_seg("squat", 0, 30), _seg("pushup", 30, 60)]
        report = estimate_calories(segs)
        assert len(report.segments) == 2

    def test_total_equals_sum_of_segments(self):
        segs = [_seg("squat", 0, 1800), _seg("burpee", 1800, 3600)]
        report = estimate_calories(segs, weight_kg=70.0)
        for tier in TIERS:
            seg_sum = sum(s.kcal[tier] for s in report.segments)
            assert report.total_kcal[tier] == pytest.approx(seg_sum, rel=1e-3)

    def test_empty_segments_returns_zeros(self):
        report = estimate_calories([], weight_kg=70.0)
        assert len(report.segments) == 0
        for tier in TIERS:
            assert report.total_kcal[tier] == pytest.approx(0.0)

    def test_weight_affects_total(self):
        segs = [_seg("squat", 0, 3600)]
        light = estimate_calories(segs, weight_kg=60.0)
        heavy = estimate_calories(segs, weight_kg=90.0)
        for tier in TIERS:
            assert heavy.total_kcal[tier] > light.total_kcal[tier]

    def test_to_dict_has_required_keys(self):
        segs = [_seg("squat", 0, 60)]
        report = estimate_calories(segs)
        d = report.to_dict()
        assert {"weight_kg", "total_kcal", "total_segments", "segments"}.issubset(d.keys())

    def test_default_weight_is_70(self):
        report = estimate_calories([])
        assert report.weight_kg == DEFAULT_WEIGHT_KG


# ─── CLI ──────────────────────────────────────────────────────────────────────

class TestCalorieCLI:
    def test_cli_writes_json(self, tmp_path):
        # Create a minimal fused segments JSON
        segs = [_seg("squat", 0, 60), _seg("pushup", 60, 120)]
        fused_path = tmp_path / "fused.json"
        fused_path.write_text(
            json.dumps({"segments": [s.to_dict() for s in segs]}),
            encoding="utf-8",
        )
        out = tmp_path / "calories.json"
        main([str(fused_path), "--weight", "70", "--out", str(out)])
        assert out.exists()
        data = json.loads(out.read_text())
        assert "total_kcal" in data
        assert "segments" in data

    def test_cli_missing_file_raises(self):
        with pytest.raises((FileNotFoundError, SystemExit)):
            main(["/no/such/file.json"])
