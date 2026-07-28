"""
fusion_pipeline/fusion/test_fusion.py
────────────────────────────────────────────────────────────────────────────────
Unit tests for fusion.py and duration.py.

All tests use synthetic Segment objects — no real video required.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.schemas import Segment, Source
from fusion_pipeline.fusion.fusion import fuse_segments, _overlap, configure_logger
from fusion_pipeline.fusion.duration import (
    duration_seconds, annotate_durations, _parse_timer_string,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _seg(
    label: str,
    start: float,
    end: float,
    conf: float = 0.8,
    source: Source = Source.pose,
    seg_id: str | None = None,
) -> Segment:
    return Segment(
        segment_id = seg_id or f"{source.value}_{int(start*10):04d}",
        start_time = start,
        end_time   = end,
        label      = label,
        confidence = conf,
        source     = source,
    )


def _pose(label, start, end, conf=0.8) -> Segment:
    return _seg(label, start, end, conf=conf, source=Source.pose)


def _ocr(label, start, end, conf=0.8) -> Segment:
    return _seg(label, start, end, conf=conf, source=Source.ocr)


# ─── _overlap ─────────────────────────────────────────────────────────────────

class TestOverlap:
    def test_full_overlap(self):
        a = _pose("squat", 0.0, 10.0)
        b = _ocr("squat", 0.0, 10.0)
        assert _overlap(a, b) == pytest.approx(10.0)

    def test_partial_overlap(self):
        a = _pose("squat", 0.0, 6.0)
        b = _ocr("squat", 4.0, 10.0)
        assert _overlap(a, b) == pytest.approx(2.0)

    def test_no_overlap(self):
        a = _pose("squat", 0.0, 5.0)
        b = _ocr("squat", 6.0, 10.0)
        assert _overlap(a, b) == pytest.approx(0.0)

    def test_adjacent_no_overlap(self):
        a = _pose("squat", 0.0, 5.0)
        b = _ocr("squat", 5.0, 10.0)
        assert _overlap(a, b) == pytest.approx(0.0)


# ─── fuse_segments ────────────────────────────────────────────────────────────

class TestFuseSegments:
    def _run(self, pose, ocr, **kw):
        """Helper: run fuse_segments with a temp log file."""
        import logging
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            log_path = Path(f.name)
        try:
            result = fuse_segments(pose, ocr, log_path=log_path, **kw)
        finally:
            # Close all file handlers for this log path to avoid Windows file lock
            logger_name = f"pipeline_run.{hash(str(log_path)) & 0xFFFFFF:06x}"
            lg = logging.getLogger(logger_name)
            for h in lg.handlers[:]:
                h.close()
                lg.removeHandler(h)
            log_path.unlink(missing_ok=True)
        return result

    # Output contract
    def test_returns_list(self):
        result = self._run([_pose("squat", 0, 5)], [])
        assert isinstance(result, list)

    def test_each_item_is_segment(self):
        result = self._run([_pose("squat", 0, 5)], [])
        for s in result:
            assert isinstance(s, Segment)

    def test_segment_ids_prefixed_fused(self):
        result = self._run([_pose("squat", 0, 5)], [])
        for s in result:
            assert s.segment_id.startswith("fused_")

    def test_segment_ids_unique(self):
        pose = [_pose("squat", i * 5.0, i * 5.0 + 5.0) for i in range(5)]
        result = self._run(pose, [])
        ids = [s.segment_id for s in result]
        assert len(ids) == len(set(ids))

    def test_output_count_equals_pose_count(self):
        pose = [_pose("squat", 0, 5), _pose("pushup", 5, 10), _pose("plank", 10, 15)]
        result = self._run(pose, [])
        assert len(result) == 3

    def test_empty_pose_returns_empty(self):
        result = self._run([], [_ocr("squat", 0, 5)])
        assert result == []

    # AGREE path
    def test_agree_gives_fused_source(self):
        pose = [_pose("squat", 0, 10, conf=0.7)]
        ocr  = [_ocr("squat", 2, 8,  conf=0.8)]
        result = self._run(pose, ocr)
        assert result[0].source == Source.fused

    def test_agree_boosts_confidence(self):
        pose = [_pose("squat", 0, 10, conf=0.7)]
        ocr  = [_ocr("squat", 2, 8,  conf=0.8)]
        result = self._run(pose, ocr)
        assert result[0].confidence > 0.7

    # POSE-ONLY path
    def test_no_ocr_keeps_pose_source(self):
        pose = [_pose("squat", 0, 5, conf=0.9)]
        result = self._run(pose, [])
        assert result[0].source == Source.pose
        assert result[0].label == "squat"

    # DISAGREE path
    def test_disagree_keeps_pose_label(self):
        pose = [_pose("squat",  0, 10, conf=0.8)]
        ocr  = [_ocr("pushup", 2,  8, conf=0.75)]
        result = self._run(pose, ocr)
        assert result[0].label == "squat"

    def test_disagree_sets_confidence_050(self):
        pose = [_pose("squat",  0, 10, conf=0.8)]
        ocr  = [_ocr("pushup", 2,  8, conf=0.75)]
        result = self._run(pose, ocr)
        assert result[0].confidence == pytest.approx(0.5)

    def test_disagree_logs_to_file(self):
        """Disagreements must appear in the log file."""
        import logging
        pose = [_pose("squat",  0, 10, conf=0.8)]
        ocr  = [_ocr("pushup", 2,  8, conf=0.75)]
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False, mode="w") as f:
            log_path = Path(f.name)
        try:
            fuse_segments(pose, ocr, log_path=log_path)
            # Flush + close the logger's file handler before reading
            logger_name = f"pipeline_run.{hash(str(log_path)) & 0xFFFFFF:06x}"
            lg = logging.getLogger(logger_name)
            for h in lg.handlers[:]:
                h.flush(); h.close(); lg.removeHandler(h)
            log_content = log_path.read_text(encoding="utf-8")
            assert "DISAGREE" in log_content
        finally:
            log_path.unlink(missing_ok=True)

    def test_low_conf_logs_to_file(self):
        """Low-confidence pose predictions must be logged."""
        import logging
        pose = [_pose("squat", 0, 5, conf=0.3)]  # below LOW_CONF_THRESHOLD=0.5
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False, mode="w") as f:
            log_path = Path(f.name)
        try:
            fuse_segments(pose, [], log_path=log_path)
            # Flush + close the logger's file handler before reading
            logger_name = f"pipeline_run.{hash(str(log_path)) & 0xFFFFFF:06x}"
            lg = logging.getLogger(logger_name)
            for h in lg.handlers[:]:
                h.flush(); h.close(); lg.removeHandler(h)
            log_content = log_path.read_text(encoding="utf-8")
            assert "LOW_CONF" in log_content
        finally:
            log_path.unlink(missing_ok=True)

    def test_confidence_capped_at_one(self):
        pose = [_pose("squat", 0, 10, conf=0.99)]
        ocr  = [_ocr("squat", 0, 10, conf=0.99)]
        result = self._run(pose, ocr)
        assert result[0].confidence <= 1.0

    # Ordering
    def test_output_ordered_by_start_time(self):
        pose = [
            _pose("squat",  5.0, 10.0),
            _pose("pushup", 0.0,  5.0),
        ]
        result = self._run(pose, [])
        starts = [s.start_time for s in result]
        assert starts == sorted(starts)

    def test_pipeline_log_path_env_override(self, tmp_path, monkeypatch):
        """Confirm PIPELINE_LOG_PATH environment variable overrides default log destination."""
        custom_log = tmp_path / "custom_team_pipeline.log"
        monkeypatch.setenv("PIPELINE_LOG_PATH", str(custom_log))
        from fusion_pipeline.fusion.fusion import configure_logger
        lg = configure_logger(None)  # None forces checking env var
        # Check that one of the handlers points to custom_log
        file_paths = [Path(h.baseFilename) for h in lg.handlers if hasattr(h, 'baseFilename')]
        assert custom_log in file_paths
        for h in lg.handlers[:]:
            h.close()
            lg.removeHandler(h)


# ─── _parse_timer_string ──────────────────────────────────────────────────────

class TestParseTimerString:
    @pytest.mark.parametrize("text,expected", [
        ("0:30",       30.0),
        ("1:05",       65.0),
        ("2:15",      135.0),
        ("1:00:00",  3600.0),
        ("30s",        30.0),
        ("45 sec",     45.0),
        ("2min",      120.0),
        ("90seconds",  90.0),
    ])
    def test_valid_timers(self, text, expected):
        result = _parse_timer_string(text)
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize("text", ["squat", "10 reps", "", "pushup x15"])
    def test_no_timer_returns_none(self, text):
        assert _parse_timer_string(text) is None


# ─── duration_seconds ────────────────────────────────────────────────────────

class TestDurationSeconds:
    def test_boundary_source_when_no_ocr(self):
        seg = _pose("squat", 0.0, 10.0)
        dur, source = duration_seconds(seg)
        assert dur == pytest.approx(10.0)
        assert source == "boundary"

    def test_ocr_timer_overrides_boundary(self):
        seg = _pose("squat", 0.0, 10.0)
        dur, source = duration_seconds(seg, ocr_label="0:30")  # 30 sec OCR
        assert dur == pytest.approx(30.0)
        assert source == "ocr_timer"

    def test_bad_ocr_falls_back_to_boundary(self):
        seg = _pose("squat", 0.0, 15.0)
        dur, source = duration_seconds(seg, ocr_label="squat reps")
        assert dur == pytest.approx(15.0)
        assert source == "boundary"

    def test_zero_duration_segment(self):
        # Under unified Pydantic schema (Fix C), degenerate segments with end <= start raise ValueError
        with pytest.raises((ValueError, Exception)):
            _pose("squat", 5.0, 5.0)
        # Smallest valid positive duration works as expected
        seg = _pose("squat", 5.0, 5.001)
        dur, _ = duration_seconds(seg)
        assert dur == pytest.approx(0.001, rel=1e-3)


# ─── annotate_durations ───────────────────────────────────────────────────────

class TestAnnotateDurations:
    def test_returns_list_same_length(self):
        segs = [_pose("squat", 0, 5), _pose("pushup", 5, 10)]
        result = annotate_durations(segs)
        assert len(result) == 2

    def test_each_item_has_required_keys(self):
        segs = [_pose("squat", 0, 5)]
        result = annotate_durations(segs)
        for item in result:
            assert {"segment", "duration_seconds", "duration_source"}.issubset(item.keys())

    def test_boundary_used_when_no_ocr(self):
        segs = [_pose("squat", 0, 8)]
        result = annotate_durations(segs)
        assert result[0]["duration_seconds"] == pytest.approx(8.0)
        assert result[0]["duration_source"] == "boundary"
