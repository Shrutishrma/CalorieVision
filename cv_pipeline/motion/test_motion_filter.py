"""
cv_pipeline/motion/test_motion_filter.py
────────────────────────────────────────────────────────────────────────────────
Unit tests for motion_filter.py.

All tests use synthetic frame dicts — no real video needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cv_pipeline.motion.motion_filter import (
    ACTIVE_LABEL,
    REST_LABEL,
    DEFAULT_THRESHOLD,
    compute_motion_scores,
    filter_active,
    label_motion,
    summarise,
    main,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_landmark(x: float, y: float) -> dict:
    return {"index": 0, "name": "nose", "x": x, "y": y, "z": 0.0, "visibility": 1.0}


def _make_frame(
    frame_index: int,
    x: float = 0.5,
    y: float = 0.5,
    n_landmarks: int = 33,
    has_pose: bool = True,
) -> dict:
    if has_pose:
        landmarks = [_make_landmark(x, y) for _ in range(n_landmarks)]
    else:
        landmarks = []
    return {
        "frame_index": frame_index,
        "timestamp": frame_index / 30.0,
        "pose_detected": has_pose,
        "landmarks": landmarks,
    }


def _still_frames(n: int) -> list[dict]:
    """All landmarks at (0.5, 0.5) — zero motion."""
    return [_make_frame(i, x=0.5, y=0.5) for i in range(n)]


def _moving_frames(n: int, step: float = 0.05) -> list[dict]:
    """Each frame shifts landmark positions by `step` — high motion."""
    return [_make_frame(i, x=i * step % 1.0, y=i * step % 1.0) for i in range(n)]


# ─── compute_motion_scores ────────────────────────────────────────────────────

class TestComputeMotionScores:
    def test_first_frame_always_zero(self):
        frames = _still_frames(5)
        scores = compute_motion_scores(frames)
        assert scores[0] == pytest.approx(0.0)

    def test_still_frames_score_zero(self):
        frames = _still_frames(10)
        scores = compute_motion_scores(frames)
        # Frame 0 is always 0; subsequent frames should also be ~0 for still video
        assert all(s == pytest.approx(0.0) for s in scores)

    def test_moving_frames_score_nonzero(self):
        frames = _moving_frames(10, step=0.1)
        scores = compute_motion_scores(frames)
        # All frames except the first should have positive motion
        assert all(s > 0.0 for s in scores[1:])

    def test_returns_one_score_per_frame(self):
        frames = _still_frames(7)
        scores = compute_motion_scores(frames)
        assert len(scores) == 7

    def test_no_pose_frame_gets_zero(self):
        frames = [
            _make_frame(0, has_pose=True),
            _make_frame(1, has_pose=False),  # no pose → score = 0
            _make_frame(2, has_pose=True),
        ]
        scores = compute_motion_scores(frames)
        assert scores[1] == pytest.approx(0.0)

    def test_large_movement_gives_high_score(self):
        frames = [
            _make_frame(0, x=0.0, y=0.0),
            _make_frame(1, x=1.0, y=1.0),  # jump from corner to corner
        ]
        scores = compute_motion_scores(frames)
        assert scores[1] > DEFAULT_THRESHOLD * 10


# ─── label_motion ─────────────────────────────────────────────────────────────

class TestLabelMotion:
    def test_still_frames_labelled_rest(self):
        frames = _still_frames(5)
        labels = label_motion(frames)
        assert all(lb == REST_LABEL for lb in labels)

    def test_moving_frames_labelled_active(self):
        frames = _moving_frames(10, step=0.1)
        labels = label_motion(frames, threshold=0.001)
        # Frames 1+ should all be active
        assert all(lb == ACTIVE_LABEL for lb in labels[1:])

    def test_returns_one_label_per_frame(self):
        frames = _still_frames(8)
        labels = label_motion(frames)
        assert len(labels) == 8

    def test_only_valid_labels(self):
        frames = _moving_frames(10)
        labels = label_motion(frames)
        valid = {ACTIVE_LABEL, REST_LABEL}
        assert all(lb in valid for lb in labels)

    def test_threshold_zero_all_active(self):
        """With threshold=0, any non-zero motion → active."""
        frames = _still_frames(5)  # zero motion
        labels = label_motion(frames, threshold=0.0)
        # Still frames have motion score exactly 0.0, so with threshold=0.0
        # only scores *strictly* > 0 are active → all should be rest
        assert all(lb == REST_LABEL for lb in labels)

    def test_high_threshold_all_rest(self):
        """With very high threshold, everything is rest."""
        frames = _moving_frames(10, step=0.1)
        labels = label_motion(frames, threshold=999.0)
        assert all(lb == REST_LABEL for lb in labels)


# ─── filter_active ────────────────────────────────────────────────────────────

class TestFilterActive:
    def test_returns_tuple(self):
        frames = _moving_frames(5)
        result = filter_active(frames)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_active_frames_subset_of_input(self):
        frames = _moving_frames(10)
        active_frames, active_indices = filter_active(frames, threshold=0.001)
        assert len(active_frames) <= len(frames)

    def test_indices_match_frames(self):
        frames = _moving_frames(10, step=0.1)
        active_frames, active_indices = filter_active(frames, threshold=0.001)
        for af, ai in zip(active_frames, active_indices):
            assert af["frame_index"] == ai

    def test_still_video_no_active_frames(self):
        frames = _still_frames(10)
        active_frames, active_indices = filter_active(frames)
        assert len(active_frames) == 0
        assert active_indices == []


# ─── summarise ────────────────────────────────────────────────────────────────

class TestSummarise:
    def test_returns_dict_with_required_keys(self):
        frames = _still_frames(5)
        result = summarise(frames)
        required = {"threshold", "total_frames", "active_frames", "rest_frames",
                    "active_pct", "per_frame"}
        assert required.issubset(result.keys())

    def test_total_matches_frame_count(self):
        frames = _still_frames(7)
        result = summarise(frames)
        assert result["total_frames"] == 7

    def test_active_plus_rest_equals_total(self):
        frames = _moving_frames(10)
        result = summarise(frames)
        assert result["active_frames"] + result["rest_frames"] == result["total_frames"]

    def test_per_frame_length_matches(self):
        frames = _still_frames(6)
        result = summarise(frames)
        assert len(result["per_frame"]) == 6


# ─── CLI ──────────────────────────────────────────────────────────────────────

class TestCLI:
    def test_cli_writes_json(self, tmp_path):
        # Write a small keypoints JSON
        frames = _still_frames(5)
        kp_file = tmp_path / "kp.json"
        kp_file.write_text(json.dumps({"frames": frames}), encoding="utf-8")

        out = tmp_path / "motion.json"
        main([str(kp_file), "--out", str(out)])

        assert out.exists()
        data = json.loads(out.read_text())
        assert "total_frames" in data
        assert data["total_frames"] == 5

    def test_cli_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            main(["/no/such/file.json"])
