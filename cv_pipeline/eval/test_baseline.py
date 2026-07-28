"""
cv-pipeline/eval/test_baseline.py
────────────────────────────────────────────────────────────────────────────────
Unit tests for cv_pipeline.models.baseline.

All tests use procedurally-generated synthetic frames; no real video needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cv_pipeline.models.baseline import (
    UNKNOWN_LABEL,
    VECTOR_DIM,
    KNNBaselineClassifier,
    MajorityClassPredictor,
    _frame_to_vector,
    frames_to_matrix,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_landmark(idx: int, x: float = 0.5, y: float = 0.5, z: float = 0.0) -> dict:
    return {"index": idx, "name": f"lm_{idx}", "x": x, "y": y, "z": z, "visibility": 1.0}


def _make_frame(frame_index: int = 0, has_pose: bool = True) -> dict:
    if has_pose:
        landmarks = [_make_landmark(i) for i in range(33)]
    else:
        landmarks = []
    return {
        "frame_index": frame_index,
        "timestamp": frame_index / 30.0,
        "pose_detected": has_pose,
        "landmarks": landmarks,
    }


def _make_frames(n: int, *, no_pose_every: int | None = None) -> list[dict]:
    frames = []
    for i in range(n):
        has = (no_pose_every is None) or (i % no_pose_every != 0)
        frames.append(_make_frame(i, has_pose=has))
    return frames


def _make_labels(n: int, classes: list[str] = ("A", "B", "C")) -> list[str]:
    return [classes[i % len(classes)] for i in range(n)]


# ─── _frame_to_vector ─────────────────────────────────────────────────────────

class TestFrameToVector:
    def test_returns_none_for_no_pose(self):
        frame = _make_frame(has_pose=False)
        assert _frame_to_vector(frame) is None

    def test_returns_array_for_valid_frame(self):
        frame = _make_frame(has_pose=True)
        vec = _frame_to_vector(frame)
        assert vec is not None
        assert vec.shape == (VECTOR_DIM,)

    def test_vector_values_match_landmarks(self):
        frame = _make_frame(has_pose=True)
        vec = _frame_to_vector(frame)
        # First landmark: x=0.5, y=0.5, z=0.0 → vec[0:3] = [0.5, 0.5, 0.0]
        assert vec is not None
        assert vec[0] == pytest.approx(0.5)
        assert vec[1] == pytest.approx(0.5)
        assert vec[2] == pytest.approx(0.0)


# ─── frames_to_matrix ────────────────────────────────────────────────────────

class TestFramesToMatrix:
    def test_all_valid(self):
        frames = _make_frames(10)
        X, idx = frames_to_matrix(frames)
        assert X.shape == (10, VECTOR_DIM)
        assert idx == list(range(10))

    def test_some_missing_pose(self):
        frames = _make_frames(5, no_pose_every=2)  # frames 0, 2, 4 have no pose
        X, idx = frames_to_matrix(frames)
        assert X.shape[0] == 2  # only frames 1, 3
        assert idx == [1, 3]

    def test_all_missing_pose(self):
        frames = [_make_frame(i, has_pose=False) for i in range(5)]
        X, idx = frames_to_matrix(frames)
        assert X.shape[0] == 0
        assert idx == []


# ─── MajorityClassPredictor ───────────────────────────────────────────────────

class TestMajorityClassPredictor:
    def test_fit_predict_smoke(self):
        frames = _make_frames(30)
        labels = ["A"] * 20 + ["B"] * 10
        model = MajorityClassPredictor()
        model.fit(frames, labels)
        preds = model.predict(frames)
        assert len(preds) == 30
        # All predictions must be A (majority)
        assert all(p == "A" for p in preds)

    def test_predict_unknown_for_no_pose(self):
        frames = _make_frames(3, no_pose_every=2)  # 0,2 → no pose; 1 → pose
        labels = ["A", "A", "A"]
        model = MajorityClassPredictor()
        model.fit(frames, labels)
        preds = model.predict(frames)
        assert preds[0] == UNKNOWN_LABEL  # no pose
        assert preds[1] == "A"
        assert preds[2] == UNKNOWN_LABEL  # no pose

    def test_predict_before_fit_raises(self):
        model = MajorityClassPredictor()
        with pytest.raises(RuntimeError):
            model.predict(_make_frames(5))

    def test_mismatched_lengths_raises(self):
        model = MajorityClassPredictor()
        with pytest.raises(ValueError):
            model.fit(_make_frames(10), ["A"] * 5)

    def test_score_returns_float_in_range(self):
        frames = _make_frames(20)
        labels = ["A"] * 10 + ["B"] * 10
        model = MajorityClassPredictor()
        model.fit(frames, labels)
        score = model.score(frames, labels)
        assert 0.0 <= score <= 1.0


# ─── KNNBaselineClassifier ────────────────────────────────────────────────────

class TestKNNBaselineClassifier:
    def test_fit_predict_smoke(self):
        frames = _make_frames(30)
        labels = _make_labels(30, classes=["squat", "pushup"])
        knn = KNNBaselineClassifier(k=3, metric="euclidean")
        knn.fit(frames, labels)
        preds = knn.predict(frames)
        assert len(preds) == 30
        # All preds should be either known class or UNKNOWN_LABEL
        valid = {"squat", "pushup", UNKNOWN_LABEL}
        assert all(p in valid for p in preds)

    def test_predict_unknown_for_no_pose(self):
        frames = _make_frames(5, no_pose_every=5)  # frame 0 has no pose
        labels = _make_labels(5, classes=["A", "B"])
        knn = KNNBaselineClassifier(k=2)
        knn.fit(frames, labels)
        preds = knn.predict(frames)
        assert preds[0] == UNKNOWN_LABEL

    def test_predict_before_fit_raises(self):
        knn = KNNBaselineClassifier()
        with pytest.raises(RuntimeError):
            knn.predict(_make_frames(5))

    def test_mismatched_lengths_raises(self):
        knn = KNNBaselineClassifier()
        with pytest.raises(ValueError):
            knn.fit(_make_frames(10), ["A"] * 5)

    def test_classes_available_after_fit(self):
        frames = _make_frames(20)
        labels = _make_labels(20, classes=["X", "Y", "Z"])
        knn = KNNBaselineClassifier(k=2)
        knn.fit(frames, labels)
        assert set(knn.classes_) == {"X", "Y", "Z"}

    def test_classes_empty_before_fit(self):
        knn = KNNBaselineClassifier()
        assert knn.classes_ == []

    def test_score_returns_float_in_range(self):
        frames = _make_frames(20)
        labels = _make_labels(20, classes=["A", "B"])
        knn = KNNBaselineClassifier(k=2)
        knn.fit(frames, labels)
        score = knn.score(frames, labels)
        assert 0.0 <= score <= 1.0

    def test_predict_proba_shape(self):
        frames = _make_frames(10)
        labels = _make_labels(10, classes=["squat", "pushup"])
        knn = KNNBaselineClassifier(k=3)
        knn.fit(frames, labels)
        proba = knn.predict_proba(frames)
        assert proba.shape == (10, 2)

    def test_k_reduced_when_few_samples(self):
        """k > n_samples should not crash; k is silently clamped."""
        frames = _make_frames(3)
        labels = ["A", "B", "A"]
        knn = KNNBaselineClassifier(k=10)
        knn.fit(frames, labels)  # should not raise
        preds = knn.predict(frames)
        assert len(preds) == 3
