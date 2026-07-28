"""
cv_pipeline/eval/test_lstm.py
────────────────────────────────────────────────────────────────────────────────
Unit tests for LSTMClassifier and related helpers.

All tests use synthetic/random tensors — no real video, no real training.
Tests that require torch are skipped gracefully if torch is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Conditional torch import
try:
    import torch
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

pytestmark_torch = pytest.mark.skipif(not _TORCH_OK, reason="torch not installed")

from cv_pipeline.models.lstm_classifier import (
    EXERCISE_CLASSES,
    NUM_CLASSES,
    INPUT_DIM,
    DEFAULT_SEQ,
    DEFAULT_STRIDE,
    LSTMClassifier,
    frames_to_windows,
    predict_sequence,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_landmark(j: int, val: float = 0.5) -> dict:
    return {"index": j, "name": f"lm_{j}", "x": val, "y": val, "z": 0.0, "visibility": 1.0}


def _make_frame(i: int, has_pose: bool = True) -> dict:
    lms = [_make_landmark(j) for j in range(33)] if has_pose else []
    return {"frame_index": i, "timestamp": i / 30.0,
            "pose_detected": has_pose, "landmarks": lms}


def _make_frames(n: int) -> list[dict]:
    return [_make_frame(i) for i in range(n)]


# ─── EXERCISE_CLASSES ─────────────────────────────────────────────────────────

class TestExerciseClasses:
    def test_exactly_12_classes(self):
        assert len(EXERCISE_CLASSES) == 12

    def test_no_duplicates(self):
        assert len(EXERCISE_CLASSES) == len(set(EXERCISE_CLASSES))

    def test_known_classes_present(self):
        for cls in ("squat", "pushup", "plank", "burpee", "situp"):
            assert cls in EXERCISE_CLASSES

    def test_num_classes_matches(self):
        assert NUM_CLASSES == len(EXERCISE_CLASSES)


# ─── frames_to_windows ────────────────────────────────────────────────────────

class TestFramesToWindows:
    def test_output_shape(self):
        frames = _make_frames(60)
        windows, starts = frames_to_windows(frames, seq_len=30, stride=15)
        assert windows.ndim == 3
        assert windows.shape[1] == 30
        assert windows.shape[2] == INPUT_DIM

    def test_start_indices_ascending(self):
        frames = _make_frames(60)
        _, starts = frames_to_windows(frames, seq_len=30, stride=15)
        assert starts == sorted(starts)

    def test_no_pose_frames_get_zeros(self):
        frames = [_make_frame(i, has_pose=False) for i in range(30)]
        windows, _ = frames_to_windows(frames, seq_len=30, stride=30)
        assert (windows == 0.0).all()

    def test_short_video_still_produces_window(self):
        frames = _make_frames(10)  # shorter than default seq_len=30
        windows, starts = frames_to_windows(frames, seq_len=30, stride=30)
        assert len(windows) >= 1

    def test_dtype_is_float32(self):
        frames = _make_frames(30)
        windows, _ = frames_to_windows(frames)
        assert windows.dtype == np.float32

    def test_empty_frames_returns_empty(self):
        windows, starts = frames_to_windows([], seq_len=30, stride=15)
        assert len(windows) == 0
        assert starts == []


# ─── LSTMClassifier ──────────────────────────────────────────────────────────

@pytest.mark.skipif(not _TORCH_OK, reason="torch not installed")
class TestLSTMClassifier:
    def test_forward_output_shape(self):
        model = LSTMClassifier()
        x = torch.zeros(4, DEFAULT_SEQ, INPUT_DIM)  # batch=4
        logits = model(x)
        assert logits.shape == (4, NUM_CLASSES)

    def test_predict_returns_index_and_probs(self):
        model = LSTMClassifier()
        x = torch.zeros(3, DEFAULT_SEQ, INPUT_DIM)
        idx, probs = model.predict(x)
        assert idx.shape == (3,)
        assert probs.shape == (3, NUM_CLASSES)

    def test_predict_indices_in_valid_range(self):
        model = LSTMClassifier()
        x = torch.randn(10, DEFAULT_SEQ, INPUT_DIM)
        idx, _ = model.predict(x)
        assert (idx >= 0).all() and (idx < NUM_CLASSES).all()

    def test_probs_sum_to_one(self):
        model = LSTMClassifier()
        x = torch.randn(5, DEFAULT_SEQ, INPUT_DIM)
        _, probs = model.predict(x)
        row_sums = probs.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(5), atol=1e-5)

    def test_eval_mode_no_grad(self):
        """predict() must work without grad context."""
        model = LSTMClassifier()
        x = torch.randn(2, DEFAULT_SEQ, INPUT_DIM)
        # Should not raise even outside torch.no_grad()
        idx, probs = model.predict(x)
        assert idx is not None

    def test_save_and_load_checkpoint(self, tmp_path):
        model = LSTMClassifier()
        model.eval()  # deterministic (no dropout)
        ckpt_path = tmp_path / "test.pt"
        model.save_checkpoint(ckpt_path)

        loaded = LSTMClassifier.from_checkpoint(ckpt_path)
        loaded.eval()  # must also be in eval mode
        # Both should produce same output
        x = torch.zeros(1, DEFAULT_SEQ, INPUT_DIM)
        with torch.no_grad():
            out1 = model(x)
            out2 = loaded(x)
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_single_batch_works(self):
        model = LSTMClassifier()
        x = torch.zeros(1, DEFAULT_SEQ, INPUT_DIM)
        logits = model(x)
        assert logits.shape == (1, NUM_CLASSES)


# ─── predict_sequence ────────────────────────────────────────────────────────

@pytest.mark.skipif(not _TORCH_OK, reason="torch not installed")
class TestPredictSequence:
    def test_returns_list(self):
        model = LSTMClassifier()
        frames = _make_frames(60)
        result = predict_sequence(frames, model)
        assert isinstance(result, list)

    def test_each_item_has_required_keys(self):
        model = LSTMClassifier()
        frames = _make_frames(60)
        result = predict_sequence(frames, model)
        required = {"window_index", "start_frame", "end_frame",
                    "start_time", "end_time", "label", "confidence", "probs"}
        for item in result:
            assert required.issubset(item.keys())

    def test_labels_are_valid_classes(self):
        model = LSTMClassifier()
        frames = _make_frames(60)
        result = predict_sequence(frames, model)
        for item in result:
            assert item["label"] in EXERCISE_CLASSES

    def test_confidence_in_range(self):
        model = LSTMClassifier()
        frames = _make_frames(60)
        result = predict_sequence(frames, model)
        for item in result:
            assert 0.0 <= item["confidence"] <= 1.0

    def test_empty_frames_returns_empty(self):
        model = LSTMClassifier()
        result = predict_sequence([], model)
        assert result == []
