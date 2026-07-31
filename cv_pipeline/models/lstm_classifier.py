"""
cv_pipeline/models/lstm_classifier.py
────────────────────────────────────────────────────────────────────────────────
PyTorch LSTM action classifier for CalorieVision.

Classifies sequences of MediaPipe Pose keypoints into one of 12 exercise classes.

Architecture
------------
Input  : (batch, seq_len, 99)  — 33 landmarks × (x, y, z)
LSTM   : 2 layers, hidden_dim=128, dropout=0.3
FC head: Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→num_classes)
Output : raw logits (batch, num_classes)

The 12 locked exercise classes (leMON Team 15)
----------------------------------------------
    squat, pushup, jumping_jack, lunge, plank,
    burpee, mountain_climber, high_knees, situp,
    jump_rope, bicycle_crunch, shoulder_press

Usage (library)
---------------
    from cv_pipeline.models.lstm_classifier import LSTMClassifier, EXERCISE_CLASSES
    model = LSTMClassifier()
    logits = model(x)  # x: (batch, seq_len, 99) torch.Tensor

Usage (inference helper)
------------------------
    from cv_pipeline.models.lstm_classifier import predict_sequence
    label, confidence = predict_sequence(frames, model)

Usage (load checkpoint)
-----------------------
    import torch
    model = LSTMClassifier.from_checkpoint("cv_pipeline/models/lstm_best.pt")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─── Exercise taxonomy ────────────────────────────────────────────────────────

EXERCISE_CLASSES: List[str] = [
    # ── Original 12 classes ──
    "squat",
    "pushup",
    "jumping_jack",
    "lunge",
    "plank",
    "burpee",
    "mountain_climber",
    "high_knees",
    "situp",
    "jump_rope",
    "bicycle_crunch",
    "shoulder_press",
    # ── Expanded 14 classes ──
    "deadlift",
    "pull_up",
    "bench_press",
    "tricep_dip",
    "leg_raise",
    "wall_sit",
    "box_jump",
    "russian_twist",
    "hip_thrust",
    "calf_raise",
    "lateral_raise",
    "bicep_curl",
    "kettlebell_swing",
    "superman_hold",
]

NUM_CLASSES   = len(EXERCISE_CLASSES)
INPUT_DIM     = 99    # 33 MediaPipe landmarks × (x, y, z)
HIDDEN_DIM    = 128
NUM_LAYERS    = 2
DROPOUT       = 0.3
DEFAULT_SEQ   = 30    # frames per window (≈1 s at 30 fps)
DEFAULT_STRIDE = 15   # 50 % overlap between windows

# ─── Model definition ─────────────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn

    class LSTMClassifier(nn.Module):
        """
        2-layer LSTM + FC head for per-window exercise classification.

        Parameters
        ----------
        input_dim : int
            Feature vector size per frame.  Default 99 (33 landmarks × xyz).
        hidden_dim : int
            LSTM hidden state size.  Default 128.
        num_layers : int
            Number of stacked LSTM layers.  Default 2.
        num_classes : int
            Number of output classes.  Default 12.
        dropout : float
            Dropout applied between LSTM layers and in the FC head.
        """

        def __init__(
            self,
            input_dim:   int = INPUT_DIM,
            hidden_dim:  int = HIDDEN_DIM,
            num_layers:  int = NUM_LAYERS,
            num_classes: int = NUM_CLASSES,
            dropout:     float = DROPOUT,
        ) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes),
            )
            self.num_classes = num_classes
            self.input_dim   = input_dim
            self.hidden_dim  = hidden_dim

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Parameters
            ----------
            x : Tensor of shape (batch, seq_len, input_dim)

            Returns
            -------
            Tensor of shape (batch, num_classes) — raw logits
            """
            _, (h_n, _) = self.lstm(x)   # h_n: (num_layers, batch, hidden)
            last_hidden  = h_n[-1]        # (batch, hidden)
            return self.head(last_hidden)

        def predict(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            """
            Convenience method: returns (class_indices, probabilities).

            Parameters
            ----------
            x : Tensor (batch, seq_len, input_dim)

            Returns
            -------
            class_idx : LongTensor (batch,)
            probs     : FloatTensor (batch, num_classes)
            """
            self.eval()
            with torch.no_grad():
                logits = self.forward(x)
                probs  = torch.softmax(logits, dim=-1)
                idx    = torch.argmax(probs, dim=-1)
            return idx, probs

        @classmethod
        def from_checkpoint(cls, path: str | Path, **kwargs) -> "LSTMClassifier":
            """Load a saved checkpoint."""
            ckpt = torch.load(str(path), map_location="cpu")
            model = cls(**{**kwargs, **ckpt.get("model_kwargs", {})})
            model.load_state_dict(ckpt["model_state_dict"])
            logger.info("Loaded LSTM checkpoint from %s", path)
            return model

        def save_checkpoint(self, path: str | Path, **extra) -> None:
            """Save model state + construction kwargs."""
            ckpt = {
                "model_state_dict": self.state_dict(),
                "model_kwargs": {
                    "input_dim":   self.input_dim,
                    "hidden_dim":  self.hidden_dim,
                    "num_classes": self.num_classes,
                },
                **extra,
            }
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, str(path))
            logger.info("Checkpoint saved → %s", path)

    _TORCH_AVAILABLE = True

except ImportError:
    # torch not installed — define a stub so imports don't crash
    class LSTMClassifier:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            raise ImportError("torch is required for LSTMClassifier. pip install torch")

    _TORCH_AVAILABLE = False


# ─── Sliding-window helpers ───────────────────────────────────────────────────

def frames_to_windows(
    frames: List[dict],
    *,
    seq_len: int = DEFAULT_SEQ,
    stride:  int = DEFAULT_STRIDE,
) -> Tuple[np.ndarray, List[int]]:
    """
    Slice a frame list into overlapping fixed-length windows for LSTM inference.

    Parameters
    ----------
    frames : list[dict]
        Output of extract_keypoints() — one dict per frame.
    seq_len : int
        Window length in frames.
    stride : int
        Step between consecutive windows.

    Returns
    -------
    windows : np.ndarray, shape (n_windows, seq_len, 99)
        Float32 feature array.  Frames with no pose use a zero vector.
    start_indices : list[int]
        Frame index of the first frame of each window.
    """
    # Build frame feature matrix, zero-filling missing poses
    n_frames = len(frames)
    if n_frames == 0:
        return np.empty((0, seq_len, INPUT_DIM), dtype=np.float32), []

    X = np.zeros((n_frames, INPUT_DIM), dtype=np.float32)
    for i, frame in enumerate(frames):
        lms = frame.get("landmarks", [])
        if lms:
            try:
                vec = np.array(
                    [coord for lm in lms for coord in (lm["x"], lm["y"], lm["z"])],
                    dtype=np.float32,
                )
                if vec.shape[0] == INPUT_DIM:
                    X[i] = vec
            except (KeyError, TypeError):
                pass  # leave as zeros

    # Slice into windows
    windows: List[np.ndarray] = []
    start_indices: List[int]  = []

    for start in range(0, max(1, n_frames - seq_len + 1), stride):
        end = start + seq_len
        if end > n_frames:
            # Pad the last window with zeros
            window = np.zeros((seq_len, INPUT_DIM), dtype=np.float32)
            window[: n_frames - start] = X[start:]
        else:
            window = X[start:end]
        windows.append(window)
        start_indices.append(start)

    if not windows:
        return np.empty((0, seq_len, INPUT_DIM), dtype=np.float32), []

    return np.stack(windows, axis=0), start_indices
def predict_sequence(
    frames: List[dict],
    model: "LSTMClassifier",
    *,
    seq_len: int = DEFAULT_SEQ,
    stride:  int = DEFAULT_STRIDE,
    fps:     float = 30.0,
) -> List[dict]:
    """
    Run LSTM inference over a full frame list and return per-window predictions.

    Parameters
    ----------
    frames : list[dict]
    model  : LSTMClassifier   (must be in eval mode)
    seq_len : int             window size in frames
    stride  : int             step between windows
    fps     : float           video frame rate (for timestamp computation)

    Returns
    -------
    list[dict]  — one dict per window::

        {
            "window_index":  int,
            "start_frame":   int,
            "end_frame":     int,
            "start_time":    float,   # seconds
            "end_time":      float,
            "label":         str,     # predicted exercise class
            "confidence":    float,   # softmax probability of top class
            "probs":         dict,    # {class_name: probability}
        }
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required for predict_sequence.")

    import torch

    windows, start_indices = frames_to_windows(frames, seq_len=seq_len, stride=stride)
    if len(windows) == 0:
        return []

    x = torch.from_numpy(windows)   # (n_windows, seq_len, 99)
    class_idx, probs = model.predict(x)  # both on CPU

    results = []
    for w_idx, (start_f, c_idx, prob_vec) in enumerate(
        zip(start_indices, class_idx.tolist(), probs.tolist())
    ):
        end_f  = min(start_f + seq_len, len(frames))
        label  = EXERCISE_CLASSES[c_idx]
        conf   = float(prob_vec[c_idx])
        prob_d = {cls: round(float(p), 4) for cls, p in zip(EXERCISE_CLASSES, prob_vec)}

        results.append(
            {
                "window_index": w_idx,
                "start_frame":  start_f,
                "end_frame":    end_f,
                "start_time":   round(start_f / fps, 3),
                "end_time":     round(end_f   / fps, 3),
                "label":        label,
                "confidence":   round(conf, 4),
                "probs":        prob_d,
            }
        )

    return results
