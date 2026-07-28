"""
cv-pipeline/models/baseline.py
────────────────────────────────────────────────────────────────────────────────
Cheap, fast baseline classifiers for the CalorieVision exercise recognition
pipeline.  These models exist purely to establish a floor that the later
LSTM / 1D-CNN must beat — they are NOT intended for production use.

Two classifiers are provided:

  MajorityClassPredictor
    Always predicts the most-frequent class seen during training.
    Upper bound for any 0-effort baseline.

  KNNBaselineClassifier
    Flattens each frame's 33 MediaPipe landmarks (x, y, z) into a 99-d vector
    and runs k-Nearest Neighbours (cosine distance by default).

Input contract
--------------
Both classifiers accept data in the format produced by
cv_pipeline.keypoints.extract_keypoints.extract_keypoints():

    frames: list[dict]  — one dict per frame, each with a "landmarks" key
                          holding a list of 33 landmark dicts {x, y, z, ...}.
                          Frames where pose_detected=False have landmarks=[].

Frames with no detected pose (empty landmark list) are silently skipped
during training, and emit the UNKNOWN_LABEL sentinel at inference time.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

UNKNOWN_LABEL = "UNKNOWN"   # returned when no pose is detected in a frame
N_LANDMARKS   = 33          # MediaPipe Pose always outputs 33 landmarks
VECTOR_DIM    = N_LANDMARKS * 3  # x, y, z per landmark → 99 features


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _frame_to_vector(frame: dict) -> Optional[np.ndarray]:
    """
    Convert one frame dict (from extract_keypoints output) to a flat numpy
    array of shape (99,) = [x0, y0, z0, x1, y1, z1, …, x32, y32, z32].

    Returns None if the frame has no detected pose.
    """
    landmarks = frame.get("landmarks", [])
    if not landmarks:
        return None
    vec = np.array(
        [coord for lm in landmarks for coord in (lm["x"], lm["y"], lm["z"])],
        dtype=np.float32,
    )
    if vec.shape[0] != VECTOR_DIM:
        logger.warning(
            "Expected %d landmark coordinates, got %d; skipping frame.",
            VECTOR_DIM, vec.shape[0],
        )
        return None
    return vec


def frames_to_matrix(frames: List[dict]) -> tuple[np.ndarray, list[int]]:
    """
    Convert a list of frame dicts to a feature matrix, keeping track of which
    original frame indices contributed rows.

    Returns
    -------
    X : np.ndarray, shape (n_valid_frames, 99)
    valid_indices : list[int]  — original frame indices included in X
    """
    rows, valid_indices = [], []
    for idx, frame in enumerate(frames):
        vec = _frame_to_vector(frame)
        if vec is not None:
            rows.append(vec)
            valid_indices.append(idx)
    if not rows:
        return np.empty((0, VECTOR_DIM), dtype=np.float32), []
    return np.vstack(rows), valid_indices


# ─── Majority-class predictor ─────────────────────────────────────────────────

class MajorityClassPredictor:
    """
    Trivial baseline: always predicts the most-frequent training label.

    Scikit-learn-compatible interface (fit / predict / score).
    """

    def __init__(self) -> None:
        self._majority: Optional[str] = None

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, frames: List[dict], labels: List[str]) -> "MajorityClassPredictor":
        """
        Record the majority class from the training data.

        Parameters
        ----------
        frames : list[dict]   — extract_keypoints output (used only to filter
                                frames with a detected pose, matching predict's
                                behaviour).
        labels : list[str]    — one label per frame in `frames`.
        """
        if len(frames) != len(labels):
            raise ValueError(
                f"frames ({len(frames)}) and labels ({len(labels)}) must have "
                "the same length."
            )
        _, valid_idx = frames_to_matrix(frames)
        valid_labels = [labels[i] for i in valid_idx]
        if not valid_labels:
            logger.warning("MajorityClassPredictor.fit(): no valid poses found.")
            self._majority = UNKNOWN_LABEL
            return self
        counter = Counter(valid_labels)
        self._majority = counter.most_common(1)[0][0]
        logger.info("MajorityClassPredictor trained; majority class = %r", self._majority)
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, frames: List[dict]) -> List[str]:
        """Return the majority-class label for every frame that has a pose."""
        if self._majority is None:
            raise RuntimeError("Call fit() before predict().")
        return [
            self._majority if frame.get("landmarks") else UNKNOWN_LABEL
            for frame in frames
        ]

    def score(self, frames: List[dict], labels: List[str]) -> float:
        """Accuracy on frames that have a detected pose."""
        preds = self.predict(frames)
        correct = total = 0
        for pred, true in zip(preds, labels):
            if pred == UNKNOWN_LABEL:
                continue
            total += 1
            correct += int(pred == true)
        return correct / total if total else 0.0


# ─── k-NN baseline ────────────────────────────────────────────────────────────

class KNNBaselineClassifier:
    """
    k-Nearest Neighbours classifier on raw 99-d landmark vectors.

    Uses sklearn's KNeighborsClassifier under the hood; cosine distance is
    the default since landmark vectors are normalised to the frame dimensions.

    Parameters
    ----------
    k : int     — number of neighbours (default 5)
    metric : str — distance metric passed to KNeighborsClassifier
    """

    def __init__(self, k: int = 5, metric: str = "cosine") -> None:
        try:
            from sklearn.neighbors import KNeighborsClassifier  # lazy import
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for KNNBaselineClassifier. "
                "Install it with: pip install scikit-learn"
            ) from exc

        self.k = k
        self.metric = metric
        self._clf = KNeighborsClassifier(n_neighbors=k, metric=metric)
        self._fitted = False

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        frames: List[dict],
        labels: List[str],
    ) -> "KNNBaselineClassifier":
        """
        Fit k-NN on frames that contain a detected pose.

        Parameters
        ----------
        frames : list[dict]  — extract_keypoints output
        labels : list[str]   — per-frame class labels (same length as frames)
        """
        if len(frames) != len(labels):
            raise ValueError(
                f"frames ({len(frames)}) and labels ({len(labels)}) must have "
                "the same length."
            )
        X, valid_idx = frames_to_matrix(frames)
        y = [labels[i] for i in valid_idx]
        if len(X) == 0:
            raise ValueError("No frames with detected poses found in training data.")

        n_classes = len(set(y))
        effective_k = min(self.k, len(X))
        if effective_k < self.k:
            logger.warning(
                "Training set (%d samples) < k=%d; reducing k to %d.",
                len(X), self.k, effective_k,
            )
            self._clf.set_params(n_neighbors=effective_k)

        self._clf.fit(X, y)
        self._fitted = True
        logger.info(
            "KNNBaselineClassifier fitted: %d samples, %d classes, k=%d, metric=%r",
            len(X), n_classes, effective_k, self.metric,
        )
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, frames: List[dict]) -> List[str]:
        """
        Return per-frame predictions.

        Frames without a detected pose get the UNKNOWN_LABEL sentinel.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

        predictions: List[str] = []
        for frame in frames:
            vec = _frame_to_vector(frame)
            if vec is None:
                predictions.append(UNKNOWN_LABEL)
            else:
                label = self._clf.predict(vec.reshape(1, -1))[0]
                predictions.append(str(label))
        return predictions

    def predict_proba(self, frames: List[dict]) -> np.ndarray:
        """Return class probabilities; rows for pose-less frames are zeros."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        n_classes = len(self._clf.classes_)
        result = np.zeros((len(frames), n_classes), dtype=np.float32)
        for i, frame in enumerate(frames):
            vec = _frame_to_vector(frame)
            if vec is not None:
                result[i] = self._clf.predict_proba(vec.reshape(1, -1))[0]
        return result

    def score(self, frames: List[dict], labels: List[str]) -> float:
        """Accuracy on frames that have a detected pose."""
        preds = self.predict(frames)
        correct = total = 0
        for pred, true in zip(preds, labels):
            if pred == UNKNOWN_LABEL:
                continue
            total += 1
            correct += int(pred == true)
        return correct / total if total else 0.0

    @property
    def classes_(self) -> Sequence[str]:
        """Ordered class names (available after fit)."""
        return list(self._clf.classes_) if self._fitted else []
