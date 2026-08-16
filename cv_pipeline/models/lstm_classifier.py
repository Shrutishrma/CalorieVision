"""
cv_pipeline/models/lstm_classifier.py
────────────────────────────────────────────────────────────────────────────────
PyTorch LSTM action classifier for CalorieVision.

Classifies sequences of MediaPipe Pose keypoints into exercise classes using
BIOMECHANICAL FEATURES (joint angles, body ratios) instead of raw (x,y,z)
coordinates. This makes the model:

  * Camera-angle agnostic  (joint angles don't change with viewpoint)
  * Scale invariant        (angles are dimensionless)
  * Robust to AI/synthetic video artefacts

Feature vector per frame (INPUT_DIM = 14)
-----------------------------------------
  0  knee_angle_left      hip -> knee -> ankle
  1  knee_angle_right
  2  hip_angle_left       shoulder -> hip -> knee
  3  hip_angle_right
  4  elbow_angle_left     shoulder -> elbow -> wrist
  5  elbow_angle_right
  6  shoulder_elev_left   wrist_y relative to shoulder_y (>1 = arms raised)
  7  shoulder_elev_right
  8  torso_lean           angle of shoulder-to-hip line from vertical (0=upright)
  9  body_aspect_ratio    bbox_height / bbox_width  (tall=upright, low=horizontal)
  10 hip_height_ratio     hip_y / ankle_y           (lower value = deeper squat)
  11 wrist_height_ratio   mean_wrist_y / mean_shoulder_y
  12 knee_spread          |left_knee_x - right_knee_x|  (wide=squat/lunge)
  13 arm_spread           |left_wrist_x - right_wrist_x| (wide=jumping_jack)

All angles normalised to [0,1] by dividing by 180. All ratios clipped to [0,2].

Architecture
------------
Input  : (batch, seq_len=15, 14)
LSTM   : 2 layers, hidden_dim=128, dropout=0.3
FC head: Linear(128->64) -> ReLU -> Dropout(0.3) -> Linear(64->num_classes)
Output : raw logits (batch, num_classes)

Exercise classes (9 realistic workout exercises)
------------------------------------------------
    squat, pushup, plank, jumping_jack, lunge,
    situp, burpee, mountain_climber, rest
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─── PyTorch Imports & Base Class ─────────────────────────────────────────────

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
    _ModuleBase = nn.Module
else:
    try:
        import torch
        import torch.nn as nn
        _TORCH_AVAILABLE = True
        _ModuleBase = nn.Module
    except ImportError:
        torch = None
        nn = None
        _TORCH_AVAILABLE = False
        _ModuleBase = object

# ─── Exercise taxonomy ────────────────────────────────────────────────────────

EXERCISE_CLASSES: List[str] = [
    "squat",
    "pushup",
    "plank",
    "jumping_jack",
    "lunge",
    "situp",
    "burpee",
    "mountain_climber",
    "rest",
]

SHRUTI_CLASSES: List[str] = [
    "BodyWeightSquats",
    "JumpRope",
    "JumpingJack",
    "Lunges",
    "PullUps",
    "leg raises",
    "plank",
    "push-up",
    "russian twist",
    "shoulder press",
    "tricep dips",
]

SHRUTI_TO_CANONICAL: dict[str, str] = {
    "BodyWeightSquats": "squat",
    "push-up": "pushup",
    "PullUps": "pull_up",
    "tricep dips": "tricep_dip",
    "russian twist": "russian_twist",
    "shoulder press": "shoulder_press",
    "leg raises": "leg_raise",
    "plank": "plank",
    "Lunges": "lunge",
    "JumpingJack": "jumping_jack",
    "JumpRope": "jump_rope",
}

NUM_CLASSES    = len(EXERCISE_CLASSES)
INPUT_DIM      = 14    # biomechanical features (see module docstring)
HIDDEN_DIM     = 128
NUM_LAYERS     = 2
DROPOUT        = 0.3
DEFAULT_SEQ    = 15    # frames per window (~7.5 s at 2 fps)
DEFAULT_STRIDE = 7     # ~50% overlap


# ─── 132-dim Normalized Keypoint Feature Extraction ──────────────────────────

def extract_normalized_132_features(frame: dict) -> np.ndarray:
    """
    Extract hip-centered, torso-scale normalized (132,) vector for Shruti's KeypointLSTM.
    Format: [x0, y0, z0, v0, x1, y1, z1, v1, ..., x32, y32, z32, v32].
    """
    landmarks = frame.get("landmarks", [])
    if not landmarks or len(landmarks) < 33:
        return np.zeros(132, dtype=np.float32)

    coords = np.zeros((33, 4), dtype=np.float32)
    for i, lm in enumerate(landmarks[:33]):
        coords[i, 0] = float(lm.get("x", 0.0))
        coords[i, 1] = float(lm.get("y", 0.0))
        coords[i, 2] = float(lm.get("z", 0.0))
        coords[i, 3] = float(lm.get("visibility", 1.0))

    # 1. Hip midpoint centering (landmark 23 left_hip, 24 right_hip)
    hip_center = (coords[23, :3] + coords[24, :3]) / 2.0
    coords[:, :3] -= hip_center

    # 2. Torso scale normalization (shoulder center to hip center)
    shoulder_center = (coords[11, :3] + coords[12, :3]) / 2.0
    torso_height = float(np.linalg.norm(shoulder_center))
    if torso_height > 1e-4:
        coords[:, :3] /= torso_height

    return coords.flatten()


# ─── Biomechanical feature extraction ────────────────────────────────────────

def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return interior angle at b (degrees) for triplet a-b-c."""
    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return 90.0
    cos_a = float(np.dot(ba, bc) / denom)
    cos_a = max(-1.0, min(1.0, cos_a))
    return math.degrees(math.acos(cos_a))


def _lm(landmarks: list, idx: int) -> Optional[np.ndarray]:
    """Return (x, y) for landmark at index idx, or None."""
    if idx < len(landmarks):
        lm = landmarks[idx]
        if lm and "x" in lm and "y" in lm:
            return np.array([float(lm["x"]), float(lm["y"])], dtype=np.float32)
    return None


def extract_biometric_features(frame: dict) -> np.ndarray:
    """
    Extract the 14-dimensional biomechanical feature vector from one frame dict.
    Returns a zero vector if no pose was detected.
    """
    feat = np.zeros(INPUT_DIM, dtype=np.float32)
    lms = frame.get("landmarks", [])
    if not lms:
        return feat

    l_shoulder = _lm(lms, 11)
    r_shoulder = _lm(lms, 12)
    l_elbow    = _lm(lms, 13)
    r_elbow    = _lm(lms, 14)
    l_wrist    = _lm(lms, 15)
    r_wrist    = _lm(lms, 16)
    l_hip      = _lm(lms, 23)
    r_hip      = _lm(lms, 24)
    l_knee     = _lm(lms, 25)
    r_knee     = _lm(lms, 26)
    l_ankle    = _lm(lms, 27)
    r_ankle    = _lm(lms, 28)

    def _sa(a, b, c, default=90.0):
        if a is None or b is None or c is None:
            return default
        return _angle(a, b, c)

    # 0-1: knee angles
    feat[0] = _sa(l_hip, l_knee, l_ankle) / 180.0
    feat[1] = _sa(r_hip, r_knee, r_ankle) / 180.0
    # 2-3: hip angles
    feat[2] = _sa(l_shoulder, l_hip, l_knee) / 180.0
    feat[3] = _sa(r_shoulder, r_hip, r_knee) / 180.0
    # 4-5: elbow angles
    feat[4] = _sa(l_shoulder, l_elbow, l_wrist) / 180.0
    feat[5] = _sa(r_shoulder, r_elbow, r_wrist) / 180.0
    # 6-7: shoulder elevation
    if l_shoulder is not None and l_wrist is not None:
        feat[6] = float(np.clip((l_wrist[1] - l_shoulder[1]) + 0.5, 0, 2))
    else:
        feat[6] = 0.5
    if r_shoulder is not None and r_wrist is not None:
        feat[7] = float(np.clip((r_wrist[1] - r_shoulder[1]) + 0.5, 0, 2))
    else:
        feat[7] = 0.5
    # 8: torso lean
    if l_shoulder is not None and l_hip is not None:
        sh_hip = l_hip - l_shoulder
        denom = np.linalg.norm(sh_hip)
        if denom > 1e-6:
            cos_v = float(np.dot(sh_hip / denom, np.array([0.0, 1.0])))
            cos_v = max(-1.0, min(1.0, cos_v))
            feat[8] = math.degrees(math.acos(cos_v)) / 180.0
    # 9: body aspect ratio
    xs = [lm["x"] for lm in lms if lm and "x" in lm]
    ys = [lm["y"] for lm in lms if lm and "y" in lm]
    if xs and ys:
        bbox_w = max(xs) - min(xs)
        bbox_h = max(ys) - min(ys)
        feat[9] = float(np.clip(bbox_h / bbox_w, 0, 2)) if bbox_w > 1e-4 else 1.0
    else:
        feat[9] = 1.0
    # 10: hip height ratio
    if l_hip is not None and l_ankle is not None and l_ankle[1] > 1e-4:
        feat[10] = float(np.clip(l_hip[1] / l_ankle[1], 0, 2))
    else:
        feat[10] = 0.8
    # 11: wrist height ratio
    if l_wrist is not None and r_wrist is not None and l_shoulder is not None:
        mwy = (l_wrist[1] + r_wrist[1]) / 2
        msy = (l_shoulder[1] + (r_shoulder[1] if r_shoulder is not None else l_shoulder[1])) / 2
        feat[11] = float(np.clip(mwy / msy, 0, 2)) if msy > 1e-4 else 1.0
    else:
        feat[11] = 1.0
    # 12: knee spread
    if l_knee is not None and r_knee is not None:
        feat[12] = float(np.clip(abs(l_knee[0] - r_knee[0]), 0, 1))
    else:
        feat[12] = 0.15
    # 13: arm spread
    if l_wrist is not None and r_wrist is not None:
        feat[13] = float(np.clip(abs(l_wrist[0] - r_wrist[0]), 0, 1))
    else:
        feat[13] = 0.2

    return feat


# ─── Run resampling ───────────────────────────────────────────────────────────

def resample_frames(frames: List[dict], target_len: int = DEFAULT_SEQ) -> np.ndarray:
    """
    Extract biometric features from frames and resample to exactly `target_len`
    steps using linear interpolation. Eliminates zero-padding artefacts.

    Returns np.ndarray of shape (target_len, INPUT_DIM), dtype float32.
    """
    n = len(frames)
    if n == 0:
        return np.zeros((target_len, INPUT_DIM), dtype=np.float32)

    raw = np.stack([extract_biometric_features(f) for f in frames], axis=0)

    if n == target_len:
        return raw.astype(np.float32)

    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_len)
    resampled = np.zeros((target_len, INPUT_DIM), dtype=np.float32)
    for d in range(INPUT_DIM):
        resampled[:, d] = np.interp(x_new, x_old, raw[:, d])
    return resampled


# ─── Model definition ─────────────────────────────────────────────────────────


class LSTMClassifier(_ModuleBase):
    """2-layer LSTM + FC head. Uses 14-dim biomechanical features."""

    def __init__(
        self,
        input_dim:   int = INPUT_DIM,
        hidden_dim:  int = HIDDEN_DIM,
        num_layers:  int = NUM_LAYERS,
        num_classes: int = NUM_CLASSES,
        dropout:     float = DROPOUT,
    ) -> None:
        if not _TORCH_AVAILABLE or torch is None or nn is None:
            raise ImportError("torch is required. pip install torch")
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
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if _TORCH_AVAILABLE:
            return super().__call__(*args, **kwargs)
        return self.forward(*args, **kwargs)

    def predict(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        if not _TORCH_AVAILABLE or torch is None:
            raise ImportError("torch is required")
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs  = torch.softmax(logits, dim=-1)
            idx    = torch.argmax(probs, dim=-1)
        return idx, probs

    @classmethod
    def from_checkpoint(cls, path: str | Path, **kwargs: Any) -> "LSTMClassifier":
        if not _TORCH_AVAILABLE or torch is None:
            raise ImportError("torch is required")
        ckpt        = torch.load(str(path), map_location="cpu", weights_only=False)
        state_dict  = ckpt.get("model_state_dict", ckpt)
        model_kw    = ckpt.get("model_kwargs", {})
        input_dim   = kwargs.pop("input_dim",   None) or model_kw.get("input_dim",   INPUT_DIM)
        hidden_dim  = kwargs.pop("hidden_dim",  None) or model_kw.get("hidden_dim",  HIDDEN_DIM)
        num_classes = kwargs.pop("num_classes", None) or model_kw.get("num_classes", NUM_CLASSES)
        model = cls(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes, **kwargs)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded LSTM from %s (%d-dim, %d classes)", path, input_dim, num_classes)
        return model

    def save_checkpoint(self, path: str | Path, **extra: Any) -> None:
        if not _TORCH_AVAILABLE or torch is None:
            raise ImportError("torch is required")
        ckpt = {
            "model_state_dict": self.state_dict(),
            "model_kwargs": {
                "input_dim":   self.input_dim,
                "hidden_dim":  self.hidden_dim,
                "num_classes": self.num_classes,
            },
            "classes": EXERCISE_CLASSES,
            **extra,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, str(path))
        logger.info("Checkpoint saved -> %s", path)


class KeypointLSTM(_ModuleBase):
    """Shruti's 132-dim Keypoint LSTM trained on 995 real exercise video clips."""

    def __init__(
        self,
        input_size:  int = 132,
        hidden_size: int = 64,
        num_layers:  int = 2,
        num_classes: int = 11,
        dropout:     float = 0.2,
    ) -> None:
        if not _TORCH_AVAILABLE or torch is None or nn is None:
            raise ImportError("torch is required. pip install torch")
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        self.input_dim = input_size
        self.hidden_dim = hidden_size
        self.num_classes = num_classes

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if _TORCH_AVAILABLE:
            return super().__call__(*args, **kwargs)
        return self.forward(*args, **kwargs)

    def predict(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        if not _TORCH_AVAILABLE or torch is None:
            raise ImportError("torch is required")
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs  = torch.softmax(logits, dim=-1)
            idx    = torch.argmax(probs, dim=-1)
        return idx, probs


def load_lstm_model(checkpoint_path: str | Path | None = None) -> Tuple[Any, str]:
    """
    Load the best available LSTM model.
    Prioritizes cv_pipeline/models/lstm.pt (11-class real video model, 995 clips)
    with fallback to cv_pipeline/models/lstm_best.pt (14-dim kinematic model).

    Returns
    -------
    (model, model_type) where model_type is "keypoint_132" or "kinematic_14".
    """
    if not _TORCH_AVAILABLE or torch is None:
        raise ImportError("torch is required")

    repo_root = Path(__file__).resolve().parents[2]
    p_shruti = repo_root / "cv_pipeline" / "models" / "lstm.pt"
    p_kinematic = repo_root / "cv_pipeline" / "models" / "lstm_best.pt"

    target_path = Path(checkpoint_path) if checkpoint_path else (p_shruti if p_shruti.exists() else p_kinematic)
    if not target_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {target_path}")

    ckpt = torch.load(str(target_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    # Check input dimension from weight_ih_l0
    w_ih = state.get("lstm.weight_ih_l0")
    if w_ih is not None and w_ih.shape[1] == 132:
        fc_w = state.get("fc.weight", state.get("head.3.weight"))
        num_classes = fc_w.shape[0] if fc_w is not None else 11
        hidden_size = w_ih.shape[0] // 4
        model = KeypointLSTM(input_size=132, hidden_size=hidden_size, num_layers=2, num_classes=num_classes)
        model.load_state_dict(state)
        model.eval()
        logger.info("Loaded KeypointLSTM from %s (132-dim normalized, %d classes)", target_path.name, num_classes)
        return model, "keypoint_132"
    else:
        model = LSTMClassifier.from_checkpoint(target_path)
        model.eval()
        logger.info("Loaded LSTMClassifier from %s (14-dim kinematic, %d classes)", target_path.name, model.num_classes)
        return model, "kinematic_14"


# ─── Per-run classification (primary inference API) ──────────────────────────

def classify_run(
    frames: List[dict],
    model: Any,
    *,
    seq_len: int = 30,
) -> Tuple[str, float]:
    """
    Classify a single active run as one exercise label.
    Supports both KeypointLSTM (132-dim normalized) and LSTMClassifier (14-dim).
    """
    if not _TORCH_AVAILABLE or torch is None or not frames:
        return "unknown", 0.0

    is_132 = getattr(model, "input_dim", 14) == 132 or isinstance(model, KeypointLSTM)

    if is_132:
        # Extract normalized 132-dim features
        coords_list = [extract_normalized_132_features(f) for f in frames]
        target_len = min(seq_len, max(len(coords_list), 1))
        indices = np.linspace(0, len(coords_list) - 1, target_len, dtype=int)
        resampled = np.array([coords_list[i] for i in indices], dtype=np.float32)
        x = torch.from_numpy(resampled).unsqueeze(0)  # (1, target_len, 132)

        idx, probs = model.predict(x)
        c_idx = int(idx[0].item())
        conf = float(probs[0, c_idx].item())

        raw_label = SHRUTI_CLASSES[c_idx] if c_idx < len(SHRUTI_CLASSES) else "unknown"
        canonical_label = SHRUTI_TO_CANONICAL.get(raw_label, raw_label.lower().replace("-", "").replace(" ", "_"))
        return canonical_label, conf
    else:
        resampled = resample_frames(frames, target_len=15)
        x = torch.from_numpy(resampled).unsqueeze(0)   # (1, 15, 14)
        idx, probs = model.predict(x)
        c_idx = int(idx[0].item())
        conf = float(probs[0, c_idx].item())
        label = EXERCISE_CLASSES[c_idx] if c_idx < len(EXERCISE_CLASSES) else "unknown"
        return label, conf


# ─── Legacy sliding-window helpers (backward compatibility) ──────────────────

def frames_to_windows(
    frames: List[dict],
    *,
    seq_len: int = DEFAULT_SEQ,
    stride:  int = DEFAULT_STRIDE,
) -> Tuple[np.ndarray, List[int]]:
    """Slice frames into overlapping windows using biometric features."""
    n_frames = len(frames)
    if n_frames == 0:
        return np.empty((0, seq_len, INPUT_DIM), dtype=np.float32), []

    X = np.stack([extract_biometric_features(f) for f in frames], axis=0)

    windows: List[np.ndarray] = []
    start_indices: List[int] = []

    for start in range(0, max(1, n_frames - seq_len + 1), stride):
        end = start + seq_len
        if end > n_frames:
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
    model: LSTMClassifier,
    *,
    seq_len: int = DEFAULT_SEQ,
    stride:  int = DEFAULT_STRIDE,
    fps:     float = 2.0,
) -> List[dict]:
    """
    Sliding-window LSTM inference. Kept for backward compatibility.
    Prefer classify_run() for new code.
    """
    if not _TORCH_AVAILABLE or torch is None:
        raise ImportError("torch is required for predict_sequence.")

    windows, start_indices = frames_to_windows(frames, seq_len=seq_len, stride=stride)
    if len(windows) == 0:
        return []

    x = torch.from_numpy(windows)
    class_idx, probs = model.predict(x)

    results = []
    for w_idx, (start_f, c_idx, prob_vec) in enumerate(
        zip(start_indices, class_idx.tolist(), probs.tolist())
    ):
        end_f  = min(start_f + seq_len, len(frames))
        label  = EXERCISE_CLASSES[c_idx] if c_idx < len(EXERCISE_CLASSES) else "unknown"
        conf   = float(prob_vec[c_idx])
        start_t = float(frames[start_f].get("timestamp", start_f / fps)) if start_f < len(frames) else start_f / fps
        end_idx = min(end_f - 1, len(frames) - 1)
        end_t   = float(frames[end_idx].get("timestamp", end_f / fps)) if end_idx >= 0 else end_f / fps
        results.append({
            "window_index": w_idx,
            "start_frame":  start_f,
            "end_frame":    end_f,
            "start_time":   round(start_t, 3),
            "end_time":     round(end_t, 3),
            "label":        label,
            "confidence":   round(conf, 4),
        })
    return results
