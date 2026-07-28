"""
cv_pipeline/models/train_lstm.py
────────────────────────────────────────────────────────────────────────────────
Training harness for the LSTMClassifier.

⚠️  SYNTHETIC DATA WARNING  ⚠️
──────────────────────────────
If no real --data / --labels are provided, this script trains on PROCEDURALLY
GENERATED random keypoint sequences.  The resulting accuracy numbers are
statistically meaningless and exist ONLY to verify that the training pipeline
works end-to-end.

To get real accuracy numbers, provide:
    --data   path/to/keypoints.json   (extract_keypoints.py output)
    --labels path/to/labels.json      (list of exercise class strings, one per frame)

Evaluation
----------
After training, the script evaluates:
  1. LSTM (this model)
  2. MajorityClassPredictor (existing baseline)
  3. KNNBaselineClassifier  (existing baseline, k=5)

...on the SAME held-out test split, so results are directly comparable.
Improvement over baseline is printed explicitly.

Usage
-----
    # Synthetic smoke-test:
    python cv_pipeline/models/train_lstm.py

    # Real data:
    python cv_pipeline/models/train_lstm.py \\
        --data shared/test-videos/jNQXAC9IVRw_keypoints.json \\
        --labels shared/test-videos/jNQXAC9IVRw_labels.json  \\
        --epochs 50 --checkpoint cv_pipeline/models/lstm_best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ─── Synthetic data banner ────────────────────────────────────────────────────

_SYNTH_BANNER = textwrap.dedent("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ⚠️  SYNTHETIC TRAINING DATA — ACCURACY NUMBERS ARE MEANINGLESS  ⚠️         ║
║                                                                              ║
║  This run used procedurally generated keypoint sequences and class labels.  ║
║  The reported accuracy figures DO NOT reflect real-world performance.       ║
║  They exist ONLY to verify that the training pipeline works end-to-end.    ║
║                                                                              ║
║  Provide --data and --labels for real training.                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""").strip()

# ─── Synthetic data generation ────────────────────────────────────────────────

from cv_pipeline.models.lstm_classifier import (
    EXERCISE_CLASSES, INPUT_DIM, DEFAULT_SEQ, DEFAULT_STRIDE, frames_to_windows,
    LSTMClassifier, NUM_CLASSES,
)


def _make_synthetic_dataset(
    n_windows: int = 500,
    seq_len:   int = DEFAULT_SEQ,
    rng:       np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random (X, y) tensors for smoke-testing.

    Returns
    -------
    X : np.ndarray (n_windows, seq_len, INPUT_DIM)
    y : np.ndarray (n_windows,) int64 class indices
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    X = rng.random((n_windows, seq_len, INPUT_DIM)).astype(np.float32)
    y = rng.integers(0, NUM_CLASSES, size=n_windows).astype(np.int64)
    return X, y


def _real_dataset(
    kp_path:    Path,
    label_path: Path,
    seq_len:    int = DEFAULT_SEQ,
    stride:     int = DEFAULT_STRIDE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load real keypoints + labels and slice into windows."""
    with open(kp_path, encoding="utf-8") as fh:
        raw = json.load(fh)
    frames = raw["frames"] if isinstance(raw, dict) else raw

    with open(label_path, encoding="utf-8") as fh:
        per_frame_labels: List[str] = json.load(fh)

    if len(frames) != len(per_frame_labels):
        raise ValueError(
            f"Frame count ({len(frames)}) != label count ({len(per_frame_labels)})"
        )

    # Build windows and assign label = most common label in each window
    windows, start_indices = frames_to_windows(frames, seq_len=seq_len, stride=stride)
    if len(windows) == 0:
        raise ValueError("No windows produced — video too short?")

    y = []
    from collections import Counter
    for start in start_indices:
        end    = min(start + seq_len, len(per_frame_labels))
        window_labels = per_frame_labels[start:end]
        # Map to class indices, skip unknowns
        valid = [EXERCISE_CLASSES.index(lb) for lb in window_labels
                 if lb in EXERCISE_CLASSES]
        if valid:
            y.append(Counter(valid).most_common(1)[0][0])
        else:
            y.append(0)  # fallback to first class

    return windows, np.array(y, dtype=np.int64)


# ─── Training loop ────────────────────────────────────────────────────────────

def train(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_split:  float = 0.2,
    epochs:      int   = 30,
    batch_size:  int   = 32,
    lr:          float = 1e-3,
    patience:    int   = 10,
    checkpoint:  Path | None = None,
) -> dict:
    """
    Train the LSTM classifier and return a results dict.

    Parameters
    ----------
    X            : (n_windows, seq_len, input_dim) float32
    y            : (n_windows,) int64 class indices
    test_split   : fraction reserved for evaluation
    epochs       : max training epochs
    batch_size   : mini-batch size
    lr           : Adam learning rate
    patience     : early-stopping patience (epochs without val loss improvement)
    checkpoint   : path to save best model (None = don't save)

    Returns
    -------
    dict with keys: lstm_test_acc, best_epoch, train_acc_final
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        raise ImportError("torch is required for training. pip install torch")

    n = len(X)
    split_idx = int(n * (1 - test_split))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    X_tr = torch.from_numpy(X_train)
    y_tr = torch.from_numpy(y_train)
    X_te = torch.from_numpy(X_test)
    y_te = torch.from_numpy(y_test)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    model     = LSTMClassifier()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_epoch    = 0
    patience_ctr  = 0
    best_state    = None

    print(f"\n{'='*60}")
    print(f"LSTM Training  [{n} windows → train={split_idx} / test={n-split_idx}]")
    print(f"Epochs={epochs}  Batch={batch_size}  LR={lr}  Patience={patience}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        for xb, yb in loader:
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()

        # ── Validation (on test set — for early stopping only) ────────────────
        model.eval()
        with torch.no_grad():
            val_logits = model(X_te)
            val_loss   = criterion(val_logits, y_te).item()
            val_acc    = (val_logits.argmax(1) == y_te).float().mean().item()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch    = epoch
            patience_ctr  = 0
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_logits = model(X_te)
        test_acc    = (test_logits.argmax(1) == y_te).float().mean().item()
        train_logits = model(X_tr)
        train_acc    = (train_logits.argmax(1) == y_tr).float().mean().item()

    if checkpoint:
        model.save_checkpoint(checkpoint, best_epoch=best_epoch, test_acc=test_acc)

    return {
        "lstm_test_acc":  test_acc,
        "lstm_train_acc": train_acc,
        "best_epoch":     best_epoch,
    }


# ─── Baseline comparison ──────────────────────────────────────────────────────

def _baseline_accs(
    frames_train: List[dict],
    labels_train: List[str],
    frames_test:  List[dict],
    labels_test:  List[str],
    k:            int = 5,
) -> dict:
    """Evaluate both baselines on the same split."""
    from cv_pipeline.models.baseline import MajorityClassPredictor, KNNBaselineClassifier, UNKNOWN_LABEL

    def acc(preds, truths):
        correct = total = 0
        for p, t in zip(preds, truths):
            if p == UNKNOWN_LABEL:
                continue
            total += 1
            correct += int(p == t)
        return correct / total if total else 0.0

    maj = MajorityClassPredictor()
    maj.fit(frames_train, labels_train)
    maj_acc = acc(maj.predict(frames_test), labels_test)

    knn = KNNBaselineClassifier(k=k)
    knn.fit(frames_train, labels_train)
    knn_acc = acc(knn.predict(frames_test), labels_test)

    return {"majority_acc": maj_acc, "knn_acc": knn_acc}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train CalorieVision LSTM classifier and compare against baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data",   default=None, help="Keypoints JSON (extract_keypoints output)")
    parser.add_argument("--labels", default=None, help="Per-frame label JSON (list of strings)")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int,   default=10)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--knn-k",      type=int,   default=5)
    parser.add_argument(
        "--checkpoint", default=str(Path(__file__).parent / "lstm_best.pt"),
        help="Where to save the best model checkpoint.",
    )
    parser.add_argument(
        "--no-checkpoint", action="store_true",
        help="Do not save a checkpoint (useful for smoke-testing).",
    )
    args = parser.parse_args(argv)

    is_synthetic = args.data is None or args.labels is None

    # ── Data loading ──────────────────────────────────────────────────────────
    if is_synthetic:
        print("\n" + _SYNTH_BANNER + "\n")
        X, y = _make_synthetic_dataset()

        # For baseline comparison we also need frame/label lists
        # Build minimal frame dicts from the synthetic X windows
        rng = np.random.default_rng(seed=99)
        n = len(X)
        split_idx = int(n * (1 - args.test_split))
        # Synthetic frames for baseline (each "frame" = first step of each window)
        def _xy_to_frames(X_sub):
            frames = []
            for seq in X_sub:
                vec = seq[0]  # first frame of window
                lms = [
                    {"index": j, "name": f"lm_{j}",
                     "x": float(vec[j*3]),
                     "y": float(vec[j*3+1]),
                     "z": float(vec[j*3+2]),
                     "visibility": 1.0}
                    for j in range(33)
                ]
                frames.append({"frame_index": 0, "timestamp": 0.0,
                                "pose_detected": True, "landmarks": lms})
            return frames

        frames_train = _xy_to_frames(X[:split_idx])
        frames_test  = _xy_to_frames(X[split_idx:])
        labels_train = [EXERCISE_CLASSES[i] for i in y[:split_idx]]
        labels_test  = [EXERCISE_CLASSES[i] for i in y[split_idx:]]
    else:
        kp_path    = Path(args.data)
        label_path = Path(args.labels)
        X, y = _real_dataset(kp_path, label_path)

        # Build frame/label lists from raw files for baseline comparison
        with open(kp_path, encoding="utf-8") as fh:
            raw = json.load(fh)
        all_frames = raw["frames"] if isinstance(raw, dict) else raw
        with open(label_path, encoding="utf-8") as fh:
            all_labels: list[str] = json.load(fh)

        n = len(all_frames)
        sf = int(n * (1 - args.test_split))
        frames_train, frames_test = all_frames[:sf], all_frames[sf:]
        labels_train, labels_test = all_labels[:sf],  all_labels[sf:]

    # ── Train LSTM ────────────────────────────────────────────────────────────
    ckpt = None if args.no_checkpoint else Path(args.checkpoint)
    lstm_result = train(
        X, y,
        test_split=args.test_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        checkpoint=ckpt,
    )

    # ── Baseline comparison ───────────────────────────────────────────────────
    print("\nEvaluating baselines on same test split …")
    baseline = _baseline_accs(frames_train, labels_train, frames_test, labels_test, k=args.knn_k)

    # ── Print final report ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  Data source       : {'SYNTHETIC ⚠️' if is_synthetic else str(args.data)}")
    print(f"  Total windows     : {len(X)}")
    print(f"  Best epoch (LSTM) : {lstm_result['best_epoch']}")
    print()
    print(f"  MajorityClass acc : {baseline['majority_acc']:.4f}")
    print(f"  k-NN (k={args.knn_k}) acc   : {baseline['knn_acc']:.4f}")
    print(f"  LSTM test acc     : {lstm_result['lstm_test_acc']:.4f}")
    print()
    lstm_vs_knn = lstm_result['lstm_test_acc'] - baseline['knn_acc']
    lstm_vs_maj = lstm_result['lstm_test_acc'] - baseline['majority_acc']
    print(f"  LSTM vs k-NN      : {lstm_vs_knn:+.4f}")
    print(f"  LSTM vs Majority  : {lstm_vs_maj:+.4f}")
    print(f"{'='*60}")

    if is_synthetic:
        print("\n" + _SYNTH_BANNER + "\n")

    if ckpt and not args.no_checkpoint:
        print(f"\nCheckpoint saved → {ckpt}")


if __name__ == "__main__":
    main()
