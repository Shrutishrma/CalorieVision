"""
cv_pipeline/eval/eval_baseline.py
────────────────────────────────────────────────────────────────────────────────
Evaluation script for the MajorityClassPredictor and KNNBaselineClassifier
baselines defined in cv-pipeline/models/baseline.py.

⚠️  IMPORTANT — ABOUT THE DATA SOURCE  ⚠️
─────────────────────────────────────────
If no real labelled keypoint JSON file is provided via --data, this script
generates SYNTHETIC data (random landmark coordinates + random class labels)
solely to verify that the pipeline interface works end-to-end.

SYNTHETIC ACCURACY NUMBERS ARE MEANINGLESS.  They will be prominently flagged
in both the console output and any saved results file so they can never be
mistaken for real evaluation figures.

Usage
-----
# Against synthetic data (interface smoke-test only):
    python cv-pipeline/eval/eval_baseline.py

# Against real labelled data:
    python cv-pipeline/eval/eval_baseline.py --data path/to/keypoints.json --labels path/to/labels.json

Label file format
-----------------
A JSON array of strings, one per frame, in the same order as the keypoint
JSON file produced by extract_keypoints.py.

    ["squat", "squat", "squat", "standing", "pushup", ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

# ── Ensure repo root is importable ───────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cv_pipeline.models.baseline import (
    KNNBaselineClassifier,
    MajorityClassPredictor,
    UNKNOWN_LABEL,
)

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Synthetic data banner ────────────────────────────────────────────────────

_SYNTH_BANNER = textwrap.dedent("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ⚠️  SYNTHETIC DATA — NOT A REAL ACCURACY NUMBER  ⚠️                        ║
║                                                                              ║
║  This run used procedurally generated landmark coordinates and class        ║
║  labels.  The reported accuracy figures are statistically meaningless.      ║
║  They exist ONLY to verify that the baseline pipeline interface is correct. ║
║                                                                              ║
║  To get real accuracy numbers, provide --data and --labels pointing to      ║
║  actual annotated keypoint data.                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
""").strip()

# ─── Synthetic data generator ─────────────────────────────────────────────────

_EXERCISE_CLASSES = ["squat", "pushup", "jumping_jack", "lunge", "plank"]


def _make_synthetic_frames(n: int = 300, rng: np.random.Generator | None = None) -> list[dict]:
    """
    Generate `n` fake frame dicts in the extract_keypoints output format.
    ~10 % of frames have no pose (landmarks=[]) to exercise the missing-pose
    handling path.
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    frames = []
    for i in range(n):
        no_pose = rng.random() < 0.10
        if no_pose:
            landmarks = []
            pose_detected = False
        else:
            # 33 landmarks, each with random x/y/z in [0, 1] / [-0.5, 0.5]
            landmarks = [
                {
                    "index": j,
                    "name": f"lm_{j}",
                    "x": float(rng.random()),
                    "y": float(rng.random()),
                    "z": float(rng.random() - 0.5),
                    "visibility": float(rng.random()),
                }
                for j in range(33)
            ]
            pose_detected = True

        frames.append(
            {
                "frame_index": i,
                "timestamp": round(i / 30.0, 6),
                "pose_detected": pose_detected,
                "landmarks": landmarks,
            }
        )
    return frames


def _make_synthetic_labels(frames: list[dict], rng: np.random.Generator | None = None) -> list[str]:
    """Assign a random class label to every frame."""
    if rng is None:
        rng = np.random.default_rng(seed=99)
    return [
        rng.choice(_EXERCISE_CLASSES)  # type: ignore[arg-type]
        for _ in frames
    ]


# ─── Evaluation helpers ───────────────────────────────────────────────────────

def _accuracy_report(
    name: str,
    preds: list[str],
    labels: list[str],
) -> dict:
    """Compute and print a per-class accuracy breakdown."""
    correct = total = 0
    per_class: dict[str, dict[str, int]] = {}
    for pred, true in zip(preds, labels):
        if pred == UNKNOWN_LABEL:
            continue
        total += 1
        per_class.setdefault(true, {"correct": 0, "total": 0})
        per_class[true]["total"] += 1
        if pred == true:
            correct += 1
            per_class[true]["correct"] += 1

    overall_acc = correct / total if total else 0.0

    print(f"\n── {name} ──────────────────────────────────────────────────────")
    print(f"   Overall accuracy: {overall_acc:.4f}  ({correct}/{total} frames)")
    print(f"   Per-class breakdown:")
    for cls in sorted(per_class):
        c = per_class[cls]["correct"]
        t = per_class[cls]["total"]
        print(f"     {cls:<20s}: {c}/{t}  ({c/t:.2%})")

    return {
        "model": name,
        "overall_accuracy": overall_acc,
        "evaluated_frames": total,
        "per_class": {cls: v for cls, v in per_class.items()},
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CalorieVision baseline classifiers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default=None,
        help=(
            "Path to keypoints JSON produced by extract_keypoints.py. "
            "If omitted, synthetic data is generated (interface test only)."
        ),
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Path to per-frame label JSON (list of strings, same length as frames).",
    )
    parser.add_argument(
        "--test-split", type=float, default=0.2,
        help="Fraction of data to reserve for evaluation (train/test split).",
    )
    parser.add_argument(
        "--knn-k", type=int, default=5,
        help="Number of neighbours for k-NN baseline.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Path to save results as JSON. If omitted, prints only.",
    )
    args = parser.parse_args(argv)

    # ── Determine data source ──────────────────────────────────────────────────
    is_synthetic = args.data is None or args.labels is None

    if is_synthetic:
        print("\n" + _SYNTH_BANNER + "\n")
        logger.warning(
            "⚠️  SYNTHETIC DATA — NOT A REAL ACCURACY NUMBER  ⚠️  "
            "No --data / --labels provided; using procedurally generated data."
        )
        rng = np.random.default_rng(seed=42)
        frames = _make_synthetic_frames(n=500, rng=rng)
        labels = _make_synthetic_labels(frames, rng=rng)
    else:
        data_path   = Path(args.data)
        labels_path = Path(args.labels)

        with open(data_path, encoding="utf-8") as fh:
            raw = json.load(fh)
        # Accept either the full extract_keypoints output dict or a bare list
        frames = raw["frames"] if isinstance(raw, dict) else raw

        with open(labels_path, encoding="utf-8") as fh:
            labels = json.load(fh)

        if len(frames) != len(labels):
            logger.error(
                "Frame count (%d) != label count (%d). Aborting.",
                len(frames), len(labels),
            )
            sys.exit(1)

    # ── Train / test split ────────────────────────────────────────────────────
    split_idx = int(len(frames) * (1 - args.test_split))
    train_frames, test_frames = frames[:split_idx], frames[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]

    print(f"\nDataset  : {'SYNTHETIC' if is_synthetic else 'REAL'}")
    print(f"Total    : {len(frames)} frames")
    print(f"Train    : {len(train_frames)} frames")
    print(f"Test     : {len(test_frames)} frames")

    # ── Majority-class baseline ───────────────────────────────────────────────
    majority = MajorityClassPredictor()
    majority.fit(train_frames, train_labels)
    majority_preds = majority.predict(test_frames)
    majority_report = _accuracy_report("MajorityClassPredictor", majority_preds, test_labels)

    # ── k-NN baseline ─────────────────────────────────────────────────────────
    knn = KNNBaselineClassifier(k=args.knn_k, metric="cosine")
    knn.fit(train_frames, train_labels)
    knn_preds = knn.predict(test_frames)
    knn_report = _accuracy_report(f"KNNBaselineClassifier (k={args.knn_k})", knn_preds, test_labels)

    # ── Reminder banner (printed again after results) ─────────────────────────
    if is_synthetic:
        print("\n" + _SYNTH_BANNER + "\n")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": "SYNTHETIC" if is_synthetic else str(args.data),
        "SYNTHETIC_WARNING": (
            "⚠️  SYNTHETIC DATA — NOT A REAL ACCURACY NUMBER.  "
            "These results are meaningless without real labelled data."
            if is_synthetic else None
        ),
        "train_frames": len(train_frames),
        "test_frames": len(test_frames),
        "models": [majority_report, knn_report],
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\n[CalorieVision] Results saved → {out_path}")
        if is_synthetic:
            print(
                "[CalorieVision] ⚠️  SYNTHETIC DATA — NOT A REAL ACCURACY NUMBER  ⚠️"
            )
    else:
        print("\n[CalorieVision] (pass --out <file.json> to save results)")


if __name__ == "__main__":
    main()
