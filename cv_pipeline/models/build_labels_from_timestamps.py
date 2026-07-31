"""
cv_pipeline/models/build_labels_from_timestamps.py
────────────────────────────────────────────────────────────────────────────────
Utility script to generate a frame-by-frame `labels.json` for train_lstm.py
from a keypoints file and a list of timestamp ranges.

Usage:
  python build_labels_from_timestamps.py \
      --keypoints shared/test-videos/demo_keypoints.json \
      --ranges '0.0,45.0,squat; 45.0,60.0,rest; 60.0,105.0,pushup' \
      --out shared/test-videos/demo_labels.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_labels_from_ranges(
    keypoints_path: str,
    ranges_str: str,
    output_path: str,
    default_label: str = "unknown",
):
    kp_file = Path(keypoints_path)
    with open(kp_file, encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames", data if isinstance(data, list) else [])
    num_frames = len(frames)

    # Parse ranges string "start,end,label; start,end,label"
    segments = []
    for item in ranges_str.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [p.strip() for p in item.split(",")]
        if len(parts) == 3:
            start_t, end_t, lbl = float(parts[0]), float(parts[1]), parts[2]
            segments.append((start_t, end_t, lbl))

    labels = []
    for i, f in enumerate(frames):
        ts = f.get("timestamp", i / 30.0)
        assigned = default_label
        for start_t, end_t, lbl in segments:
            if start_t <= ts <= end_t:
                assigned = lbl
                break
        labels.append(assigned)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    print(f"✅ Generated {len(labels)} per-frame labels -> {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate per-frame labels from timestamp ranges.")
    parser.add_argument("--keypoints", required=True, help="Path to keypoints JSON file")
    parser.add_argument("--ranges", required=True, help="Semicolon-separated ranges: '0,30,squat; 30,60,pushup'")
    parser.add_argument("--out", required=True, help="Output labels JSON file")
    args = parser.parse_args()

    build_labels_from_ranges(args.keypoints, args.ranges, args.out)
