#!/usr/bin/env python3
"""
cv_pipeline/data/distill_dataset.py
────────────────────────────────────────────────────────────────────────────────
Generates an LSTM training dataset by using the Pose Heuristic Classifier 
to auto-label existing local videos.
"""
import json
import argparse
from pathlib import Path

# Adjust imports to allow running from repository root
import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cv_pipeline.keypoints.extract_keypoints import extract_keypoints
from cv_pipeline.motion.motion_filter import segment_runs
from cv_pipeline.models.pose_heuristic_classifier import classify_segment

def distill_dataset(video_dir: str, out_data: str, out_labels: str):
    vdir = Path(video_dir)
    all_frames = []
    all_labels = []

    allowed_videos = ["IODxDxX7oi4.mp4", "aclHkVaku9U.mp4", "dJlFmxiL11s.mp4", "jNQXAC9IVRw.mp4"]
    for video_path in vdir.glob("*.mp4"):
        if video_path.name not in allowed_videos:
            continue
        print(f"Processing {video_path.name}...")
        try:
            # 1. Extract Keypoints at 10 FPS to save time
            frames = extract_keypoints(str(video_path), sample_fps=10.0)
            print(f"  Extracted {len(frames)} frames.")
            
            # 2. Segment into active runs
            # We use conservative settings for high-confidence labels
            runs = segment_runs(frames, threshold=0.015, min_active_secs=1.5, min_gap_secs=1.0)
            print(f"  Found {len(runs)} active runs.")
            
            # Initialize all labels to "rest"
            video_labels = ["rest"] * len(frames)
            
            # 3. Classify runs
            for run in runs:
                run_frames = run["frames"]
                label, conf = classify_segment(run_frames)
                print(f"    Run [{run['start_time']:.1f}s - {run['end_time']:.1f}s]: {label} (conf: {conf:.2f})")
                
                # If the heuristic is confident, label the frames
                if conf > 0.5 and label != "unknown":
                    for f in run_frames:
                        # Find the index of this frame
                        # Since extract_keypoints gives sequential frames, we can match by timestamp
                        # or just by keeping track of the original index.
                        # Wait, extract_keypoints output contains frame_index? Let's check.
                        idx = f.get("frame_index")
                        if idx is not None and idx < len(video_labels):
                            video_labels[idx] = label
                        else:
                            # Fallback: find by timestamp if frame_index is missing
                            idx = int(f["timestamp"] * 30.0)
                            if idx < len(video_labels):
                                video_labels[idx] = label

            # 4. Append to dataset
            all_frames.extend(frames)
            all_labels.extend(video_labels)
            
        except Exception as e:
            print(f"  Failed on {video_path.name}: {e}")

    out_data_path = Path(out_data)
    out_labels_path = Path(out_labels)
    
    out_data_path.parent.mkdir(parents=True, exist_ok=True)
    out_labels_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_data_path, 'w', encoding='utf-8') as f:
        json.dump({"frames": all_frames}, f)
        
    with open(out_labels_path, 'w', encoding='utf-8') as f:
        json.dump(all_labels, f)
        
    print(f"\n✅ Distilled dataset of {len(all_frames)} frames.")
    print(f"Saved data to {out_data_path}")
    print(f"Saved labels to {out_labels_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", default="shared/test-videos")
    parser.add_argument("--out-data", default="dataset/train_data.json")
    parser.add_argument("--out-labels", default="dataset/train_labels.json")
    args = parser.parse_args()
    
    distill_dataset(args.video_dir, args.out_data, args.out_labels)
