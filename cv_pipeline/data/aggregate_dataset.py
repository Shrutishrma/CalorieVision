#!/usr/bin/env python3
"""
cv_pipeline/data/aggregate_dataset.py
────────────────────────────────────────────────────────────────────────────────
Aggregates individual clip keypoints into a single train_data.json and train_labels.json
for train_lstm.py.
"""
import json
import argparse
from pathlib import Path

def aggregate_dataset(input_dir: str, out_data: str, out_labels: str):
    base_dir = Path(input_dir)
    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist.")
        return

    all_frames = []
    all_labels = []

    # Iterate through all label subdirectories
    for label_dir in base_dir.iterdir():
        if not label_dir.is_dir():
            continue
            
        label = label_dir.name
        
        # Iterate through all JSON files in the label directory
        for kp_file in label_dir.glob("*_keypoints.json"):
            try:
                with open(kp_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                frames = data.get("frames", data if isinstance(data, list) else [])
                
                # Append frames and create a label for each frame
                all_frames.extend(frames)
                all_labels.extend([label] * len(frames))
                print(f"Added {len(frames)} frames from {kp_file.name} as '{label}'")
                
            except Exception as e:
                print(f"Error reading {kp_file}: {e}")

    if not all_frames:
        print("No frames found!")
        return
        
    out_data_path = Path(out_data)
    out_labels_path = Path(out_labels)
    
    out_data_path.parent.mkdir(parents=True, exist_ok=True)
    out_labels_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_data_path, 'w', encoding='utf-8') as f:
        json.dump({"frames": all_frames}, f)
        
    with open(out_labels_path, 'w', encoding='utf-8') as f:
        json.dump(all_labels, f)
        
    print(f"\n✅ Aggregated {len(all_frames)} frames total.")
    print(f"Saved data to {out_data_path}")
    print(f"Saved labels to {out_labels_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset/processed_keypoints")
    parser.add_argument("--out-data", default="dataset/train_data.json")
    parser.add_argument("--out-labels", default="dataset/train_labels.json")
    args = parser.parse_args()
    
    aggregate_dataset(args.input, args.out_data, args.out_labels)
