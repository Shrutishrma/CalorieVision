#!/usr/bin/env python3
"""
cv_pipeline/data/collect_training_data.py
────────────────────────────────────────────────────────────────────────────────
Utility script to collect a training dataset of pose keypoints for the LSTM.
Reads a CSV of [youtube_url, label, start_sec, end_sec].
Downloads each clip using yt-dlp, extracts keypoints via MediaPipe,
and saves the result to dataset/{label}/{video_id}_{start}_{end}_keypoints.json.
"""
import csv
import json
import os
import subprocess
from pathlib import Path

# Try importing the keypoint extractor
try:
    from cv_pipeline.keypoints.extract_keypoints import extract_keypoints
except ImportError:
    print("Must be run from repository root: python -m cv_pipeline.data.collect_training_data")
    exit(1)

def extract_video_id(url: str) -> str:
    if "v=" in url:
        return url.split("v=")[1].split("&")[0].split("?")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    if "shorts/" in url:
        return url.split("shorts/")[1].split("?")[0]
    return url.strip()

def download_clip(url: str, video_id: str, start_sec: float, end_sec: float, out_path: Path):
    """Download a specific segment of a youtube video."""
    print(f"Downloading {video_id} from {start_sec}s to {end_sec}s...")
    
    # We use yt-dlp's --download-sections to avoid downloading the whole video
    # Note: Requires ffmpeg installed
    import sys
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--quiet",
        "-f", "bestvideo[height<=480][ext=mp4]",
        "--download-sections", f"*{start_sec}-{end_sec}",
        "-o", str(out_path),
        url
    ]
    subprocess.run(cmd, check=True)

def process_csv(csv_path: str, out_dir: str):
    base_dir = Path(out_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row['youtube_url']
            label = row['label']
            start = float(row['start_sec'])
            end = float(row['end_sec'])
            
            video_id = extract_video_id(url)
            
            label_dir = base_dir / label
            label_dir.mkdir(exist_ok=True)
            
            clip_name = f"{video_id}_{int(start)}_{int(end)}"
            vid_path = label_dir / f"{clip_name}.mp4"
            kps_path = label_dir / f"{clip_name}_keypoints.json"
            
            if kps_path.exists():
                print(f"Skipping {clip_name}, already exists.")
                continue
                
            try:
                if not vid_path.exists():
                    download_clip(url, video_id, start, end, vid_path)
                    
                print(f"Extracting keypoints for {clip_name}...")
                kps = extract_keypoints(str(vid_path), sample_fps=30.0) # High FPS for training
                
                with open(kps_path, 'w', encoding='utf-8') as kf:
                    json.dump(kps, kf)
                    
                print(f"Saved {len(kps)} frames to {kps_path}")
                
                # Clean up video clip to save space
                vid_path.unlink()
                
            except Exception as e:
                print(f"Failed to process {url}: {e}")

if __name__ == "__main__":
    import sys
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "dataset/training_clips.csv"
    process_csv(csv_file, "dataset/processed_keypoints")
