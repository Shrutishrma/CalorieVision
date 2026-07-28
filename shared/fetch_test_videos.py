"""
shared/fetch_test_videos.py
────────────────────────────────────────────────────────────────────────────────
Fetches test videos for smoke tests and the difficulty-tagged evaluation set.

IMPORTANT PRODUCT CONTEXT & ARCHITECTURE NOTE:
The actual CalorieVision app takes a YouTube link as primary input (not uploaded files).
This testing script mirrors the real production input path exactly: it reads YouTube URLs
from `shared/test_videos_manifest.json` and calls the same `download_video()` function
from `fusion_pipeline.segmentation.download_video` that the real product uses to fetch a
user-submitted link.

No video files are ever committed to Git, shared via drive, or passed between teammates.
This script simply exercises the production YouTube ingestion pipeline to cache test videos
locally into a gitignored `shared/test-videos/` folder. Skip re-downloading if already present.

Usage
-----
    python shared/fetch_test_videos.py
    python shared/fetch_test_videos.py --force   # Re-download even if cached
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root is importable
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fusion_pipeline.segmentation.download_video import download_video, DownloadError


def fetch_all(manifest_path: Path | None = None, output_dir: Path | None = None, overwrite: bool = False) -> int:
    """
    Reads the manifest and downloads all listed YouTube videos via the production pipeline.

    Returns the number of videos successfully ready on disk.
    """
    if manifest_path is None:
        manifest_path = _REPO_ROOT / "shared" / "test_videos_manifest.json"
    if output_dir is None:
        output_dir = _REPO_ROOT / "shared" / "test-videos"

    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found at: {manifest_path}", file=sys.stderr)
        return 0

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fetch_test_videos] Processing {len(manifest)} test video entries from manifest...")
    ready_count = 0

    for idx, entry in enumerate(manifest, start=1):
        vid_id = entry.get("youtube_id")
        url = entry.get("url") or f"https://youtu.be/{vid_id}"
        label = entry.get("label", vid_id)

        target_file = output_dir / f"{vid_id}.mp4"
        if target_file.exists() and not overwrite:
            print(f"  [{idx}/{len(manifest)}] [CACHED] {label} ({vid_id}.mp4)")
            ready_count += 1
            continue

        print(f"  [{idx}/{len(manifest)}] [FETCH] Downloading via production pipeline: {label} ({url})...")
        try:
            download_video(url, output_dir=str(output_dir), overwrite=overwrite)
            print(f"  [{idx}/{len(manifest)}] [OK] Successfully fetched: {vid_id}.mp4")
            ready_count += 1
        except RuntimeError as e:
            # Captures ffmpeg missing error or yt-dlp failures
            print(f"  [{idx}/{len(manifest)}] [ERROR] Download failed for {label}: {e.args[0].split(chr(10))[0]}", file=sys.stderr)
        except Exception as e:
            print(f"  [{idx}/{len(manifest)}] [ERROR] Unexpected error for {label}: {e}", file=sys.stderr)

    print(f"\n[fetch_test_videos] Summary: {ready_count}/{len(manifest)} test videos ready in {output_dir.relative_to(_REPO_ROOT)}/")
    return ready_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch test videos via production YouTube download pipeline.")
    parser.add_argument("--force", "-f", action="store_true", help="Force re-download even if cached.")
    args = parser.parse_args()
    
    fetch_all(overwrite=args.force)


if __name__ == "__main__":
    main()
