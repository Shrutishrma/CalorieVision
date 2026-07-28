"""
cv_pipeline/eval/eval_set.py
────────────────────────────────────────────────────────────────────────────────
Difficulty-tagged evaluation set for CalorieVision (leMON Team 15).

⚠️  NOT OPTIONAL — EXPLICIT PROJECT DIFFERENTIATOR  ⚠️
This module is flagged as "most likely to be cut under time pressure."
It MUST NOT be deferred silently.  If scope needs to be cut, escalate
to the team lead (Shruti) explicitly rather than skipping this.

Purpose
-------
Holds the 15–20 real workout videos tagged by:
  • camera_angle   : "frontal" | "angled" | "overhead" | "mixed"
  • caption_present: True / False  (on-screen rep counter or label visible)
  • num_subjects   : 1 | 2 | 3+

These tags let us measure how difficult each video is for the pipeline and
report per-difficulty-tier accuracy (not just a single aggregate number).

Current status (2026-07-28)
---------------------------
STATUS: BLOCKED on ffmpeg install.
0 real videos loaded.  Scaffold is ready and waiting.

Filling this requires:
1.  winget install ffmpeg              (unblocks yt-dlp download)
2.  python smoke_test.py              (downloads + runs the 3 smoke videos)
3.  Manual annotation of ground-truth segment labels per video
4.  Add entries to EVAL_CATALOGUE below

The catalogue schema is stable — add entries freely once ffmpeg is installed.

Usage
-----
    from cv_pipeline.eval.eval_set import load_eval_set, EvalEntry

    entries = load_eval_set()
    for entry in entries:
        print(entry.video_id, entry.tags)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal

import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.schemas import Segment

logger = logging.getLogger(__name__)

# ─── Types ────────────────────────────────────────────────────────────────────

CameraAngle  = Literal["frontal", "angled", "overhead", "mixed"]
NumSubjects  = Literal[1, 2, 3]   # 3 = "3 or more"


@dataclass
class EvalTags:
    """Difficulty tags for a single video."""
    camera_angle:    CameraAngle
    caption_present: bool    # on-screen rep counter / label visible?
    num_subjects:    int     # 1 | 2 | 3+


@dataclass
class EvalEntry:
    """One entry in the difficulty-tagged evaluation catalogue."""
    video_id:       str           # 11-char YouTube ID
    url:            str           # full YouTube URL
    description:    str
    tags:           EvalTags
    ground_truth:   List[Segment] = field(default_factory=list)
    # Path to the locally downloaded video (set by the loader)
    local_path:     str | None    = None


# ─── Catalogue ────────────────────────────────────────────────────────────────
# Dynamically loaded from shared/test_videos_manifest.json (shared source of truth)
# No video files are committed or shared directly; all are fetched via YouTube URLs.

def _load_manifest_catalogue() -> List[EvalEntry]:
    manifest_path = _REPO_ROOT / "shared" / "test_videos_manifest.json"
    if not manifest_path.exists():
        return []
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = []
        for item in data:
            tags_dict = item.get("tags", {})
            tags = EvalTags(
                camera_angle=tags_dict.get("camera_angle", "frontal"),
                caption_present=tags_dict.get("caption_present", False),
                num_subjects=tags_dict.get("num_subjects", 1),
            )
            entries.append(EvalEntry(
                video_id=item["youtube_id"],
                url=item.get("url", f"https://youtu.be/{item['youtube_id']}"),
                description=item.get("description", item.get("label", "")),
                tags=tags,
                ground_truth=[],
            ))
        return entries
    except Exception as e:
        logger.warning("[eval_set] Error reading manifest: %s", e)
        return []

EVAL_CATALOGUE: List[EvalEntry] = _load_manifest_catalogue()


# ─── Loader ───────────────────────────────────────────────────────────────────

def load_eval_set(
    videos_dir: Path | None = None,
) -> List[EvalEntry]:
    """
    Return the evaluation catalogue, attaching local_path for entries whose
    video has been downloaded to `videos_dir`.

    Parameters
    ----------
    videos_dir : Path to the directory containing downloaded MP4s.
                 Defaults to shared/test-videos/.

    Returns
    -------
    list[EvalEntry]  (may be empty if no videos downloaded yet)
    """
    if videos_dir is None:
        videos_dir = _REPO_ROOT / "shared" / "test-videos"

    if not EVAL_CATALOGUE:
        logger.warning(
            "[eval_set] Evaluation catalogue is empty. "
            "Install ffmpeg, download videos, and add entries to EVAL_CATALOGUE in eval_set.py."
        )
        return []

    entries = []
    for entry in EVAL_CATALOGUE:
        candidate = videos_dir / f"{entry.video_id}.mp4"
        if candidate.exists():
            entry.local_path = str(candidate)
        else:
            logger.warning(
                "[eval_set] Video %s not found at %s — run smoke_test.py to download.",
                entry.video_id, candidate,
            )
        entries.append(entry)

    loaded = sum(1 for e in entries if e.local_path)
    logger.info(
        "[eval_set] %d / %d eval videos available on disk.",
        loaded, len(entries),
    )
    return entries


def eval_set_status() -> dict:
    """
    Return a summary dict suitable for reporting in status reports and CI.

    Example output::

        {
          "total_entries": 0,
          "available_on_disk": 0,
          "blocked_reason": "ffmpeg not installed — run: winget install ffmpeg",
          "tag_distribution": {}
        }
    """
    entries = load_eval_set()
    n_available = sum(1 for e in entries if e.local_path)

    tag_dist: dict = {}
    for entry in entries:
        angle = entry.tags.camera_angle
        tag_dist[angle] = tag_dist.get(angle, 0) + 1

    blocked_reason = None
    if not entries:
        blocked_reason = "Catalogue is empty — install ffmpeg then add videos to EVAL_CATALOGUE"

    return {
        "total_entries":     len(entries),
        "available_on_disk": n_available,
        "blocked_reason":    blocked_reason,
        "tag_distribution":  tag_dist,
    }


# ─── CLI / quick status ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import pprint
    status = eval_set_status()
    print("\n── Evaluation Set Status ──")
    pprint.pprint(status)
