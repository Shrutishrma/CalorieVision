"""
eval/run_manifest_pass.py
────────────────────────────────────────────────────────────────────────────────
Robustness pass script for CalorieVision.

Runs all 15 videos defined in shared/test_videos_manifest.json through the
central pipeline orchestrator (fusion_pipeline/pipeline.py):
  download -> keypoints -> motion_filter -> scene_detect -> classify -> ocr -> fusion -> calorie

Disagreements and low-confidence predictions are automatically written to
eval/failure_log.jsonl.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fusion_pipeline.pipeline import run_pipeline

logger = logging.getLogger("eval.manifest_pass")


def run_robustness_pass(
    manifest_path: Path | None = None,
    log_path: Path | None = None,
) -> dict:
    """
    Run pipeline robustness pass on all manifest entries.
    """
    if manifest_path is None:
        manifest_path = _REPO_ROOT / "shared" / "test_videos_manifest.json"
    if log_path is None:
        log_path = _REPO_ROOT / "eval" / "failure_log.jsonl"

    log_path.parent.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        return {"total_videos": 0, "processed_videos": 0, "disagreements": 0}

    manifest_entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    logger.info("Loaded %d manifest entries from %s", len(manifest_entries), manifest_path)

    processed_count = 0
    total_fused_segments = 0
    classifiers_used = {}

    for entry in manifest_entries:
        vid_id = entry["youtube_id"]
        logger.info("Processing manifest video %s...", vid_id)

        try:
            res = run_pipeline(
                video_source=vid_id,
                weight_kg=70.0,
                user_tier="intermediate",
                force_recompute=False,
            )
            processed_count += 1
            total_fused_segments += len(res.fused_segments)
            cls_name = res.classifier_used
            classifiers_used[cls_name] = classifiers_used.get(cls_name, 0) + 1

        except Exception as exc:
            logger.warning("Pipeline run failed for %s: %s", vid_id, exc)

    # Read failure log stats
    disagreements, low_confs = 0, 0
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            if not line.strip() or not line.strip().startswith("{"):
                continue
            try:
                ev = json.loads(line)
                if ev.get("event_type") == "DISAGREE":
                    disagreements += 1
                elif ev.get("event_type") == "LOW_CONF":
                    low_confs += 1
            except Exception:
                pass

    summary = {
        "total_manifest_entries": len(manifest_entries),
        "processed_videos": processed_count,
        "total_fused_segments": total_fused_segments,
        "classifiers_used": classifiers_used,
        "logged_disagreements": disagreements,
        "logged_low_confidence": low_confs,
        "failure_log_path": str(log_path),
    }

    logger.info("Robustness pass complete: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = run_robustness_pass()
    print(json.dumps(res, indent=2))
