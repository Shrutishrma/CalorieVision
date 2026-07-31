"""
eval/eval_tagged.py
────────────────────────────────────────────────────────────────────────────────
Difficulty-Tagged Evaluation Curation & Reporting Script.

Evaluates pipeline output across difficulty tags:
  • Camera Angle (single vs. multi)
  • Caption Presence (present vs. absent)
  • Number of Subjects (single vs. multiple)

Saves summary output to eval/tagged_eval_results.json.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fusion_pipeline.pipeline import run_pipeline

logger = logging.getLogger("eval.eval_tagged")


def run_tagged_eval(
    manifest_path: Path | None = None,
    output_path: Path | None = None,
) -> dict:
    if manifest_path is None:
        manifest_path = _REPO_ROOT / "shared" / "test_videos_manifest.json"
    if output_path is None:
        output_path = _REPO_ROOT / "eval" / "tagged_eval_results.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        return {}

    manifest_entries = json.loads(manifest_path.read_text(encoding="utf-8"))

    tag_groups: Dict[str, Dict[str, List[dict]]] = {
        "camera_angle": {"single": [], "multi": []},
        "caption_present": {"present": [], "absent": []},
        "num_subjects": {"single": [], "multiple": []},
    }

    results_list = []

    for entry in manifest_entries:
        vid_id = entry["youtube_id"]
        tags = entry.get("tags", {})

        try:
            res = run_pipeline(video_source=vid_id, weight_kg=70.0)

            video_res = {
                "video_id": vid_id,
                "label": entry.get("label", ""),
                "tags": tags,
                "duration_secs": res.duration_secs,
                "classifier_used": res.classifier_used,
                "segment_count": len(res.fused_segments),
                "disagreement_count": len(res.failure_events),
                "stage_timings": res.stage_timings,
            }
            results_list.append(video_res)

            # Categorise
            c_angle = tags.get("camera_angle", "single")
            if c_angle in tag_groups["camera_angle"]:
                tag_groups["camera_angle"][c_angle].append(video_res)

            c_present = "present" if tags.get("caption_present") else "absent"
            if c_present in tag_groups["caption_present"]:
                tag_groups["caption_present"][c_present].append(video_res)

            n_subj = tags.get("num_subjects", "single")
            if n_subj in tag_groups["num_subjects"]:
                tag_groups["num_subjects"][n_subj].append(video_res)

        except Exception as exc:
            logger.warning("Evaluation failed for %s: %s", vid_id, exc)

    # Compute aggregate stats per tag
    tag_summaries = {}
    for tag_cat, sub_dict in tag_groups.items():
        tag_summaries[tag_cat] = {}
        for sub_tag, vids in sub_dict.items():
            tot_segs = sum(v["segment_count"] for v in vids)
            tot_disagreements = sum(v["disagreement_count"] for v in vids)
            avg_duration = sum(v["duration_secs"] for v in vids) / max(len(vids), 1)

            tag_summaries[tag_cat][sub_tag] = {
                "video_count": len(vids),
                "total_segments": tot_segs,
                "total_disagreements": tot_disagreements,
                "avg_duration_secs": round(avg_duration, 1),
            }

    report = {
        "total_evaluated_videos": len(results_list),
        "tag_summaries": tag_summaries,
        "per_video_results": results_list,
    }

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Tagged evaluation complete -> %s", output_path)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = run_tagged_eval()
    print(json.dumps(res.get("tag_summaries", {}), indent=2))
