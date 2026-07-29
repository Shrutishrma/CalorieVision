"""
smoke_test.py
────────────────────────────────────────────────────────────────────────────────
Early smoke test for CalorieVision pipeline (Section 7 of evaluation plan).

What it does
------------
1. Downloads 2-3 diverse workout videos from YouTube into shared/test-videos/.
2. Runs extract_keypoints.py on each (with --max-frames 150 for speed).
3. Runs scene_detect.py on each.
4. Captures stdout / stderr and any exceptions from every step.
5. Writes a structured Markdown report to smoke_test_notes.md.

This is NOT a pytest test — it is a standalone script run manually when you
want to validate the end-to-end pipeline on real footage.

Usage
-----
    python smoke_test.py                          # full run
    python smoke_test.py --max-frames 60          # faster (2 s of footage)
    python smoke_test.py --skip-download          # reuse already-downloaded files

Test videos
-----------
  1. "Me at the zoo" — jNQXAC9IVRw  (single-person, outdoor, stable camera)
  2. "Squat tutorial" — K-CrEL0DxMQ  (single-person, indoor, exercise-focused)
  3. "Group workout"  — B-MkMGGpHig  (multi-person, typical gym setting)

Findings are written to smoke_test_notes.md in the repo root.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import textwrap
import traceback
from datetime import datetime, timezone
from pathlib import Path

# ── Ensure repo root importable ───────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_venv_python = _REPO_ROOT / ".venv" / "Scripts" / "python.exe"
_PYTHON = str(_venv_python) if _venv_python.exists() else sys.executable
_VIDEOS_DIR = _REPO_ROOT / "shared" / "test-videos"
_KEYPOINTS_DIR = _REPO_ROOT / "shared" / "test-videos"

# ── Test video catalogue (loaded from manifest) ────────────────────────────────

def _load_manifest_videos() -> list[dict]:
    manifest_path = _REPO_ROOT / "shared" / "test_videos_manifest.json"
    if not manifest_path.exists():
        return []
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        res = []
        for item in data:
            tags = item.get("tags", {})
            tag_list = [f"camera:{tags.get('camera_angle', '')}"]
            if tags.get("caption_present"):
                tag_list.append("caption")
            res.append({
                "youtube_id": item["youtube_id"],
                "url": item.get("url", f"https://youtu.be/{item['youtube_id']}"),
                "description": item.get("description", item.get("label", "")),
                "tags": tag_list,
            })
        return res
    except Exception:
        return []

TEST_VIDEOS = _load_manifest_videos()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _run(cmd: list[str], timeout: int = 300) -> dict:
    """
    Run a subprocess and return a dict with stdout, stderr, returncode, error.
    """
    result = {
        "cmd": " ".join(cmd),
        "stdout": "",
        "stderr": "",
        "returncode": None,
        "exception": None,
    }
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(_REPO_ROOT),
        )
        result["stdout"] = proc.stdout.strip()
        result["stderr"] = proc.stderr.strip()
        result["returncode"] = proc.returncode
    except subprocess.TimeoutExpired:
        result["exception"] = f"TimeoutExpired (>{timeout}s)"
    except Exception as exc:
        result["exception"] = traceback.format_exc()
    return result


def _md_code(text: str, lang: str = "") -> str:
    return f"```{lang}\n{text}\n```"


def _status_icon(ok: bool) -> str:
    return "✅" if ok else "❌"


# ─── Smoke test runner ────────────────────────────────────────────────────────

def run_smoke_test(max_frames: int = 150, skip_download: bool = False) -> str:
    """
    Run the smoke test and return the Markdown report as a string.
    """
    _VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    findings: list[dict] = []

    for video_info in TEST_VIDEOS:
        vid_id = video_info["youtube_id"]
        url    = video_info["url"]
        desc   = video_info["description"]
        tags   = video_info["tags"]
        local  = _VIDEOS_DIR / f"{vid_id}.mp4"
        kp_out = _KEYPOINTS_DIR / f"{vid_id}_keypoints.json"

        finding = {
            "youtube_id": vid_id,
            "description": desc,
            "tags": tags,
            "local_path": str(local),
            "download": None,
            "keypoints": None,
            "scene_detect": None,
            "notes": [],
        }

        # ── Step 1: Download ─────────────────────────────────────────────────
        if skip_download and local.exists():
            finding["download"] = {
                "skipped": True,
                "reason": "File already exists and --skip-download set",
                "file_size_mb": round(local.stat().st_size / 1_048_576, 2),
            }
        else:
            print(f"\n[smoke_test] Downloading {vid_id} …")
            dl_result = _run([
                _PYTHON, "fusion_pipeline/segmentation/download_video.py",
                url, "--out", str(_VIDEOS_DIR), "--max-height", "360",
            ], timeout=180)
            dl_ok = dl_result["returncode"] == 0 and local.exists()
            finding["download"] = {
                "ok": dl_ok,
                "returncode": dl_result["returncode"],
                "stdout": dl_result["stdout"][-500:],
                "stderr": dl_result["stderr"][-500:],
                "exception": dl_result["exception"],
                "file_size_mb": round(local.stat().st_size / 1_048_576, 2) if local.exists() else None,
            }
            if not dl_ok:
                finding["notes"].append(f"❌ Download failed (rc={dl_result['returncode']})")
                findings.append(finding)
                continue

        # ── Step 2: Keypoint extraction ──────────────────────────────────────
        print(f"[smoke_test] Extracting keypoints from {vid_id} (max_frames={max_frames}) …")
        kp_result = _run([
            _PYTHON, "cv_pipeline/keypoints/extract_keypoints.py",
            str(local),
            "--out", str(kp_out),
            "--max-frames", str(max_frames),
            "--model-complexity", "0",
        ], timeout=300)

        kp_ok = kp_result["returncode"] == 0 and kp_out.exists()
        kp_stats = {}
        if kp_ok:
            try:
                with open(kp_out) as fh:
                    kp_data = json.load(fh)
                total_f   = kp_data.get("total_frames", 0)
                detected  = kp_data.get("detected_frames", 0)
                rate      = detected / total_f if total_f else 0.0
                kp_stats  = {
                    "total_frames": total_f,
                    "detected_frames": detected,
                    "detection_rate_pct": round(100 * rate, 1),
                }
                if rate < 0.30:
                    finding["notes"].append(
                        f"⚠️  Low pose detection rate: {rate:.0%} "
                        f"({detected}/{total_f} frames)"
                    )
            except Exception as exc:
                finding["notes"].append(f"⚠️  Could not parse keypoints JSON: {exc}")

        finding["keypoints"] = {
            "ok": kp_ok,
            "returncode": kp_result["returncode"],
            "stdout": kp_result["stdout"][-600:],
            "stderr": kp_result["stderr"][-400:],
            "exception": kp_result["exception"],
            **kp_stats,
        }
        if not kp_ok:
            finding["notes"].append(f"❌ Keypoint extraction failed (rc={kp_result['returncode']})")

        # ── Step 3: Scene detection ──────────────────────────────────────────
        print(f"[smoke_test] Running scene detection on {vid_id} …")
        scene_out = _VIDEOS_DIR / f"{vid_id}_scenes.json"
        sd_result = _run([
            _PYTHON, "fusion_pipeline/segmentation/scene_detect.py",
            str(local),
            "--out", str(scene_out),
        ], timeout=120)

        sd_ok = sd_result["returncode"] == 0 and scene_out.exists()
        sd_stats = {}
        if sd_ok:
            try:
                scene_data = json.loads(scene_out.read_text())
                total_scenes = scene_data.get("total_scenes", 0)
                sd_stats = {"total_scenes": total_scenes}
                if total_scenes == 0:
                    finding["notes"].append(
                        "ℹ️  No scene cuts detected (single-scene or scene detection too insensitive)"
                    )
            except Exception as exc:
                finding["notes"].append(f"⚠️  Could not parse scene JSON: {exc}")

        finding["scene_detect"] = {
            "ok": sd_ok,
            "returncode": sd_result["returncode"],
            "stdout": sd_result["stdout"][-400:],
            "stderr": sd_result["stderr"][-400:],
            "exception": sd_result["exception"],
            **sd_stats,
        }
        if not sd_ok:
            finding["notes"].append(f"❌ Scene detection failed (rc={sd_result['returncode']})")

        findings.append(finding)

    # ── Build Markdown report ─────────────────────────────────────────────────
    md = _build_report(findings, started_at=started_at, max_frames=max_frames)
    return md


# ─── Report builder ───────────────────────────────────────────────────────────

def _build_report(findings: list[dict], *, started_at: str, max_frames: int) -> str:
    lines: list[str] = []

    lines.append("# CalorieVision — Early Smoke Test Notes")
    lines.append("")
    lines.append("> **Section 7 — Early Pipeline Validation**")
    lines.append("> Do not leave smoke testing to Week 5.")
    lines.append("")
    lines.append(f"**Generated:** {started_at}  ")
    lines.append(f"**Max frames per video:** {max_frames}  ")
    lines.append(f"**Stages tested:** Download → Keypoint Extraction → Scene Detection  ")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Video | Tags | Download | Keypoints | Detection Rate | Scenes | Notes |")
    lines.append("|-------|------|----------|-----------|----------------|--------|-------|")

    for f in findings:
        vid  = f["youtube_id"]
        tags = ", ".join(f["tags"])
        dl   = f["download"] or {}
        kp   = f["keypoints"] or {}
        sd   = f["scene_detect"] or {}

        dl_icon  = "⏭ skipped" if dl.get("skipped") else _status_icon(dl.get("ok", False))
        kp_icon  = _status_icon(kp.get("ok", False))
        rate_str = f'{kp.get("detection_rate_pct", "—")}%' if kp.get("ok") else "—"
        sd_count = sd.get("total_scenes", "—") if sd.get("ok") else "—"
        sd_icon  = _status_icon(sd.get("ok", False))
        notes    = "; ".join(f["notes"]) if f["notes"] else "—"

        lines.append(
            f"| {vid} | {tags} | {dl_icon} | {kp_icon} | {rate_str} | {sd_count} {sd_icon} | {notes} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-video detail
    lines.append("## Per-Video Detail")
    lines.append("")

    for f in findings:
        vid  = f["youtube_id"]
        desc = f["description"]
        tags = ", ".join(f["tags"])
        dl   = f["download"] or {}
        kp   = f["keypoints"] or {}
        sd   = f["scene_detect"] or {}

        lines.append(f"### `{vid}` — {desc}")
        lines.append("")
        lines.append(f"**Tags:** {tags}  ")
        lines.append(f"**Local file:** `{f['local_path']}`  ")
        lines.append("")

        # Download
        lines.append("#### Download")
        if dl.get("skipped"):
            lines.append(f"- **Skipped** — {dl.get('reason')}  ")
            lines.append(f"- File size: {dl.get('file_size_mb')} MB  ")
        else:
            ok = dl.get("ok", False)
            lines.append(f"- **Status:** {_status_icon(ok)} (rc={dl.get('returncode')})  ")
            if dl.get("file_size_mb"):
                lines.append(f"- File size: {dl.get('file_size_mb')} MB  ")
            if dl.get("exception"):
                lines.append(f"- **Exception:** `{dl['exception']}`  ")
            if dl.get("stderr"):
                lines.append(f"- stderr: {_md_code(dl['stderr'])}")
        lines.append("")

        # Keypoints
        lines.append("#### Keypoint Extraction")
        kp_ok = kp.get("ok", False)
        lines.append(f"- **Status:** {_status_icon(kp_ok)} (rc={kp.get('returncode')})  ")
        if kp_ok:
            lines.append(f"- Frames: {kp.get('total_frames')} total, {kp.get('detected_frames')} with pose ({kp.get('detection_rate_pct')}%)  ")
        if kp.get("exception"):
            lines.append(f"- **Exception:** `{kp['exception']}`  ")
        if kp.get("stdout"):
            lines.append(f"- stdout: {_md_code(kp['stdout'])}")
        if kp.get("stderr") and not kp_ok:
            lines.append(f"- stderr: {_md_code(kp['stderr'])}")
        lines.append("")

        # Scene detection
        lines.append("#### Scene Detection")
        sd_ok = sd.get("ok", False)
        lines.append(f"- **Status:** {_status_icon(sd_ok)} (rc={sd.get('returncode')})  ")
        if sd_ok:
            lines.append(f"- Scenes detected: {sd.get('total_scenes')}  ")
        if sd.get("exception"):
            lines.append(f"- **Exception:** `{sd['exception']}`  ")
        if sd.get("stdout") and not sd_ok:
            lines.append(f"- stdout: {_md_code(sd['stdout'])}")
        if sd.get("stderr") and not sd_ok:
            lines.append(f"- stderr: {_md_code(sd['stderr'])}")
        lines.append("")

        # Notes
        if f["notes"]:
            lines.append("#### Findings / Anomalies")
            for note in f["notes"]:
                lines.append(f"- {note}  ")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Issues log
    lines.append("## Open Issues")
    lines.append("")
    all_notes = [(f["youtube_id"], note) for f in findings for note in f["notes"]]
    if all_notes:
        lines.append("| Video | Issue |")
        lines.append("|-------|-------|")
        for vid, note in all_notes:
            lines.append(f"| {vid} | {note} |")
    else:
        lines.append("_No issues detected during this run._")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Next steps
    lines.append("## Next Steps")
    lines.append("")
    lines.append(
        "- [ ] Review detection rates — low rates (<30%) suggest camera angle or occlusion issues\n"
        "- [ ] Check scene-cut counts — 0 cuts on exercise videos might mean threshold needs lowering\n"
        "- [ ] Attempt OCR stage on same videos once `ocr_detector.py` models are downloaded\n"
        "- [ ] Expand test set to include multi-person and angled-camera videos\n"
        "- [ ] Document specific timestamps where the pipeline fails for future regression tracking"
    )
    lines.append("")

    return "\n".join(lines)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="CalorieVision early smoke test (Section 7 eval).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--max-frames", type=int, default=150,
        help="Max frames to extract per video (lower = faster).",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip downloading if the file already exists.",
    )
    parser.add_argument(
        "--out", default=str(_REPO_ROOT / "smoke_test_notes.md"),
        help="Output Markdown file path.",
    )
    args = parser.parse_args(argv)

    print("[smoke_test] Starting CalorieVision pipeline smoke test ...")
    report = run_smoke_test(max_frames=args.max_frames, skip_download=args.skip_download)

    out_path = Path(args.out)
    out_path.write_text(report, encoding="utf-8")
    print(f"\n[smoke_test] Report written -> {out_path}")


if __name__ == "__main__":
    main()
