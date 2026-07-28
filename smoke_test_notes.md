# CalorieVision — Early Smoke Test Notes

> **Section 7 — Early Pipeline Validation**
> Do not leave smoke testing to Week 5.

**Generated:** 2026-07-21T07:26:05Z  
**Max frames per video:** 150  
**Stages tested:** Download → Keypoint Extraction → Scene Detection  
**Run environment:** Windows, Python 3.13, yt-dlp (no ffmpeg), no Deno JS runtime  

---

## Summary

| Video | Tags | Download | Keypoints | Detection Rate | Scenes | Notes |
|-------|------|----------|-----------|----------------|--------|-------|
| jNQXAC9IVRw | single-person, outdoor, short | ❌ | — | — | — | ffmpeg missing — yt-dlp cannot merge formats |
| K-CrEL0DxMQ | single-person, indoor, exercise | ❌ | — | — | — | Video unavailable / region-blocked |
| B-MkMGGpHig | multi-person, gym | ❌ | — | — | — | Video unavailable / region-blocked |

> ⚠️ **None of the pipeline stages past download could run** because all three download attempts failed. The root causes are documented below.

---

## Per-Video Detail

### `jNQXAC9IVRw` — Me at the zoo (CC-BY, ~19 s)

**Tags:** single-person, outdoor, short, stable-camera  

#### Download
- **Status:** ❌ (rc=1)
- **Root cause:** `ffmpeg` is not installed on this machine. yt-dlp downloads video and audio as separate streams and needs `ffmpeg` to merge them into a single `.mp4`.
- **Raw error:**
  ```
  ERROR: You have requested merging of multiple formats but ffmpeg is not installed.
  ```
- **Secondary issue:** yt-dlp warns that YouTube extraction without a JS runtime (e.g. Deno) is deprecated. Some formats may be missing even after ffmpeg is added.

#### Fix
1. Install ffmpeg: `winget install ffmpeg` or download from https://ffmpeg.org/
2. Install Deno for the JS runtime warning: `winget install Deno.Deno`

---

### `K-CrEL0DxMQ` — Squat form tutorial

**Tags:** single-person, indoor, exercise, tutorial  

#### Download
- **Status:** ❌ (rc=1)
- **Root cause:** Video is unavailable (deleted, private, or region-restricted).
- **Raw error:**
  ```
  ERROR: [youtube] K-CrEL0DxMQ: Video unavailable
  ```
- **Action required:** Replace this video ID in the smoke test catalogue with a verified public exercise tutorial.

---

### `B-MkMGGpHig` — Group workout / aerobics

**Tags:** multi-person, gym, dynamic-camera  

#### Download
- **Status:** ❌ (rc=1)
- **Root cause:** Video is unavailable (same issue as above).
- **Raw error:**
  ```
  ERROR: [youtube] B-MkMGGpHig: Video unavailable
  ```
- **Action required:** Replace with a verified public multi-person workout video.

---

## Open Issues

| Priority | Issue | Action |
|----------|-------|--------|
| 🔴 BLOCKER | `ffmpeg` not installed — yt-dlp cannot merge video/audio streams | `winget install ffmpeg` |
| 🔴 BLOCKER | No JS runtime (Deno/Node) — yt-dlp YouTube extraction partially degraded | `winget install Deno.Deno` |
| 🟡 MEDIUM | 2 of 3 test video IDs are unavailable/region-blocked | Replace K-CrEL0DxMQ and B-MkMGGpHig in `smoke_test.py` catalogue |
| 🟡 MEDIUM | Keypoint/scene stages completely untested on real footage | Blocked by ffmpeg fix above |
| 🟢 LOW | MediaPipe multi-person limitation: `num_poses=1` in `extract_keypoints.py` | Multi-person testing deferred; increase `num_poses` when running group videos |

---

## Pipeline Stages Not Yet Reachable

Because all downloads failed, the following stages have NOT been validated on real footage:

- **Keypoint extraction** — known to work on synthetic video (see `test_keypoints.py`); real-world pose detection rate unknown
- **Scene detection** — known to work on synthetic video (see `test_scene_detect.py`); effectiveness on real workout cuts unknown
- **OCR detection** — untested end-to-end; model weights not yet downloaded

---

## Next Steps (Ordered by Priority)

- [ ] **Install ffmpeg** (unblocks everything else)
- [ ] **Replace unavailable video IDs** in `smoke_test.py` with verified public videos
- [ ] Re-run `python smoke_test.py --skip-download` after fixing the above
- [ ] Capture actual pose detection rates for single-person vs. multi-person videos
- [ ] Measure scene cut counts on real exercise footage and tune `threshold` if needed
- [ ] Run OCR stage on a video with on-screen rep counters and log detection quality
- [ ] Document specific timestamps where the pipeline fails for Section 7 regression tracking

---

## Environment Details

```
OS        : Windows 11
Python    : 3.13.0
yt-dlp    : installed (no ffmpeg, no Deno JS runtime)
mediapipe : installed
scenedetect : 0.7 (installed during this sprint)
scikit-learn : 1.9.0 (installed during this sprint)
easyocr   : installed (weights not yet downloaded)
ffmpeg    : NOT INSTALLED ← primary blocker
```
