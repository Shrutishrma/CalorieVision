# CalorieVision — Early Smoke Test Notes

> **Section 7 — Early Pipeline Validation**
> Do not leave smoke testing to Week 5.

**Generated:** 2026-08-09T14:23:05.503106+00:00  
**Max frames per video:** 150  
**Stages tested:** Download → Keypoint Extraction → Scene Detection  

---

## Summary

| Video | Tags | Download | Keypoints | Detection Rate | Scenes | Notes |
|-------|------|----------|-----------|----------------|--------|-------|
| IODxDxX7oi4 | camera:single | ✅ | ❌ | — | 15 ✅ | ❌ Keypoint extraction failed (rc=1) |
| aclHkVaku9U | camera:single | ✅ | ❌ | — | 8 ✅ | ❌ Keypoint extraction failed (rc=1) |
| v7AYKMP6rOE | camera:single | ✅ | ❌ | — | — ❌ | ❌ Keypoint extraction failed (rc=1); ❌ Scene detection failed (rc=None) |
| UBMkG03HOHU | camera:multi, caption | ❌ | ❌ | — | — ❌ | ❌ Download failed (rc=1) |
| gC_L9qAHVJ8 | camera:single | ✅ | ❌ | — | — ❌ | ❌ Keypoint extraction failed (rc=1); ❌ Scene detection failed (rc=None) |
| ml6cT4AZdqI | camera:single, caption | ✅ | ❌ | — | — ❌ | ❌ Keypoint extraction failed (rc=1); ❌ Scene detection failed (rc=None) |
| cbKkB3POqaY | camera:multi, caption | ✅ | ❌ | — | — ❌ | ❌ Keypoint extraction failed (rc=1); ❌ Scene detection failed (rc=None) |
| 2pLT-ilgU7w | camera:single, caption | ❌ | ❌ | — | — ❌ | ❌ Download failed (rc=1) |
| bO_xN_0aB-E | camera:single | ❌ | ❌ | — | — ❌ | ❌ Download failed (rc=1) |
| dJlFmxiL11s | camera:single, caption | ✅ | ❌ | — | 164 ✅ | ❌ Keypoint extraction failed (rc=1) |
| L_xrDAtykMI | camera:single, caption | ✅ | ❌ | — | 73 ✅ | ❌ Keypoint extraction failed (rc=1) |
| wI__eN_Jm0s | camera:multi, caption | ❌ | ❌ | — | — ❌ | ❌ Download failed (rc=1) |

---

## Per-Video Detail

### `IODxDxX7oi4` — The Perfect Push Up · Push-ups

**Tags:** camera:single  
**Local file:** `C:\SattyGithub\CalorieVision\shared\test-videos\IODxDxX7oi4.mp4`  

#### Download
- **Status:** ✅ (rc=0)  
- File size: 13.88 MB  

#### Keypoint Extraction
- **Status:** ❌ (rc=1)  
- stdout: ```
[CalorieVision] Extracting keypoints from: C:\SattyGithub\CalorieVision\shared\test-videos\IODxDxX7oi4.mp4
[CalorieVision] Model complexity: 0
[CalorieVision] Processed 150 frames
[CalorieVision] Pose detected in 140/150 frames (93.3%)
```
- stderr: ```
  File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2192' in position 29: character maps to <undefined>
```

#### Scene Detection
- **Status:** ✅ (rc=0)  
- Scenes detected: 15  

#### Findings / Anomalies
- ❌ Keypoint extraction failed (rc=1)  

---

### `aclHkVaku9U` — Squats for Beginners · Squats

**Tags:** camera:single  
**Local file:** `C:\SattyGithub\CalorieVision\shared\test-videos\aclHkVaku9U.mp4`  

#### Download
- **Status:** ✅ (rc=0)  
- File size: 3.64 MB  
- stderr: ```
WARNING: [youtube] No supported JavaScript runtime could be found. Only deno is enabled by default; to use another runtime add  --js-runtimes RUNTIME[:PATH]  to your command/config. YouTube extraction without a JS runtime has been deprecated, and some formats may be missing. See  https://github.com/yt-dlp/yt-dlp/wiki/EJS  for details on installing one
```

#### Keypoint Extraction
- **Status:** ❌ (rc=1)  
- stdout: ```
[CalorieVision] Extracting keypoints from: C:\SattyGithub\CalorieVision\shared\test-videos\aclHkVaku9U.mp4
[CalorieVision] Model complexity: 0
[CalorieVision] Processed 150 frames
[CalorieVision] Pose detected in 141/150 frames (94.0%)
```
- stderr: ```
  File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2192' in position 29: character maps to <undefined>
```

#### Scene Detection
- **Status:** ✅ (rc=0)  
- Scenes detected: 8  

#### Findings / Anomalies
- ❌ Keypoint extraction failed (rc=1)  

---

### `v7AYKMP6rOE` — 20 Minute Yoga For Beginners · Yoga

**Tags:** camera:single  
**Local file:** `C:\SattyGithub\CalorieVision\shared\test-videos\v7AYKMP6rOE.mp4`  

#### Download
- **Status:** ✅ (rc=0)  
- File size: 35.46 MB  
- stderr: ```
WARNING: [youtube] No supported JavaScript runtime could be found. Only deno is enabled by default; to use another runtime add  --js-runtimes RUNTIME[:PATH]  to your command/config. YouTube extraction without a JS runtime has been deprecated, and some formats may be missing. See  https://github.com/yt-dlp/yt-dlp/wiki/EJS  for details on installing one
```

#### Keypoint Extraction
- **Status:** ❌ (rc=1)  
- stdout: ```
[CalorieVision] Extracting keypoints from: C:\SattyGithub\CalorieVision\shared\test-videos\v7AYKMP6rOE.mp4
[CalorieVision] Model complexity: 0
[CalorieVision] Processed 150 frames
[CalorieVision] Pose detected in 149/150 frames (99.3%)
```
- stderr: ```
  File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2192' in position 29: character maps to <undefined>
```

#### Scene Detection
- **Status:** ❌ (rc=None)  
- **Exception:** `TimeoutExpired (>120s)`  

#### Findings / Anomalies
- ❌ Keypoint extraction failed (rc=1)  
- ❌ Scene detection failed (rc=None)  

---

### `UBMkG03HOHU` — 20 Min Full Body HIIT · HIIT

**Tags:** camera:multi, caption  
**Local file:** `C:\SattyGithub\CalorieVision\shared\test-videos\UBMkG03HOHU.mp4`  

#### Download
- **Status:** ❌ (rc=1)  
- stderr: ```
RNING: [youtube] No supported JavaScript runtime could be found. Only deno is enabled by default; to use another runtime add  --js-runtimes RUNTIME[:PATH]  to your command/config. YouTube extraction without a JS runtime has been deprecated, and some formats may be missing. See  https://github.com/yt-dlp/yt-dlp/wiki/EJS  for details on installing one
ERROR: [youtube] UBMkG03HOHU: Video unavailable

[ERROR] Video at 'https://youtu.be/UBMkG03HOHU' is private, deleted, or unavailable in your region.
```

#### Keypoint Extraction
- **Status:** ❌ (rc=None)  

#### Scene Detection
- **Status:** ❌ (rc=None)  

#### Findings / Anomalies
- ❌ Download failed (rc=1)  

---

### `gC_L9qAHVJ8` — 5 Min Plank Challenge · Plank

**Tags:** camera:single  
**Local file:** `C:\SattyGithub\CalorieVision\shared\test-videos\gC_L9qAHVJ8.mp4`  

#### Download
- **Status:** ✅ (rc=0)  
- File size: 84.05 MB  
- stderr: ```
WARNING: [youtube] No supported JavaScript runtime could be found. Only deno is enabled by default; to use another runtime add  --js-runtimes RUNTIME[:PATH]  to your command/config. YouTube extraction without a JS runtime has been deprecated, and some formats may be missing. See  https://github.com/yt-dlp/yt-dlp/wiki/EJS  for details on installing one
```

#### Keypoint Extraction
- **Status:** ❌ (rc=1)  
- stdout: ```
[CalorieVision] Extracting keypoints from: C:\SattyGithub\CalorieVision\shared\test-videos\gC_L9qAHVJ8.mp4
[CalorieVision] Model complexity: 0
[CalorieVision] Processed 150 frames
[CalorieVision] Pose detected in 150/150 frames (100.0%)
```
- stderr: ```
  File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2192' in position 29: character maps to <undefined>
```

#### Scene Detection
- **Status:** ❌ (rc=None)  
- **Exception:** `TimeoutExpired (>120s)`  

#### Findings / Anomalies
- ❌ Keypoint extraction failed (rc=1)  
- ❌ Scene detection failed (rc=None)  

---

### `ml6cT4AZdqI` — 10 Min Ab Workout · Core & Sit-ups

**Tags:** camera:single, caption  
**Local file:** `C:\SattyGithub\CalorieVision\shared\test-videos\ml6cT4AZdqI.mp4`  

#### Download
- **Status:** ✅ (rc=0)  
- File size: 77.72 MB  
- stderr: ```
WARNING: [youtube] No supported JavaScript runtime could be found. Only deno is enabled by default; to use another runtime add  --js-runtimes RUNTIME[:PATH]  to your command/config. YouTube extraction without a JS runtime has been deprecated, and some formats may be missing. See  https://github.com/yt-dlp/yt-dlp/wiki/EJS  for details on installing one
```

#### Keypoint Extraction
- **Status:** ❌ (rc=1)  
- stdout: ```
[CalorieVision] Extracting keypoints from: C:\SattyGithub\CalorieVision\shared\test-videos\ml6cT4AZdqI.mp4
[CalorieVision] Model complexity: 0
[CalorieVision] Processed 150 frames
[CalorieVision] Pose detected in 148/150 frames (98.7%)
```
- stderr: ```
  File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2192' in position 29: character maps to <undefined>
```

#### Scene Detection
- **Status:** ❌ (rc=None)  
- **Exception:** `TimeoutExpired (>120s)`  

#### Findings / Anomalies
- ❌ Keypoint extraction failed (rc=1)  
- ❌ Scene detection failed (rc=None)  

---

### `cbKkB3POqaY` — 15 Min HIIT Cardio · High Knees & Burpees

**Tags:** camera:multi, caption  
**Local file:** `C:\SattyGithub\CalorieVision\shared\test-videos\cbKkB3POqaY.mp4`  

#### Download
- **Status:** ✅ (rc=0)  
- File size: 60.64 MB  

#### Keypoint Extraction
- **Status:** ❌ (rc=1)  
- stdout: ```
[CalorieVision] Extracting keypoints from: C:\SattyGithub\CalorieVision\shared\test-videos\cbKkB3POqaY.mp4
[CalorieVision] Model complexity: 0
[CalorieVision] Processed 150 frames
[CalorieVision] Pose detected in 150/150 frames (100.0%)
```
- stderr: ```
  File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2192' in position 29: character maps to <undefined>
```

#### Scene Detection
- **Status:** ❌ (rc=None)  
- **Exception:** `TimeoutExpired (>120s)`  

#### Findings / Anomalies
- ❌ Keypoint extraction failed (rc=1)  
- ❌ Scene detection failed (rc=None)  

---

### `2pLT-ilgU7w` — 10 Min Leg Workout · Squats & Lunges

**Tags:** camera:single, caption  
**Local file:** `C:\SattyGithub\CalorieVision\shared\test-videos\2pLT-ilgU7w.mp4`  

#### Download
- **Status:** ❌ (rc=1)  
- stderr: ```
RNING: [youtube] No supported JavaScript runtime could be found. Only deno is enabled by default; to use another runtime add  --js-runtimes RUNTIME[:PATH]  to your command/config. YouTube extraction without a JS runtime has been deprecated, and some formats may be missing. See  https://github.com/yt-dlp/yt-dlp/wiki/EJS  for details on installing one
ERROR: [youtube] 2pLT-ilgU7w: Video unavailable

[ERROR] Video at 'https://youtu.be/2pLT-ilgU7w' is private, deleted, or unavailable in your region.
```

#### Keypoint Extraction
- **Status:** ❌ (rc=None)  

#### Scene Detection
- **Status:** ❌ (rc=None)  

#### Findings / Anomalies
- ❌ Download failed (rc=1)  

---

### `bO_xN_0aB-E` — 5 Min Jump Rope · Cardio

**Tags:** camera:single  
**Local file:** `C:\SattyGithub\CalorieVision\shared\test-videos\bO_xN_0aB-E.mp4`  

#### Download
- **Status:** ❌ (rc=1)  
- stderr: ```
RNING: [youtube] No supported JavaScript runtime could be found. Only deno is enabled by default; to use another runtime add  --js-runtimes RUNTIME[:PATH]  to your command/config. YouTube extraction without a JS runtime has been deprecated, and some formats may be missing. See  https://github.com/yt-dlp/yt-dlp/wiki/EJS  for details on installing one
ERROR: [youtube] bO_xN_0aB-E: Video unavailable

[ERROR] Video at 'https://youtu.be/bO_xN_0aB-E' is private, deleted, or unavailable in your region.
```

#### Keypoint Extraction
- **Status:** ❌ (rc=None)  

#### Scene Detection
- **Status:** ❌ (rc=None)  

#### Findings / Anomalies
- ❌ Download failed (rc=1)  

---

### `dJlFmxiL11s` — 10 Min Arm Workout · Shoulder Press & Push-ups

**Tags:** camera:single, caption  
**Local file:** `C:\SattyGithub\CalorieVision\shared\test-videos\dJlFmxiL11s.mp4`  

#### Download
- **Status:** ✅ (rc=0)  
- File size: 24.94 MB  
- stderr: ```
WARNING: [youtube] No supported JavaScript runtime could be found. Only deno is enabled by default; to use another runtime add  --js-runtimes RUNTIME[:PATH]  to your command/config. YouTube extraction without a JS runtime has been deprecated, and some formats may be missing. See  https://github.com/yt-dlp/yt-dlp/wiki/EJS  for details on installing one
```

#### Keypoint Extraction
- **Status:** ❌ (rc=1)  
- stdout: ```
[CalorieVision] Extracting keypoints from: C:\SattyGithub\CalorieVision\shared\test-videos\dJlFmxiL11s.mp4
[CalorieVision] Model complexity: 0
[CalorieVision] Processed 150 frames
[CalorieVision] Pose detected in 97/150 frames (64.7%)
```
- stderr: ```
  File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2192' in position 29: character maps to <undefined>
```

#### Scene Detection
- **Status:** ✅ (rc=0)  
- Scenes detected: 164  

#### Findings / Anomalies
- ❌ Keypoint extraction failed (rc=1)  

---

### `L_xrDAtykMI` — 10 Min High Knees Cardio · Cardio

**Tags:** camera:single, caption  
**Local file:** `C:\SattyGithub\CalorieVision\shared\test-videos\L_xrDAtykMI.mp4`  

#### Download
- **Status:** ✅ (rc=0)  
- File size: 21.8 MB  
- stderr: ```
WARNING: [youtube] No supported JavaScript runtime could be found. Only deno is enabled by default; to use another runtime add  --js-runtimes RUNTIME[:PATH]  to your command/config. YouTube extraction without a JS runtime has been deprecated, and some formats may be missing. See  https://github.com/yt-dlp/yt-dlp/wiki/EJS  for details on installing one
```

#### Keypoint Extraction
- **Status:** ❌ (rc=1)  
- stdout: ```
[CalorieVision] Extracting keypoints from: C:\SattyGithub\CalorieVision\shared\test-videos\L_xrDAtykMI.mp4
[CalorieVision] Model complexity: 0
[CalorieVision] Processed 150 frames
[CalorieVision] Pose detected in 22/150 frames (14.7%)
```
- stderr: ```
  File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2192' in position 29: character maps to <undefined>
```

#### Scene Detection
- **Status:** ✅ (rc=0)  
- Scenes detected: 73  

#### Findings / Anomalies
- ❌ Keypoint extraction failed (rc=1)  

---

### `wI__eN_Jm0s` — 10 Min Squat Challenge · Squats

**Tags:** camera:multi, caption  
**Local file:** `C:\SattyGithub\CalorieVision\shared\test-videos\wI__eN_Jm0s.mp4`  

#### Download
- **Status:** ❌ (rc=1)  
- stderr: ```
RNING: [youtube] No supported JavaScript runtime could be found. Only deno is enabled by default; to use another runtime add  --js-runtimes RUNTIME[:PATH]  to your command/config. YouTube extraction without a JS runtime has been deprecated, and some formats may be missing. See  https://github.com/yt-dlp/yt-dlp/wiki/EJS  for details on installing one
ERROR: [youtube] wI__eN_Jm0s: Video unavailable

[ERROR] Video at 'https://youtu.be/wI__eN_Jm0s' is private, deleted, or unavailable in your region.
```

#### Keypoint Extraction
- **Status:** ❌ (rc=None)  

#### Scene Detection
- **Status:** ❌ (rc=None)  

#### Findings / Anomalies
- ❌ Download failed (rc=1)  

---

## Open Issues

| Video | Issue |
|-------|-------|
| IODxDxX7oi4 | ❌ Keypoint extraction failed (rc=1) |
| aclHkVaku9U | ❌ Keypoint extraction failed (rc=1) |
| v7AYKMP6rOE | ❌ Keypoint extraction failed (rc=1) |
| v7AYKMP6rOE | ❌ Scene detection failed (rc=None) |
| UBMkG03HOHU | ❌ Download failed (rc=1) |
| gC_L9qAHVJ8 | ❌ Keypoint extraction failed (rc=1) |
| gC_L9qAHVJ8 | ❌ Scene detection failed (rc=None) |
| ml6cT4AZdqI | ❌ Keypoint extraction failed (rc=1) |
| ml6cT4AZdqI | ❌ Scene detection failed (rc=None) |
| cbKkB3POqaY | ❌ Keypoint extraction failed (rc=1) |
| cbKkB3POqaY | ❌ Scene detection failed (rc=None) |
| 2pLT-ilgU7w | ❌ Download failed (rc=1) |
| bO_xN_0aB-E | ❌ Download failed (rc=1) |
| dJlFmxiL11s | ❌ Keypoint extraction failed (rc=1) |
| L_xrDAtykMI | ❌ Keypoint extraction failed (rc=1) |
| wI__eN_Jm0s | ❌ Download failed (rc=1) |

---

## Next Steps

- [ ] Review detection rates — low rates (<30%) suggest camera angle or occlusion issues
- [ ] Check scene-cut counts — 0 cuts on exercise videos might mean threshold needs lowering
- [ ] Attempt OCR stage on same videos once `ocr_detector.py` models are downloaded
- [ ] Expand test set to include multi-person and angled-camera videos
- [ ] Document specific timestamps where the pipeline fails for future regression tracking
