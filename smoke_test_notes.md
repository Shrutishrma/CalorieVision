# CalorieVision — Early Smoke Test Notes

> **Section 7 — Early Pipeline Validation**
> Do not leave smoke testing to Week 5.

**Generated:** 2026-07-28T07:54:37.159670+00:00  
**Max frames per video:** 60  
**Stages tested:** Download → Keypoint Extraction → Scene Detection  

---

## Summary

| Video | Tags | Download | Keypoints | Detection Rate | Scenes | Notes |
|-------|------|----------|-----------|----------------|--------|-------|
| jNQXAC9IVRw | camera:frontal | ✅ | ❌ | — | — ❌ | ❌ Keypoint extraction failed (rc=1); ❌ Scene detection failed (rc=1) |
| IODxDxX7oi4 | camera:frontal | ✅ | ❌ | — | — ❌ | ❌ Keypoint extraction failed (rc=1); ❌ Scene detection failed (rc=1) |
| aclHkVaku9U | camera:angled | ✅ | ❌ | — | — ❌ | ❌ Keypoint extraction failed (rc=1); ❌ Scene detection failed (rc=None) |

---

## Per-Video Detail

### `jNQXAC9IVRw` — Me at the zoo — single person, outdoor, stable, ~19 s (CC-BY)

**Tags:** camera:frontal  
**Local file:** `C:\Users\shrut\OneDrive\Desktop\CV PROJECT\CalorieVision\shared\test-videos\jNQXAC9IVRw.mp4`  

#### Download
- **Status:** ✅ (rc=0)  
- File size: 0.51 MB  

#### Keypoint Extraction
- **Status:** ❌ (rc=1)  
- stdout: ```
[CalorieVision] Extracting keypoints from: C:\Users\shrut\OneDrive\Desktop\CV PROJECT\CalorieVision\shared\test-videos\jNQXAC9IVRw.mp4
[CalorieVision] Model complexity: 0
[CalorieVision] Processed 60 frames
[CalorieVision] Pose detected in 60/60 frames (100.0%)
```
- stderr: ```
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\shrut\AppData\Local\Programs\Python\Python313\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2192' in position 29: character maps to <undefined>
```

#### Scene Detection
- **Status:** ❌ (rc=1)  
- stderr: ```
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\shrut\AppData\Local\Programs\Python\Python313\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2192' in position 34: character maps to <undefined>
```

#### Findings / Anomalies
- ❌ Keypoint extraction failed (rc=1)  
- ❌ Scene detection failed (rc=1)  

---

### `IODxDxX7oi4` — The Perfect Push Up — single person, indoor, exercise-focused (verified public)

**Tags:** camera:frontal  
**Local file:** `C:\Users\shrut\OneDrive\Desktop\CV PROJECT\CalorieVision\shared\test-videos\IODxDxX7oi4.mp4`  

#### Download
- **Status:** ✅ (rc=0)  
- File size: 13.88 MB  

#### Keypoint Extraction
- **Status:** ❌ (rc=1)  
- stdout: ```
[CalorieVision] Extracting keypoints from: C:\Users\shrut\OneDrive\Desktop\CV PROJECT\CalorieVision\shared\test-videos\IODxDxX7oi4.mp4
[CalorieVision] Model complexity: 0
[CalorieVision] Processed 60 frames
[CalorieVision] Pose detected in 50/60 frames (83.3%)
```
- stderr: ```
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\shrut\AppData\Local\Programs\Python\Python313\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2192' in position 29: character maps to <undefined>
```

#### Scene Detection
- **Status:** ❌ (rc=1)  
- stderr: ```
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\shrut\AppData\Local\Programs\Python\Python313\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2192' in position 35: character maps to <undefined>
```

#### Findings / Anomalies
- ❌ Keypoint extraction failed (rc=1)  
- ❌ Scene detection failed (rc=1)  

---

### `aclHkVaku9U` — Squats for Beginners — exercise instruction, indoor studio setting (verified public)

**Tags:** camera:angled  
**Local file:** `C:\Users\shrut\OneDrive\Desktop\CV PROJECT\CalorieVision\shared\test-videos\aclHkVaku9U.mp4`  

#### Download
- **Status:** ✅ (rc=0)  
- File size: 8.04 MB  

#### Keypoint Extraction
- **Status:** ❌ (rc=1)  
- stdout: ```
[CalorieVision] Extracting keypoints from: C:\Users\shrut\OneDrive\Desktop\CV PROJECT\CalorieVision\shared\test-videos\aclHkVaku9U.mp4
[CalorieVision] Model complexity: 0
[CalorieVision] Processed 60 frames
[CalorieVision] Pose detected in 51/60 frames (85.0%)
```
- stderr: ```
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\shrut\AppData\Local\Programs\Python\Python313\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2192' in position 29: character maps to <undefined>
```

#### Scene Detection
- **Status:** ❌ (rc=None)  
- **Exception:** `TimeoutExpired (>120s)`  

#### Findings / Anomalies
- ❌ Keypoint extraction failed (rc=1)  
- ❌ Scene detection failed (rc=None)  

---

## Open Issues

| Video | Issue |
|-------|-------|
| jNQXAC9IVRw | ❌ Keypoint extraction failed (rc=1) |
| jNQXAC9IVRw | ❌ Scene detection failed (rc=1) |
| IODxDxX7oi4 | ❌ Keypoint extraction failed (rc=1) |
| IODxDxX7oi4 | ❌ Scene detection failed (rc=1) |
| aclHkVaku9U | ❌ Keypoint extraction failed (rc=1) |
| aclHkVaku9U | ❌ Scene detection failed (rc=None) |

---

## Next Steps

- [ ] Review detection rates — low rates (<30%) suggest camera angle or occlusion issues
- [ ] Check scene-cut counts — 0 cuts on exercise videos might mean threshold needs lowering
- [ ] Attempt OCR stage on same videos once `ocr_detector.py` models are downloaded
- [ ] Expand test set to include multi-person and angled-camera videos
- [ ] Document specific timestamps where the pipeline fails for future regression tracking
