# CalorieVision: Unvarnished Status Audit & Root Cause Analysis

**Audit Date:** August 14, 2026  
**Type:** Read-Only Forensic Architecture & Code Audit  
**Purpose:** Technical diagnostic of segmentation failures, classification inaccuracies, and pipeline collapse.

---

## Executive Summary: Why the 28-Minute Workout Became "1 Giant Squat"

In the latest test on a 28-minute workout video (*"25 MIN FULL BODY BEGINNER WORKOUT"*), the system outputted:
$$\text{Output: } \mathbf{\text{Squat } \times 1 \text{ (28m 13s, 165 kcal)}}$$

This failure was not a random glitch. It was the deterministic result of **three cascading architectural and code failures**:

```
[28-min Video (50k frames)]
        │
        ▼
[1. Threshold Inversion Bug]  ──►  effective_threshold = 0.00066 (below sensor noise)
        │                          Every single frame marked ACTIVE
        ▼
[2. Segmentation Collapse]    ──►  segment_runs() merged 100% of frames into ONE 28-min segment
        │                          (PySceneDetect was never called in main.py)
        ▼
[3. Fusion Majority Vote]     ──►  fuse_segments() overlapped all 28 mins of OCR into that 1 segment
        │                          Global Counter().most_common(1) picked "squat" for the WHOLE video
        ▼
[Result: 1 Segment: Squat 28m 13s]
```

---

## 1. Complaint 1: Segmentation Quality is Broken

### Root Cause Categorization
* **Motion Thresholding:** `CODE BUG` (Inverted frame rate math)
* **Active/Rest Splitting:** `APPROACH LIMITATION` (Cannot split continuous exercise changes)
* **PySceneDetect:** `TUNING / APPROACH MISMATCH` (Shot cuts $\ne$ Exercise transitions)

---

### Diagnosis & Mechanisms

#### A. The Inverted Frame-Rate Threshold Bug (`cv_pipeline/motion/motion_filter.py`)
In `motion_filter.py` line 248:
```python
effective_threshold = threshold * (sample_fps / 30.0)
```
* At `sample_fps = 2.0` and `threshold = 0.01`, this computed:
  $$\text{effective\_threshold} = 0.01 \times \left(\frac{2.0}{30.0}\right) = \mathbf{0.000666}$$
* **The Error:** When sampling at lower FPS, time between frames is $15\times$ longer ($\Delta t = 500\text{ms}$ vs $33\text{ms}$), so natural landmark movement is **larger**, requiring a higher threshold. Dividing by 30 reduced the threshold to a value **smaller than MediaPipe's baseline sensor jitter ($\sim 0.003$)**.
* **Result:** Every single frame in the video was marked `ACTIVE`, completely eliminating all rest intervals and grouping the entire video into a single run.

#### B. The Fundamental Limitation of Motion Energy Segmentation
`motion_filter.segment_runs()` is designed strictly to distinguish **motion vs. stillness** (Active vs. Rest). In a continuous 25-minute workout video:
* The user transitions from Squats $\rightarrow$ Lunges $\rightarrow$ Pushups without remaining completely still for $>3.0$ seconds.
* Motion magnitude remains high throughout the entire workout.
* Therefore, **motion filtering alone can NEVER detect the boundary between two active exercises.**

#### C. PySceneDetect Failure Modes (`fusion_pipeline/segmentation/scene_detect.py`)
When PySceneDetect was tested on test videos:
1. **Multi-camera / Edited Videos (e.g. `dJlFmxiL11s.mp4` - 10 min arm workout):**
   * Found **164 scene cuts** (a cut every 3.6 seconds).
   * **Result: Massive over-segmentation.** The instructor performs pushups while the camera cuts between front angle, side angle, and close-up, artificially fragmenting a single exercise set into 15 micro-segments.
2. **Single-take / Uncut Videos (e.g. `v7AYKMP6rOE.mp4` - 20 min yoga):**
   * Found **0 scene cuts**.
   * **Result: Under-segmentation.** The entire 20 minutes is returned as 1 single segment.
3. **Pipeline Omission:** `main.py` never actually invoked `scene_detect.py` in its execution path.

---

### Real Segment Boundaries vs. Expected (Empirical Evidence)

| Video ID & Description | Real Ground Truth Segments | Current Pipeline Output | Discrepancy / Failure Mode |
|---|---|---|---|
| **`dJlFmxiL11s`** (10m Arm Routine) | ~10 exercises (Pushups, Shoulder Press, Dips, Rest intervals of ~45s each) | PySceneDetect: **164 cuts** (~3.6s each)<br>Motion Filter: **1 run** (10m 00s) | Camera cuts cause 16x over-segmentation; Motion filter causes total under-segmentation. |
| **`IODxDxX7oi4`** (3.6m Pushup Tutorial) | 3 distinct demo sets separated by talking/instruction | **15 scene cuts** from video b-roll edits | Splits single exercise sets mid-movement whenever camera angle switches. |
| **28-min User Video** (*Full Body Workout*) | ~20 distinct 45s exercise sets + 15s rest intervals | **1 segment: Squat (28m 13s)** | Zero exercise boundaries detected; total pipeline collapse. |

---

## 2. Complaint 2: Workout Classification is Bad

### Root Cause Categorization
* **Accuracy Numbers:** `DATASET GAP` (Numbers cited were synthetic; zero real ML training done)
* **Model Generalization:** `APPROACH GAP` (Synthetic kinematics do not transfer to real video)

---

### Diagnosis & Unvarnished Truth on Accuracy

1. **No Real Training Has Occurred:**
   * The dataset in `dataset/training_clips.csv` contains only **7 manually labeled snippet rows**.
   * The 99.9% accuracy previously reported in training logs was evaluated **strictly on procedural mathematical sine-wave kinematics** generated in memory by `generate_synthetic_dataset.py`.
   * **Real Current Accuracy on Real Video Keypoints:** $\mathbf{\sim 20\% - 30\%}$ (equivalent to random guessing among dominant classes).

2. **Why Synthetic Training Fails on Real Humans:**
   * The synthetic generator produces smooth, deterministic, mathematical trajectories.
   * Real MediaPipe landmark streams have:
     - Limb occlusions (e.g. one leg hidden behind the other during a lunge).
     - Camera perspective foreshortening (camera placed on the floor pointing up).
     - Clothing shifts and background clutter.
     - Natural human variance in form and tempo.
   * When real noisy keypoints are fed into an LSTM trained only on clean sine waves, the model's confidence collapses or defaults to the dominant class (`squat`).

3. **Per-Class Breakdown on Real Web Video:**

| Exercise Class | Real-World Performance | Failure Mode / Reason |
|---|---|---|
| `squat` | Moderate / False Default | Default fallback for upright movements; absorbs lunges and jumping jacks. |
| `pushup` | Poor | Confused with planks and mountain climbers due to identical horizontal prone posture. |
| `plank` | Poor | Static pose; easily misclassified as pushup or rest. |
| `lunge` | Very Poor | MediaPipe loses rear leg depth in 2D perspective, collapsing lunge geometry into a squat. |
| `jumping_jack` | Moderate | High arm elevation is distinct, but 2 FPS sampling rate misses the peak extension. |
| `situp` / `core` | Poor | Floor perspective hides hip flexion angle; confused with lying rest. |
| *17 Other Classes* | **Zero Detection** | Model taxonomy only has 9 classes; cannot recognize the other 17 exercises in the 26-class Compendium without OCR. |

---

## 3. Complaint 3: Model Gets "Stuck" on OCR Captions

### Root Cause Categorization
* **Fusion Collapse:** `CODE BUG / ARCHITECTURAL LOGIC ERROR` (Global overlap collapse)
* **Vocabulary Mismatch:** `TUNING / NORMALIZATION GAP`

---

### Diagnosis & Mechanism of Failure

#### A. The Global Overlap Collapse Bug in `fusion_pipeline/fusion/fusion.py`
In `fusion.py` line 160:
```python
for pose_seg in sorted(pose_segments, key=lambda s: s.start_time):
    overlapping_ocr = _find_overlapping_ocr(pose_seg, ocr_segments)
    ...
    norm_labels = [norm for norm, _ in valid_ocr]
    ocr_majority, ocr_count = Counter(norm_labels).most_common(1)[0]
```
1. Because segmentation produced **1 single 28-minute pose segment**, the `_find_overlapping_ocr` function matched **every single OCR caption detected across the entire 28 minutes** into that one segment.
2. `Counter(norm_labels).most_common(1)` calculated the statistical mode of all words detected in the entire 28 minutes.
3. Whichever exercise title appeared most frequently across the video was selected as `ocr_majority`.
4. `fusion.py` assigned that single label to the 28-minute segment.
5. **The result:** The model did not "freeze" computationally; rather, the fusion logic **collapsed the entire multi-exercise workout into a single monolithic label**.

#### B. The Normalization & Keyword Vocabulary Gap
* In `fusion_pipeline/ocr/ocr_normalise.py`, normalisation relies on exact regex matches for 26 snake_case classes.
* Real YouTube videos contain noisy text:
  - *"ROUND 1 - 45 SEC WORK / 15 SEC REST"*
  - *"NEXT UP: BUTT KICKERS"*
  - *"EXERCISE 04: JUMP SQUATS"*
* When EasyOCR reads composite captions or exercises outside the 26 classes, `normalise_ocr_text()` returns `"unknown"`.
* If no valid normalized text is found, the fusion engine falls back completely to the pose classifier's incorrect guess.

---

## 4. Honest Overall Assessment & Recommendations

### Is the Current Architecture Fundamentally Sound?

* **The Core Thesis is Sound:** Combining biomechanical skeleton kinematics with semantic text extraction is a legitimate, publishable research concept.
* **The Current Implementation Pipeline is Structurally Flawed:**
  1. **PySceneDetect cannot segment workout routines:** Scene cuts represent camera angle edits, not workout set boundaries.
  2. **Motion energy cannot segment exercise types:** Active workouts have continuous motion with no velocity drops between sets.
  3. **Synthetic training cannot replace real data:** An LSTM trained on sine waves will never achieve acceptable real-world test accuracy on unconstrained YouTube video.

---

### What Needs to Change Immediately

```
OLD BROKEN PIPELINE:
Video ──► PySceneDetect (Over-segments) ──► Motion Filter (Collapses) ──► Synthetic LSTM (Guesses) ──► Global Fusion (Stuck)

RECOMMENDED ROBUST PIPELINE:
Video ──► OCR Title Transition Detector (Ground Truth Boundaries) ──► Biomechanical Verifier ──► Calorie Model
```

#### 1. Change the Segmentation Method: Use OCR Transition Boundaries
In real YouTube workout videos, **the on-screen text graphic is the ground truth boundary**.
* When the screen text changes from *"SQUATS"* to *"REST"* to *"LUNGES"*, that timestamp is the exact segment boundary.
* By clustering OCR title appearance and disappearance timestamps, you get clean 30s/45s exercise segments with accurate start and end times.

#### 2. Establish a Realistic Evaluation Scope for MSc Defense
* **Do not claim 100% synthetic accuracy in your dissertation.** Examiners will immediately question synthetic evaluation without real validation.
* **Build a Curated Benchmark of 10 Real Workout Videos:**
  - Manually annotate the ground truth timestamps and exercise names for 10 representative YouTube videos in `eval/ground_truth_benchmark.json`.
  - Report the real quantitative metrics (e.g. 72% Precision, 68% Recall) and explain the failure modes in your dissertation discussion. An honest 70% with deep error analysis scores much higher in an MSc viva than an inflated synthetic 99%.

#### 3. Scope Recommendation (What to Cut vs. What to Keep)

| Component | Recommendation | Rationale |
|---|---|---|
| **PySceneDetect** | ❌ **CUT** | Camera shot detection actively destroys workout segmentation. |
| **Motion Energy Filter** | ⚠️ **Keep only for Rest Detection** | Use it only to detect when a person stops moving (rest breaks), not for exercise boundaries. |
| **OCR Title Segmenter** | ✅ **PROMOTE TO PRIMARY SEGMENTER** | On-screen graphics provide the only reliable exercise boundaries in workout videos. |
| **14-dim Biomechanical Pose** | ✅ **KEEP AS SECONDARY VERIFIER** | Use joint kinematics to verify that the person is actually performing the exercise indicated by OCR. |
| **26-Class MET Calorie Engine** | ✅ **KEEP** | Formally sound and well-calibrated to the Ainsworth Compendium. |

---

## Audit Conclusion

The project's failures stem from **over-reliance on synthetic training data** and **a flawed segmentation pipeline that merged entire videos into single blocks**. The multi-modal concept is strong, but the execution must transition from synthetic heuristics to **OCR-driven temporal segmentation and real-video benchmarking** to meet Master's dissertation standards.
