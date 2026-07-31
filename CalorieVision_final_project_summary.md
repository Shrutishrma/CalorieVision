# CalorieVision: AI-Based Exercise Recognition & Calorie Estimation System
## Complete Project Summary & Implementation Plan

---

## 1. Project Overview

**Problem:** People record or watch workout videos but have no automatic way to know what exercises were done, for how long, and how many calories were burned — without wearables or manual logging.

**Solution:** A hybrid computer vision system that takes a workout video (YouTube link or upload) plus the user's weight, and automatically outputs a full timeline: which exercises were performed, for how long, and calorie expenditure at Beginner/Intermediate/Advanced intensity tiers.

**Why this is a strong final-year CV project:**
- It's a complete, working **system**, not a single model in a notebook — video in, structured results out, served through a real web app.
- It combines two independent detection signals (pose-based classification + OCR text validation) that cross-check each other, which gives you a genuine robustness story instead of a single point-of-failure model.
- It's built and evaluated specifically on **unconstrained real-world video** (camera cuts, trainer talk, inconsistent or absent captions, multiple people in frame) rather than clean pre-trimmed benchmark clips — this is the project's core technical differentiator, and it's honest and demonstrable, not an overstated claim.
- Every dataset and reference table used is free, public, and directly fit-for-purpose (not adjacent/repurposed).

**Positioning for report/resume:** This is applied systems engineering solving a real robustness problem, not novel ML research — MediaPipe→LSTM action classification is a well-documented recipe. The differentiator is the *system*: segmentation, dual-signal fusion, and rigorous evaluation on messy real video, which most comparable student projects skip by testing only on clean clips.

**One-line pitch:**
> "Built an end-to-end hybrid CV system combining pose-based action recognition (MediaPipe + PyTorch) with OCR verification to segment, classify, and estimate calorie expenditure across 12 exercise types in unconstrained real-world workout video."

---

## 2. System Architecture

```
[React frontend]
   Upload form (video/YouTube URL + weight)
   Timeline view, calorie breakdown by tier
   Pose-skeleton overlay, per-prediction confidence display
        |  REST (JSON)
        v
[FastAPI backend]
   POST /analyze        -> starts pipeline, returns job_id
   GET  /status/{job_id} -> poll for progress / final result
        |
        v
[CV Pipeline]
   Stage 1: Segmentation       -> PySceneDetect + OCR caption change + motion-pattern shift
   Stage 2: Active/rest filter -> motion-magnitude thresholding
   Stage 3: Classification     -> Pose LSTM (primary) + OCR (validation), fused
   Stage 4: Duration           -> segment boundaries or OCR-read timer
   Stage 5: Calorie calc       -> MET x weight_kg x duration_hours, 3 tiers
   Stage 6 (stretch): Form-quality scoring via joint-angle consistency
        |
        v
[Outputs] -> JSON result + auto-logged disagreements/low-confidence predictions
```

---

## 3. Tech Stack (100% free / open-source)

| Layer | Tool | Link |
|---|---|---|
| Language | Python 3.10+ | — |
| Pose estimation | MediaPipe Pose Landmarker (33 body landmarks) | https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker |
| Action classifier | PyTorch — LSTM/1D-CNN over keypoint sequences | https://pytorch.org/docs/stable/index.html |
| Baseline model | Majority-class + k-NN (scikit-learn) | https://scikit-learn.org/stable/modules/neighbors.html |
| OCR | EasyOCR | https://github.com/JaidedAI/EasyOCR |
| Video processing | OpenCV | https://docs.opencv.org/ |
| Segmentation | PySceneDetect | https://www.scenedetect.com/ |
| Video download | yt-dlp | https://github.com/yt-dlp/yt-dlp |
| Backend API | FastAPI | https://fastapi.tiangolo.com/ |
| Frontend | React (Vite) | https://react.dev/ |
| Compute (training) | Google Colab (free tier) | https://colab.research.google.com/ |
| Version control | GitHub | — |
| Dev environment | Google Antigravity (agentic IDE) | — |

Total cost: **$0**, within Colab free-tier GPU limits for a 12-class LSTM classifier.

---

## 4. Datasets & Reference Data (verified, with links)

| Purpose | Source | Link |
|---|---|---|
| Labeled exercise clips (primary training data) | Kaggle — "Gym Workout/Exercises Video" dataset | https://www.kaggle.com/datasets/philosopher0808/gym-workoutexercises-video |
| Labeled exercise clips (alternative/supplement) | Kaggle — "Physical Exercise Recognition" (MediaPipe keypoints already extracted, from Countix/YouTube clips) | https://www.kaggle.com/datasets/muhannadtuameh/exercise-recognition |
| Labeled exercise clips (time-series keypoint variant) | Kaggle — "Physical Exercise Recognition — Time Series Dataset" | https://www.kaggle.com/datasets/muhannadtuameh/exercise-recognition-time-series |
| Joint-angle / form-correction data (useful for Stage 6 stretch goal) | Kaggle — "Exercise Detection dataset" | https://www.kaggle.com/datasets/mrigaankjaswal/exercise-detection-dataset |
| Academic benchmark clips — has labeled PushUps, PullUps, JumpRope, BodyWeightSquats, JumpingJack, Lunges classes already | UCF101 (official, Center for Research in Computer Vision, UCF) | https://www.crcv.ucf.edu/data/UCF101.php |
| MET values for calorie calculation (standard reference table used in exercise science) | Compendium of Physical Activities (official site, Ainsworth et al.) | https://pacompendium.com/ |
| Testing/demo videos | Real YouTube workout videos, with and without on-screen captions — sourced manually via yt-dlp | https://github.com/yt-dlp/yt-dlp |

**How to find more if needed:** on Kaggle, search "exercise recognition dataset," "workout classification video," or "gym pose landmarks"; on UCF101's page, the relevant fitness-adjacent classes (BodyWeightSquats, JumpingJack, JumpRope, Lunges, PushUps, PullUps, HandstandPushups) can be extracted as a subset without downloading the full 13,320-clip, ~7GB dataset.

**Exercise vocabulary — locked at 12 classes:**
Squats, Push-ups, Jumping Jacks, Lunges, Plank, Burpees, Mountain Climbers, High Knees, Sit-ups, Jump Rope, Bicycle Crunches, Shoulder Press.

---

## 5. Pipeline (build order)

**Stage 0 — Data Prep:** Download labeled clips (Kaggle + UCF101 subset) → extract MediaPipe keypoints per clip (33 landmarks × x,y,z,visibility per frame) → label each sequence by exercise class → cache to disk.

**Stage 1 — Segmentation:** PySceneDetect for hard camera cuts. Within shots, OCR (EasyOCR) detects exercise-name captions/timers — a text change signals a new segment. If no captions, fall back to detecting a sustained shift in joint-angle velocity across a sliding window.

**Stage 2 — Active vs. Rest Detection:** Compute motion magnitude from consecutive pose keypoints. Low, sustained motion + static pose = rest/talking segment → excluded from calorie calculation.

**Stage 3 — Classification:** Feed each active segment's keypoint sequence into the trained LSTM/1D-CNN classifier (primary). OCR text, if present, cross-checks the prediction (secondary). Agreement → high confidence; only one signal available → use it; disagreement → flagged and logged automatically (not silently resolved) — this becomes real material for the evaluation section.

**Stage 4 — Duration Extraction:** Segment length from Stage 1 boundaries, or directly from an OCR-read on-screen countdown timer if visible.

**Stage 5 — Calorie Calculation:**
```
Calories = MET(exercise, intensity_tier) × weight_kg × duration_hours
```
MET values sourced from the Compendium of Physical Activities, tiered into Beginner/Intermediate/Advanced multipliers per exercise. Output per-segment and total estimates across all three tiers.

**Stage 6 — Form Quality Scoring (stretch goal only):** Joint-angle consistency across reps (e.g., squat depth consistency, left/right symmetry) — framed explicitly as experimental, not a core claim.

**Stage 7 — Serve + Visualize:** FastAPI returns results as JSON; React renders a timeline view, per-tier calorie breakdown, and a pose-skeleton overlay synced to detected labels and confidence scores.

---

## 6. Models Used

| Model | Role | Notes |
|---|---|---|
| MediaPipe Pose Landmarker | Keypoint extraction | Pretrained, no training needed — 33 landmarks/frame |
| Majority-class classifier | Baseline | Trivial — predicts the most common class always |
| k-NN on raw keypoints | Baseline | cosine similarity over flattened keypoint vectors |
| LSTM (2-layer) or 1D-CNN | Primary action classifier | Trained on extracted keypoint sequences; this is the project's core trained model |
| EasyOCR (pretrained) | Text detection | No training needed — used for caption/timer reading |

The baselines exist specifically so the LSTM's accuracy is reported as an *improvement over baseline*, not a standalone number — this is what makes the evaluation section defensible rather than just an assertion.

---

## 7. Evaluation Plan (report backbone — do not skip)

1. **Baseline comparison:** report LSTM accuracy against majority-class and k-NN baselines on the same held-out split.
2. **Difficulty-tagged test set:** 15-20 real videos tagged by camera angle (single/multi), captions (present/absent), and subjects (single/multiple people). Report accuracy broken down by tag — this is the evidence for the "robust to real-world video" claim, not just an assertion.
3. **Automatic failure logging:** every Path A/B disagreement and low-confidence prediction logged to file during pipeline runs, analyzed in the report's evaluation section.
4. **Early real-world smoke test:** test on messy real (not clean) YouTube videos starting in Week 2, not late — catch segmentation/robustness problems while there's still time to fix them.

---

## 8. Work Distribution (3 people, 8 weeks)

| Role | Owns |
|---|---|
| **Pose/CV Lead** | MediaPipe extraction, dataset labeling, LSTM/1D-CNN training + evaluation, baseline model, Stage 6 stretch |
| **Pipeline/Fusion Lead** | Segmentation, OCR integration, active/rest detection, fusion logic, difficulty-tagged test-set curation, failure logging |
| **Application Lead** | MET table, calorie module, FastAPI backend, React frontend, skeleton overlay, deployment, CI, integration, report/demo |

---

## 9. Timeline (8 weeks)

| Week | Pose/CV | Pipeline/Fusion | Application |
|---|---|---|---|
| 1-2 | Datasets, keypoint extraction, start LSTM training, build baseline in parallel | Segmentation + OCR working end-to-end; early multi-angle/multi-person smoke test | MET table; FastAPI + React skeletons; CI setup |
| 3 | Finish training, accuracy vs. baseline | Active/rest detection; fix smoke-test issues | Calorie module; React↔FastAPI connection |
| 4 | Export model, test on unseen videos | Fuse OCR + pose logic | **Checkpoint:** real pipeline wired end-to-end through FastAPI into React |
| 5 | Build tagged test set + run full evaluation (joint with Pipeline lead) | Same | Full timeline view + calorie breakdown; start skeleton overlay |
| 6 | Stretch: form-quality scoring | Robustness testing on 10+ real videos | Finish overlay, visible confidence scores, UI polish |
| 7 | Full integration, review failure log | | Deploy (Vercel/Render) with upload size/duration cap |
| 8 | Buffer, write report (evaluation section = baseline + tagged breakdown), demo, slides | | |

---

## 10. Report Structure

1. Problem statement — action recognition + calorie estimation
2. Related work — UCF101/Kinetics as benchmarks, MediaPipe pose estimation, hybrid OCR+pose framing
3. Core contribution: pose classifier — architecture, training curves, per-class confusion matrix, baseline comparison
4. OCR as robustness layer — explicitly secondary, not primary
5. System pipeline — full architecture diagram
6. Evaluation — tagged-set accuracy breakdown, pose-only vs. OCR-only vs. fused comparison, failure case discussion
7. Limitations & future work — form-quality as future work if incomplete, calorie estimates bounded by MET-table accuracy (not clinical-grade), multi-person support
