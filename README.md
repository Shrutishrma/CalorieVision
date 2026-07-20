# CalorieVision 🏋️‍♂️🔥

> **AI-powered exercise recognition and calorie estimation from video.**

CalorieVision ingests a workout video (file upload or YouTube link) and returns a richly annotated timeline: every exercise segment is labelled, timed, and mapped to a calorie burn estimate — all powered by a multi-stage computer vision pipeline.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Input Layer                             │
│            Video Upload  ─────────  YouTube URL                 │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CV Pipeline                                │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │  MediaPipe Pose  │    │  SceneDetect    │                    │
│  │  Estimation      │    │  Segmentation   │                    │
│  │  (33 keypoints)  │    │  (shot cuts)    │                    │
│  └────────┬─────────┘    └────────┬────────┘                   │
│           │                       │                              │
│           ▼                       ▼                              │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │  LSTM / 1D-CNN   │    │   EasyOCR       │                    │
│  │  Action          │    │   On-screen     │                    │
│  │  Classifier      │    │   Text (reps)   │                    │
│  └────────┬─────────┘    └────────┬────────┘                   │
│           │                       │                              │
│           └──────────┬────────────┘                             │
│                      ▼                                          │
│             ┌────────────────┐                                  │
│             │  Fusion Layer   │  ← arbitrates pose vs OCR       │
│             │  + Calorie Calc │                                  │
│             └────────┬───────┘                                  │
└──────────────────────┼──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                              │
│                                                                 │
│  GET  /health         → liveness probe                          │
│  POST /analyse        → submit video, get job_id                │
│  GET  /results/{id}   → poll for Segment list + calories        │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   React + Vite Frontend                          │
│                                                                 │
│  • Video player with skeleton overlay                           │
│  • Interactive exercise timeline                                 │
│  • Per-exercise calorie breakdown chart                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Layout

```
CalorieVision/
├── cv-pipeline/
│   ├── keypoints/       ← MediaPipe keypoint extraction
│   ├── models/          ← LSTM / 1D-CNN weights & training scripts
│   └── eval/            ← Evaluation metrics & confusion matrices
├── fusion-pipeline/
│   ├── segmentation/    ← SceneDetect integration
│   ├── ocr/             ← EasyOCR text detection
│   └── fusion/          ← Segment fusion + calorie calculation
├── app/
│   ├── backend/         ← FastAPI application
│   └── frontend/        ← React + Vite application
├── shared/
│   ├── schemas.py       ← JSON data contract (Segment, Source)
│   └── test-videos/     ← Sample videos for local testing
├── tests/               ← pytest test suite
└── .github/workflows/   ← GitHub Actions CI
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Git

---

### 1 — Clone

```bash
git clone <your-repo-url>
cd CalorieVision
```

---

### 2 — Backend

```bash
# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the API server
uvicorn app.backend.main:app --reload --port 8000
```

The API will be available at **http://localhost:8000**
Interactive docs at **http://localhost:8000/docs**

---

### 3 — Frontend

```bash
cd app/frontend
npm install
npm run dev
```

The UI will be available at **http://localhost:5173**

---

### 4 — Run Tests

```bash
# From the repo root, with the virtual environment activated:

# Fast tests only (no network calls, no heavy ML):
pytest tests/ fusion-pipeline/segmentation/ -m "not network" -v

# All tests including real YouTube download:
pytest -m network -v
```

---

## Running the Pipeline

> **Important:** The download step must run **before** `extract_keypoints.py`.
> `extract_keypoints.py` only accepts a local file path — it does not fetch URLs.

### From a YouTube URL (two-step flow)

```bash
# Step 1 — Download the video (saves to shared/test-videos/<video_id>.mp4)
python fusion-pipeline/segmentation/download_video.py "https://youtu.be/jNQXAC9IVRw" --out shared/test-videos

# Step 2 — Extract pose keypoints from the downloaded file
python cv-pipeline/keypoints/extract_keypoints.py shared/test-videos/jNQXAC9IVRw.mp4 --out output.json
```

### From a local video file

```bash
# Skip the download step — go straight to extraction:
python cv-pipeline/keypoints/extract_keypoints.py path/to/workout.mp4 --out output.json
```

### CLI options

```
download_video.py URL [--out DIR] [--max-height PX] [--overwrite]

  URL            YouTube video URL (long or short form)
  --out DIR      Output directory (default: shared/test-videos)
  --max-height   Cap resolution — 360, 480, 720 (default), 1080
  --overwrite    Re-download even if file already exists

extract_keypoints.py VIDEO [--out FILE] [--max-frames N] [--model-complexity 0|1|2]

  VIDEO              Path to local .mp4 file
  --out FILE         Output JSON path
  --max-frames N     Stop after N frames (fast preview)
  --model-complexity 0=Lite  1=Full (default)  2=Heavy
```

---

## Data Contract

All pipeline stages communicate via the `Segment` schema defined in [`shared/schemas.py`](shared/schemas.py):

| Field        | Type    | Description                              |
|--------------|---------|------------------------------------------|
| `segment_id` | `str`   | Unique segment identifier (e.g. `seg_0001`) |
| `start_time` | `float` | Start of segment in seconds              |
| `end_time`   | `float` | End of segment in seconds                |
| `label`      | `str`   | Exercise label (e.g. `"squat"`)          |
| `confidence` | `float` | Model confidence `[0.0, 1.0]`            |
| `source`     | `enum`  | `"pose"` \| `"ocr"` \| `"fused"`        |

---

## Tech Stack

| Layer        | Technology                        |
|--------------|-----------------------------------|
| Download     | yt-dlp                            |
| Pose         | MediaPipe (Tasks API)             |
| Classifier   | PyTorch (LSTM / 1D-CNN)           |
| OCR          | EasyOCR                           |
| Segmentation | SceneDetect                       |
| Backend      | FastAPI + Uvicorn                 |
| Frontend     | React + Vite                      |
| Testing      | pytest + httpx                    |
| CI           | GitHub Actions                    |

---

## Contributing

1. Fork the repo and create a feature branch.
2. Ensure `pytest tests/` passes before opening a PR.
3. GitHub Actions CI will run automatically on every pull request.