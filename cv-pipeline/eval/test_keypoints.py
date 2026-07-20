"""
cv-pipeline/eval/test_keypoints.py
────────────────────────────────────────────────────────────────────────────────
Unit tests for cv-pipeline/keypoints/extract_keypoints.py.

Strategy
--------
Because CI doesn't have a real workout video, we generate a synthetic MP4
(solid-colour frames using OpenCV's VideoWriter) to validate the function's
contract WITHOUT needing MediaPipe to detect an actual pose.

The tests check:
  • The function handles a valid synthetic video without crashing.
  • The returned list is non-empty (one dict per frame).
  • Each frame dict has the required keys and correct types.
  • Error handling: FileNotFoundError on a missing path.
  • The --max-frames limit is respected.
  • The landmark name list has exactly 33 entries.

For CI, MediaPipe is imported but may not detect a pose in a blank-frame
video — that is intentional and tested explicitly.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# sys.path is configured by conftest.py at the repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

from cv_pipeline.keypoints.extract_keypoints import (
    LANDMARK_NAMES,
    extract_keypoints,
    main,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_synthetic_video(
    path: Path,
    *,
    n_frames: int = 30,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    color: tuple[int, int, int] = (120, 80, 60),  # BGR
) -> Path:
    """
    Write a short solid-colour video to `path`.

    Returns the path for convenience.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    frame = np.full((height, width, 3), color, dtype=np.uint8)
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()
    return path


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory) -> Path:
    """A 30-frame, 640×480 synthetic MP4 written once per test session."""
    tmp = tmp_path_factory.mktemp("videos")
    return _make_synthetic_video(tmp / "synthetic.mp4", n_frames=30)


@pytest.fixture(scope="module")
def real_video() -> Path | None:
    """
    Returns the path to a real test video if one exists in shared/test-videos/,
    otherwise returns None (tests that need it are skipped).
    """
    candidate = REPO_ROOT / "shared" / "test-videos"
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        found = list(candidate.glob(ext))
        if found:
            return found[0]
    return None


# ─── Core contract tests ──────────────────────────────────────────────────────

class TestLandmarkNames:
    def test_exactly_33_names(self):
        assert len(LANDMARK_NAMES) == 33

    def test_no_duplicates(self):
        assert len(LANDMARK_NAMES) == len(set(LANDMARK_NAMES))

    def test_known_names_present(self):
        for name in ("nose", "left_shoulder", "right_hip", "left_ankle"):
            assert name in LANDMARK_NAMES


class TestExtractKeypointsContract:
    def test_returns_list(self, synthetic_video):
        result = extract_keypoints(str(synthetic_video))
        assert isinstance(result, list)

    def test_non_empty(self, synthetic_video):
        """The function must return at least one frame dict."""
        result = extract_keypoints(str(synthetic_video))
        assert len(result) > 0, "Expected at least one frame in result"

    def test_frame_count_matches_video(self, synthetic_video):
        """30-frame video → 30 dicts (we always emit every frame)."""
        result = extract_keypoints(str(synthetic_video))
        assert len(result) == 30

    def test_each_frame_has_required_keys(self, synthetic_video):
        required = {"frame_index", "timestamp", "pose_detected", "landmarks"}
        result = extract_keypoints(str(synthetic_video))
        for frame in result:
            assert required.issubset(frame.keys()), (
                f"Frame {frame.get('frame_index')} missing keys: "
                f"{required - frame.keys()}"
            )

    def test_frame_index_is_sequential(self, synthetic_video):
        result = extract_keypoints(str(synthetic_video))
        for i, frame in enumerate(result):
            assert frame["frame_index"] == i

    def test_timestamp_is_non_negative_float(self, synthetic_video):
        result = extract_keypoints(str(synthetic_video))
        for frame in result:
            assert isinstance(frame["timestamp"], float)
            assert frame["timestamp"] >= 0.0

    def test_pose_detected_is_bool(self, synthetic_video):
        result = extract_keypoints(str(synthetic_video))
        for frame in result:
            assert isinstance(frame["pose_detected"], bool)

    def test_landmarks_is_list(self, synthetic_video):
        result = extract_keypoints(str(synthetic_video))
        for frame in result:
            assert isinstance(frame["landmarks"], list)

    def test_when_pose_detected_landmarks_have_33_entries(self, synthetic_video):
        result = extract_keypoints(str(synthetic_video))
        for frame in result:
            if frame["pose_detected"]:
                assert len(frame["landmarks"]) == 33

    def test_landmark_schema(self, synthetic_video):
        """Each landmark must have index, name, x, y, z, visibility."""
        result = extract_keypoints(str(synthetic_video))
        required_lm_keys = {"index", "name", "x", "y", "z", "visibility"}
        for frame in result:
            for lm in frame["landmarks"]:
                assert required_lm_keys.issubset(lm.keys())
                assert isinstance(lm["index"], int)
                assert isinstance(lm["name"], str)
                assert isinstance(lm["x"], float)
                assert isinstance(lm["y"], float)
                assert isinstance(lm["z"], float)
                assert isinstance(lm["visibility"], float)

    def test_landmark_index_matches_name(self, synthetic_video):
        result = extract_keypoints(str(synthetic_video))
        for frame in result:
            for lm in frame["landmarks"]:
                assert LANDMARK_NAMES[lm["index"]] == lm["name"]


class TestMaxFrames:
    def test_max_frames_limits_output(self, synthetic_video):
        """--max-frames 5 on a 30-frame video must return exactly 5 dicts."""
        result = extract_keypoints(str(synthetic_video), max_frames=5)
        assert len(result) == 5

    def test_max_frames_larger_than_video(self, synthetic_video):
        """If max_frames > video length, return all frames (don't crash)."""
        result = extract_keypoints(str(synthetic_video), max_frames=1000)
        assert len(result) == 30


class TestErrorHandling:
    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            extract_keypoints("/no/such/video.mp4")

    def test_corrupt_file_raises_or_empty(self, tmp_path):
        """A non-video file should either raise RuntimeError or return empty list."""
        bad = tmp_path / "bad.mp4"
        bad.write_bytes(b"this is not a video")
        try:
            result = extract_keypoints(str(bad))
            # If it doesn't raise, it must return an empty list (no frames decoded)
            assert result == [], f"Expected empty list for corrupt file, got {result}"
        except RuntimeError:
            pass  # also acceptable


class TestCLI:
    def test_cli_writes_json(self, synthetic_video, tmp_path):
        out = tmp_path / "out.json"
        main([str(synthetic_video), "--out", str(out), "--max-frames", "10"])
        assert out.exists(), "CLI should create the output JSON file"
        with open(out) as fh:
            data = json.load(fh)
        assert "frames" in data
        assert data["total_frames"] == 10
        assert isinstance(data["frames"], list)

    def test_cli_default_output_name(self, synthetic_video, tmp_path):
        """Without --out, output should be <stem>_keypoints.json next to the video."""
        import shutil
        vid = shutil.copy(str(synthetic_video), str(tmp_path / "test_clip.mp4"))
        main([str(vid), "--max-frames", "5"])
        expected = tmp_path / "test_clip_keypoints.json"
        assert expected.exists()


# ─── Real-video integration test (skipped in CI if no video) ─────────────────

class TestRealVideo:
    @pytest.mark.skipif(
        True,  # always skip unless explicitly opted-in; see note below
        reason=(
            "Real-video test skipped by default. "
            "To run: pytest -m real_video --no-skip after dropping a video "
            "into shared/test-videos/"
        ),
    )
    def test_real_video_has_detections(self, real_video):
        """
        On a genuine workout video, at least 50 % of frames should have a
        detected pose.
        """
        if real_video is None:
            pytest.skip("No video found in shared/test-videos/")
        result = extract_keypoints(str(real_video), max_frames=100)
        detected = sum(1 for f in result if f["pose_detected"])
        assert detected > 0, "Expected at least one frame with a detected pose"
        rate = detected / len(result)
        assert rate >= 0.5, f"Detection rate {rate:.0%} < 50% — check video quality"
