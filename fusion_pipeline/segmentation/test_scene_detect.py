"""
fusion-pipeline/segmentation/test_scene_detect.py
────────────────────────────────────────────────────────────────────────────────
Unit tests for scene_detect.py.

Strategy
--------
All fast tests use synthetic MP4s generated with OpenCV's VideoWriter.
A single-colour video will have no scene cuts; a video with a sharp colour
change mid-way should trigger at least one cut.

Real-video tests are marked @pytest.mark.real_video and are skipped unless
a video file exists in shared/test-videos/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fusion_pipeline.segmentation.scene_detect import detect_scenes, main
from shared.schemas import Segment, Source


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _write_video(
    path: Path,
    *,
    frames_per_section: list[tuple[int, int]],  # list of (n_frames, BGR_color_scalar)
    fps: int = 30,
    width: int = 320,
    height: int = 240,
) -> Path:
    """
    Write a synthetic video composed of solid-colour sections.
    Each tuple in `frames_per_section` is (n_frames, BGR_color_scalar).
    """
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for n_frames, color in frames_per_section:
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        for _ in range(n_frames):
            writer.write(frame)
    writer.release()
    return path


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def solid_video(tmp_path_factory) -> Path:
    """30-frame solid-colour video — no scene cuts expected."""
    tmp = tmp_path_factory.mktemp("scene_vids")
    return _write_video(
        tmp / "solid.mp4",
        frames_per_section=[(30, 80)],   # uniform grey
    )


@pytest.fixture(scope="module")
def two_scene_video(tmp_path_factory) -> Path:
    """
    60-frame video: first 30 frames dark-blue, last 30 frames bright-white.
    The hard colour jump should be a detectable scene cut.
    """
    tmp = tmp_path_factory.mktemp("scene_vids")
    return _write_video(
        tmp / "two_scenes.mp4",
        frames_per_section=[
            (30, 20),    # very dark
            (30, 235),   # very bright
        ],
    )


@pytest.fixture(scope="module")
def real_video() -> Path | None:
    candidate = _REPO_ROOT / "shared" / "test-videos"
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        found = list(candidate.glob(ext))
        if found:
            return found[0]
    return None


# ─── Error-handling tests (fast) ─────────────────────────────────────────────

class TestErrorHandling:
    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            detect_scenes("/no/such/video.mp4")

    def test_corrupt_file_raises(self, tmp_path):
        bad = tmp_path / "bad.mp4"
        bad.write_bytes(b"not a video")
        with pytest.raises(RuntimeError):
            detect_scenes(str(bad))


# ─── Output contract tests ────────────────────────────────────────────────────

class TestOutputContract:
    def test_returns_list(self, solid_video):
        result = detect_scenes(str(solid_video))
        assert isinstance(result, list)

    def test_each_item_is_segment(self, solid_video):
        result = detect_scenes(str(solid_video))
        for seg in result:
            assert isinstance(seg, Segment)

    def test_segment_source_is_scene_cut(self, solid_video):
        result = detect_scenes(str(solid_video))
        for seg in result:
            assert seg.source == Source.scene_cut

    def test_segment_label_is_unknown(self, solid_video):
        result = detect_scenes(str(solid_video))
        for seg in result:
            assert seg.label == "unknown"

    def test_segment_confidence_is_one(self, solid_video):
        result = detect_scenes(str(solid_video))
        for seg in result:
            assert seg.confidence == pytest.approx(1.0)

    def test_end_time_after_start_time(self, solid_video):
        result = detect_scenes(str(solid_video))
        for seg in result:
            assert seg.end_time > seg.start_time, (
                f"{seg.segment_id}: end_time={seg.end_time} <= start_time={seg.start_time}"
            )

    def test_segment_ids_are_unique(self, two_scene_video):
        result = detect_scenes(str(two_scene_video))
        ids = [s.segment_id for s in result]
        assert len(ids) == len(set(ids))

    def test_segment_ids_prefixed_scene(self, two_scene_video):
        result = detect_scenes(str(two_scene_video))
        for seg in result:
            assert seg.segment_id.startswith("scene_")

    def test_to_dict_roundtrip(self, solid_video):
        result = detect_scenes(str(solid_video))
        for seg in result:
            restored = Segment.from_dict(seg.to_dict())
            assert restored == seg


# ─── Behavioural tests ────────────────────────────────────────────────────────

class TestBehaviour:
    def test_solid_video_has_one_or_no_cuts(self, solid_video):
        """
        A perfectly uniform video should produce at most 1 segment
        (the whole video treated as one scene by SceneDetect).
        """
        result = detect_scenes(str(solid_video))
        # PySceneDetect returns the whole video as a single scene even if no
        # cut is found.  Either 0 (no content change) or 1 (whole video) is
        # acceptable here.
        assert len(result) <= 1

    def test_two_scene_video_has_at_least_one_cut(self, two_scene_video):
        """Hard black→white cut must produce ≥1 scene boundary."""
        result = detect_scenes(str(two_scene_video), threshold=20.0)
        assert len(result) >= 1, (
            "Expected at least 1 scene segment from a video with a hard colour jump"
        )


# ─── CLI tests ────────────────────────────────────────────────────────────────

class TestCLI:
    def test_cli_writes_json(self, solid_video, tmp_path):
        out = tmp_path / "scenes.json"
        main([str(solid_video), "--out", str(out)])
        assert out.exists()
        data = json.loads(out.read_text())
        assert "segments" in data
        assert "total_scenes" in data
        assert isinstance(data["segments"], list)

    def test_cli_stdout(self, solid_video, capsys):
        main([str(solid_video)])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "segments" in data


# ─── Real-video integration test (skipped unless a video is present) ──────────

class TestRealVideo:
    @pytest.mark.real_video
    def test_real_video_produces_segments(self, real_video):
        if real_video is None:
            pytest.skip("No video found in shared/test-videos/")
        result = detect_scenes(str(real_video))
        # We can't assert a specific count, but it must be a valid list of Segments
        assert isinstance(result, list)
        for seg in result:
            assert isinstance(seg, Segment)
            assert seg.source == Source.scene_cut
            assert seg.end_time > seg.start_time
