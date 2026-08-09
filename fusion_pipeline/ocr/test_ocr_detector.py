"""
fusion-pipeline/ocr/test_ocr_detector.py
────────────────────────────────────────────────────────────────────────────────
Unit tests for ocr_detector.py.

Strategy
--------
EasyOCR is a heavy dependency (~200 MB model download) and is GPU-optional.
Rather than actually running OCR in CI, we monkey-patch `easyocr.Reader`
with a lightweight stub that returns predictable results.

This lets us test:
  • Frame-sampling logic (every N seconds)
  • Confidence filtering
  • Segment schema compliance (source=ocr, valid ids, end>start)
  • CLI output format
  • Error handling (missing file, bad video)

Real-model tests are gated behind @pytest.mark.network and skipped in CI.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.schemas import Segment, Source


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _write_solid_video(
    path: Path,
    n_frames: int = 60,
    fps: int = 30,
    width: int = 160,
    height: int = 120,
    color: int = 100,
) -> Path:
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    frame = np.full((height, width, 3), color, dtype=np.uint8)
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()
    return path


class _FakeReader:
    """Stub for easyocr.Reader that returns deterministic results."""

    def __init__(self, *args, call_count: int = 0, **kwargs):
        self._call_count = call_count  # mutable list trick via outer scope

    def readtext(self, img, **kwargs):
        # Returns one result per call with a fixed label and confidence.
        # Format: (bbox, text, confidence)
        return [
            (None, "10 reps", 0.85),
        ]


def _make_fake_easyocr_module(reader_cls: type = _FakeReader) -> types.ModuleType:
    """Build a minimal fake `easyocr` module containing `Reader`."""
    mod = types.ModuleType("easyocr")
    mod.Reader = reader_cls  # type: ignore[attr-defined]
    return mod


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def solid_video(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("ocr_vids")
    return _write_solid_video(tmp / "test.mp4", n_frames=90, fps=30)  # 3 s


# ─── Error-handling tests ─────────────────────────────────────────────────────

class TestErrorHandling:
    def test_missing_file_raises(self):
        from fusion_pipeline.ocr.ocr_detector import detect_ocr_segments
        with pytest.raises(FileNotFoundError):
            detect_ocr_segments("/no/such/video.mp4")

    def test_corrupt_file_raises(self, tmp_path):
        from fusion_pipeline.ocr.ocr_detector import detect_ocr_segments
        bad = tmp_path / "bad.mp4"
        bad.write_bytes(b"not a video")
        with pytest.raises(RuntimeError):
            with patch.dict(sys.modules, {"easyocr": _make_fake_easyocr_module()}):
                detect_ocr_segments(str(bad))


# ─── Sampling and output contract tests ──────────────────────────────────────

class TestOutputContract:

    def _run_with_fake_ocr(self, video_path: str, **kwargs) -> list[Segment]:
        """Helper: run detect_ocr_segments with stubbed EasyOCR."""
        fake_mod = _make_fake_easyocr_module()
        with patch.dict(sys.modules, {"easyocr": fake_mod}):
            from fusion_pipeline.ocr import ocr_detector
            # Also patch the module-level import inside ocr_detector
            with patch.object(ocr_detector, "_easyocr" if hasattr(ocr_detector, "_easyocr") else "__name__", new=None, create=True):
                # Patch at the easyocr import inside the function
                import importlib
                import fusion_pipeline.ocr.ocr_detector as mod
                original = sys.modules.get("easyocr")
                sys.modules["easyocr"] = fake_mod
                try:
                    return mod.detect_ocr_segments(video_path, **kwargs)
                finally:
                    if original is None:
                        sys.modules.pop("easyocr", None)
                    else:
                        sys.modules["easyocr"] = original

    def test_returns_list(self, solid_video):
        result = self._run_with_fake_ocr(str(solid_video), sample_every_n_seconds=1.0)
        assert isinstance(result, list)

    def test_each_item_is_segment(self, solid_video):
        result = self._run_with_fake_ocr(str(solid_video), sample_every_n_seconds=1.0)
        for seg in result:
            assert isinstance(seg, Segment)

    def test_source_is_ocr(self, solid_video):
        result = self._run_with_fake_ocr(str(solid_video), sample_every_n_seconds=1.0)
        for seg in result:
            assert seg.source == Source.ocr

    def test_end_time_after_start_time(self, solid_video):
        result = self._run_with_fake_ocr(str(solid_video), sample_every_n_seconds=1.0)
        for seg in result:
            assert seg.end_time > seg.start_time

    def test_segment_ids_unique(self, solid_video):
        result = self._run_with_fake_ocr(str(solid_video), sample_every_n_seconds=1.0)
        ids = [s.segment_id for s in result]
        assert len(ids) == len(set(ids))

    def test_segment_ids_prefixed_ocr(self, solid_video):
        result = self._run_with_fake_ocr(str(solid_video), sample_every_n_seconds=1.0)
        for seg in result:
            assert seg.segment_id.startswith("ocr_")

    def test_confidence_in_valid_range(self, solid_video):
        result = self._run_with_fake_ocr(str(solid_video), sample_every_n_seconds=1.0)
        for seg in result:
            assert 0.0 <= seg.confidence <= 1.0

    def test_to_dict_roundtrip(self, solid_video):
        result = self._run_with_fake_ocr(str(solid_video), sample_every_n_seconds=1.0)
        for seg in result:
            restored = Segment.from_dict(seg.to_dict())
            assert restored == seg

    def test_sampling_rate_respected(self, solid_video):
        """A 3-second video sampled every 1 second should produce ≤3 segments."""
        result = self._run_with_fake_ocr(str(solid_video), sample_every_n_seconds=1.0)
        # 3 s video, 1 s sample → at most 3 sampled frames (0, 1, 2 s)
        assert len(result) <= 3

    def test_min_confidence_filters_low_conf(self, solid_video):
        """With min_confidence=0.99, no results from the fake reader (conf=0.85) should pass."""

        class _LowConfReader:
            def __init__(self, *a, **kw):
                pass
            def readtext(self, img, **kw):
                return [(None, "reps", 0.30)]  # below any threshold > 0.30

        fake_mod = _make_fake_easyocr_module(reader_cls=_LowConfReader)
        sys.modules["easyocr"] = fake_mod
        try:
            import fusion_pipeline.ocr.ocr_detector as mod
            result = mod.detect_ocr_segments(
                str(solid_video),
                sample_every_n_seconds=1.0,
                min_confidence=0.99,
            )
            assert result == [], f"Expected empty list with min_confidence=0.99, got {result}"
        finally:
            sys.modules.pop("easyocr", None)


# ─── CLI tests ────────────────────────────────────────────────────────────────

class TestCLI:
    def test_cli_writes_json(self, solid_video, tmp_path):
        out = tmp_path / "ocr_out.json"
        fake_mod = _make_fake_easyocr_module()
        sys.modules["easyocr"] = fake_mod
        try:
            from fusion_pipeline.ocr.ocr_detector import main
            main([str(solid_video), "--sample-rate", "1.0", "--out", str(out)])
        finally:
            sys.modules.pop("easyocr", None)

        assert out.exists()
        data = json.loads(out.read_text())
        assert "segments" in data
        assert "total_ocr_segments" in data
        assert isinstance(data["segments"], list)
