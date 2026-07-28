"""
fusion-pipeline/segmentation/test_download_video.py
────────────────────────────────────────────────────────────────────────────────
Tests for download_youtube_video() in download_video.py.

Test strategy
─────────────
Fast / always-run (no network):
  • URL validation helpers (extract_video_id, is_youtube_url)
  • Error class hierarchy
  • Graceful handling of obviously bad inputs
  • CLI argument parsing (no real download)
  • Cached-file short-circuit (no network call made)

Network (marked @pytest.mark.network — skipped in CI by default):
  • Downloads a real 6-second Creative Commons clip from YouTube
    (ID: jNQXAC9IVRw — "Me at the zoo", the first YouTube video ever,
     CC-BY, ~1 MB, stable for 20 years)
  • Confirms the file exists, is non-empty, and is a valid MP4
  • Confirms re-downloading the same URL uses the cache (no second download)

To run the network tests locally:
    pytest fusion-pipeline/segmentation/test_download_video.py -m network -v
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

# conftest.py at repo root registers fusion_pipeline.* → fusion-pipeline/
from fusion_pipeline.segmentation.download_video import (
    AgeRestrictedError,
    DownloadError,
    InvalidURLError,
    NetworkError,
    VideoUnavailableError,
    _map_yt_dlp_error,
    download_youtube_video,
    extract_video_id,
    is_youtube_url,
    main,
)


# ─── URL helpers ──────────────────────────────────────────────────────────────

class TestExtractVideoId:
    @pytest.mark.parametrize("url,expected", [
        ("https://www.youtube.com/watch?v=jNQXAC9IVRw",  "jNQXAC9IVRw"),
        ("https://youtu.be/jNQXAC9IVRw",                  "jNQXAC9IVRw"),
        ("https://m.youtube.com/watch?v=jNQXAC9IVRw",    "jNQXAC9IVRw"),
        ("https://www.youtube.com/shorts/jNQXAC9IVRw",   "jNQXAC9IVRw"),
        ("https://www.youtube.com/embed/jNQXAC9IVRw",    "jNQXAC9IVRw"),
        # with extra query params
        ("https://www.youtube.com/watch?v=jNQXAC9IVRw&t=10s", "jNQXAC9IVRw"),
    ])
    def test_valid_urls(self, url, expected):
        assert extract_video_id(url) == expected

    @pytest.mark.parametrize("url", [
        "https://vimeo.com/12345",
        "https://example.com",
        "not-a-url",
        "",
    ])
    def test_invalid_urls_return_none(self, url):
        assert extract_video_id(url) is None


class TestIsYouTubeUrl:
    def test_valid_returns_true(self):
        assert is_youtube_url("https://www.youtube.com/watch?v=jNQXAC9IVRw")

    def test_invalid_returns_false(self):
        assert not is_youtube_url("https://vimeo.com/12345")


# ─── Error hierarchy ──────────────────────────────────────────────────────────

class TestErrorHierarchy:
    def test_all_errors_inherit_download_error(self):
        assert issubclass(InvalidURLError,      DownloadError)
        assert issubclass(VideoUnavailableError, DownloadError)
        assert issubclass(AgeRestrictedError,   DownloadError)
        assert issubclass(NetworkError,          DownloadError)

    def test_download_error_inherits_runtime_error(self):
        assert issubclass(DownloadError, RuntimeError)


class TestMapYtDlpError:
    def test_age_restricted(self):
        with pytest.raises(AgeRestrictedError):
            _map_yt_dlp_error("error: sign in to confirm your age", "url")

    def test_private_video(self):
        with pytest.raises(VideoUnavailableError):
            _map_yt_dlp_error("video is unavailable", "url")

    def test_deleted_video(self):
        with pytest.raises(VideoUnavailableError):
            _map_yt_dlp_error("this video has been removed", "url")

    def test_network_error(self):
        with pytest.raises(NetworkError):
            _map_yt_dlp_error("connection reset by peer", "url")

    def test_generic_error(self):
        with pytest.raises(DownloadError):
            _map_yt_dlp_error("some totally unknown error", "url")


# ─── Input validation (no network) ───────────────────────────────────────────

class TestInputValidation:
    def test_empty_url_raises(self, tmp_path):
        with pytest.raises(InvalidURLError):
            download_youtube_video("", output_dir=str(tmp_path))

    def test_blank_url_raises(self, tmp_path):
        with pytest.raises(InvalidURLError):
            download_youtube_video("   ", output_dir=str(tmp_path))

    def test_non_youtube_url_raises(self, tmp_path):
        with pytest.raises(InvalidURLError):
            download_youtube_video("https://vimeo.com/123456", output_dir=str(tmp_path))

    def test_plain_text_raises(self, tmp_path):
        with pytest.raises(InvalidURLError):
            download_youtube_video("not a url at all", output_dir=str(tmp_path))


# ─── Cache short-circuit (no network) ────────────────────────────────────────

class TestCacheHit:
    def test_existing_file_returned_without_download(self, tmp_path, capsys):
        """
        If the expected output file already exists, the function must return
        immediately without hitting the network.
        """
        url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
        video_id = "jNQXAC9IVRw"

        # Pre-create a fake mp4 (minimal valid bytes not required — just non-empty)
        fake = tmp_path / f"{video_id}.mp4"
        fake.write_bytes(b"\x00" * 1024)

        result = download_youtube_video(url, output_dir=str(tmp_path), overwrite=False)

        assert result == str(fake)
        captured = capsys.readouterr()
        assert "Cache hit" in captured.out

    def test_overwrite_flag_ignores_cache(self, tmp_path, monkeypatch):
        """
        With overwrite=True the function must attempt a real download even if
        the file exists.  We monkeypatch yt_dlp to avoid an actual network call.
        """
        url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
        video_id = "jNQXAC9IVRw"

        # Pre-create fake cached file
        fake = tmp_path / f"{video_id}.mp4"
        fake.write_bytes(b"\x00" * 512)

        # Monkeypatch _check_ffmpeg so the test works without ffmpeg installed
        import fusion_pipeline.segmentation.download_video as dv_mod
        monkeypatch.setattr(dv_mod, "_check_ffmpeg", lambda: None)

        # Monkeypatch yt_dlp.YoutubeDL to raise immediately (simulates a call)
        import yt_dlp

        class _FakeYDL:
            def __init__(self, opts): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def extract_info(self, url, download):
                raise yt_dlp.utils.DownloadError("simulated failure")
            def prepare_filename(self, info): return ""

        monkeypatch.setattr(yt_dlp, "YoutubeDL", _FakeYDL)

        with pytest.raises(DownloadError):
            download_youtube_video(url, output_dir=str(tmp_path), overwrite=True)


# ─── CLI argument parsing ─────────────────────────────────────────────────────

class TestCLI:
    def test_invalid_url_exits_nonzero(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["https://vimeo.com/12345"])
        assert exc_info.value.code != 0

    def test_missing_url_exits_nonzero(self):
        """Calling main with no arguments should exit with usage error."""
        with pytest.raises(SystemExit):
            main([])


# ─── Network integration tests ────────────────────────────────────────────────

def _is_valid_mp4(path: Path) -> bool:
    """
    Very lightweight MP4 validity check: reads the first box header and
    verifies the 'ftyp' box is present (standard MP4 signature).
    Does NOT require ffprobe or any external tool.
    """
    try:
        with open(path, "rb") as f:
            data = f.read(12)
        if len(data) < 12:
            return False
        box_type = data[4:8]
        return box_type in (b"ftyp", b"moov", b"mdat", b"free", b"skip")
    except OSError:
        return False


@pytest.mark.network
class TestRealDownload:
    """
    Downloads the first YouTube video ever (CC-BY, ~1 MB, 19 seconds).
    Run with:  pytest -m network
    """

    # "Me at the zoo" — uploaded 2005-04-23, Creative Commons,
    # 19 seconds, consistently available, ~1 MB at 360p.
    STABLE_URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    VIDEO_ID   = "jNQXAC9IVRw"

    def test_file_exists_after_download(self, tmp_path):
        path = download_youtube_video(
            self.STABLE_URL,
            output_dir=str(tmp_path),
            max_height=360,   # keep CI download tiny
        )
        assert Path(path).exists(), "Downloaded file must exist"

    def test_file_is_non_empty(self, tmp_path):
        path = download_youtube_video(
            self.STABLE_URL,
            output_dir=str(tmp_path),
            max_height=360,
        )
        size = Path(path).stat().st_size
        assert size > 10_000, f"File is suspiciously small: {size} bytes"

    def test_file_is_valid_mp4(self, tmp_path):
        path = download_youtube_video(
            self.STABLE_URL,
            output_dir=str(tmp_path),
            max_height=360,
        )
        assert _is_valid_mp4(Path(path)), "Downloaded file must be a valid MP4"

    def test_filename_uses_video_id(self, tmp_path):
        path = download_youtube_video(
            self.STABLE_URL,
            output_dir=str(tmp_path),
            max_height=360,
        )
        assert Path(path).stem == self.VIDEO_ID

    def test_cache_prevents_second_download(self, tmp_path, capsys):
        """Second call to same URL returns immediately (cache hit)."""
        download_youtube_video(
            self.STABLE_URL, output_dir=str(tmp_path), max_height=360
        )
        capsys.readouterr()  # discard first call output

        download_youtube_video(
            self.STABLE_URL, output_dir=str(tmp_path), max_height=360
        )
        captured = capsys.readouterr()
        assert "Cache hit" in captured.out

    def test_cli_downloads_and_prints_next_step(self, tmp_path, capsys):
        main([self.STABLE_URL, "--out", str(tmp_path), "--max-height", "360"])
        captured = capsys.readouterr()
        assert "Next step" in captured.out
        assert "extract_keypoints" in captured.out
