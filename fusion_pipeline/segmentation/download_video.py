"""
fusion_pipeline/segmentation/download_video.py
────────────────────────────────────────────────────────────────────────────────
Stage 0 of the CalorieVision pipeline: YouTube (and generic URL) video download.

Downloads a video to a local file, then returns the path so the next stage
(cv-pipeline/keypoints/extract_keypoints.py) can process it.

Usage (library)
---------------
    from fusion_pipeline.segmentation.download_video import download_youtube_video
    path = download_youtube_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

Usage (CLI)
-----------
    python download_video.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    python download_video.py "https://youtu.be/dQw4w9WgXcQ" --out shared/test-videos
    python download_video.py URL --out /tmp/vids --max-height 480
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional


# ─── Error hierarchy ──────────────────────────────────────────────────────────

class DownloadError(RuntimeError):
    """Base class for all download failures."""


class InvalidURLError(DownloadError):
    """Raised when the URL is not recognised as a supported video URL."""


class VideoUnavailableError(DownloadError):
    """Raised when the video exists but is private, deleted, or geo-blocked."""


class AgeRestrictedError(DownloadError):
    """Raised when the video requires age verification and no cookies are set."""


class NetworkError(DownloadError):
    """Raised on transient network failures."""


# ─── ffmpeg check ──────────────────────────────────────────────────────────────

import subprocess as _subprocess


def _check_ffmpeg() -> None:
    """
    Verify that ffmpeg is installed and available on PATH.

    yt-dlp downloads video and audio as separate streams and requires ffmpeg
    to merge them into a single .mp4.  Without it, downloads silently fail.

    Raises
    ------
    RuntimeError
        With platform-specific installation instructions if ffmpeg is missing.
    """
    import os
    venv_bin = Path(sys.executable).parent
    if (venv_bin / "ffmpeg.exe").exists() or (venv_bin / "ffmpeg").exists():
        os.environ["PATH"] = f"{str(venv_bin)}{os.pathsep}{os.environ.get('PATH', '')}"

    try:
        _subprocess.run(
            ["ffmpeg", "-version"],
            stdout=_subprocess.DEVNULL,
            stderr=_subprocess.DEVNULL,
            check=True,
        )
    except (FileNotFoundError, _subprocess.CalledProcessError):
        raise RuntimeError(
            "ffmpeg is not installed or not found on PATH.\n"
            "yt-dlp requires ffmpeg to merge video and audio streams.\n\n"
            "Install instructions:\n"
            "  Windows : winget install ffmpeg\n"
            "  macOS   : brew install ffmpeg\n"
            "  Linux   : sudo apt install ffmpeg\n\n"
            "After installing, open a NEW terminal and retry."
        )


# ─── URL validation ───────────────────────────────────────────────────────────

# Matches both youtube.com/watch?v= and youtu.be/ short links, plus Shorts.
_YOUTUBE_RE = re.compile(
    r"(?:https?://)?(?:www\.|m\.)?(?:"
    r"youtube\.com/(?:watch\?v=|shorts/|embed/|v/)"
    r"|youtu\.be/"
    r")([\w\-]{11})",
    re.IGNORECASE,
)


def extract_video_id(url: str) -> Optional[str]:
    """Return the 11-character YouTube video ID, or None if not a YouTube URL."""
    m = _YOUTUBE_RE.search(url)
    return m.group(1) if m else None


def is_youtube_url(url: str) -> bool:
    """Return True if the URL looks like a valid YouTube video link."""
    return _YOUTUBE_RE.search(url) is not None


# ─── Core downloader ──────────────────────────────────────────────────────────

def download_youtube_video(
    url: str,
    output_dir: str = "shared/test-videos",
    *,
    max_height: int = 720,
    overwrite: bool = False,
) -> str:
    """
    Download a YouTube video and return its local file path.
    (Also available as `download_video` alias).

    Parameters
    ----------
    url : str
        YouTube video URL.  Both long-form
        (``https://www.youtube.com/watch?v=XXXXXXXXXXX``) and short-form
        (``https://youtu.be/XXXXXXXXXXX``) are accepted.
    output_dir : str
        Directory in which to save the downloaded file.  Created if it does
        not exist.  Defaults to ``shared/test-videos``.
    max_height : int
        Maximum vertical resolution to download.  Defaults to 720 (HD).
        Use 480 for smaller files, 1080 for higher quality.
    overwrite : bool
        If ``True``, re-download even if the file already exists.
        If ``False`` (default), return the cached path immediately.

    Returns
    -------
    str
        Absolute path to the downloaded ``.mp4`` file.

    Raises
    ------
    InvalidURLError
        The URL is not a recognised YouTube link.
    VideoUnavailableError
        The video is private, deleted, removed, or geo-blocked.
    AgeRestrictedError
        The video requires age verification.
    NetworkError
        A transient network error occurred during download.
    DownloadError
        Any other yt-dlp error.
    """
    import yt_dlp  # import here so the module is importable even without yt-dlp

    # ── Validate URL ──────────────────────────────────────────────────────────
    if not url or not url.strip():
        raise InvalidURLError("URL must not be empty.")

    video_id = extract_video_id(url.strip())
    if video_id is None:
        raise InvalidURLError(
            f"Not a recognised YouTube URL: {url!r}\n"
            "Expected format: https://www.youtube.com/watch?v=XXXXXXXXXXX "
            "or https://youtu.be/XXXXXXXXXXX"
        )

    # ── Resolve output path ───────────────────────────────────────────────────
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filename is based on the video ID only — avoids all title encoding issues.
    out_path = out_dir / f"{video_id}.mp4"

    if out_path.exists() and not overwrite:
        print(f"[CalorieVision] Cache hit — using existing file: {out_path}")
        return str(out_path)

    # ── Check system dependency (only needed for actual download) ────────────
    _check_ffmpeg()

    # ── yt-dlp options ────────────────────────────────────────────────────────
    # Format selector explanation:
    #   bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]
    #     → best MP4 video track at or below max_height + best M4A audio
    #   /bestvideo[height<={max_height}]+bestaudio
    #     → fallback: any container, will be remuxed to mp4
    #   /best[height<={max_height}]
    #     → last resort: single merged stream
    fmt = (
        f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]"
        f"/bestvideo[height<={max_height}]+bestaudio"
        f"/best[height<={max_height}]"
        f"/best"
    )

    ydl_opts: dict = {
        "format":           fmt,
        "outtmpl":          str(out_dir / f"{video_id}.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet":            True,        # suppress verbose yt-dlp output
        "noplaylist":       True,        # never download a whole playlist
        "nocheckcertificate": True,       # prevent Windows Python SSL verification failures
        "socket_timeout":   30,          # seconds - fail fast on stale connections
        "retries":          3,
        "fragment_retries": 3,
        "postprocessors": [
            {
                # Ensure the output is a proper mp4 container
                "key":            "FFmpegVideoConvertor",
                "preferedformat": "mp4",
            }
        ],
    }

    print(f"[CalorieVision] Downloading video ID={video_id} (max {max_height}p)...")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url.strip(), download=True)
            # Resolve the actual output filename (yt-dlp may change extension)
            actual_path = Path(ydl.prepare_filename(info))
            # If yt-dlp produced a non-mp4 name, point at the remuxed mp4
            if not actual_path.suffix.lower() == ".mp4":
                actual_path = actual_path.with_suffix(".mp4")

    except yt_dlp.utils.DownloadError as exc:
        msg = str(exc).lower()
        _map_yt_dlp_error(msg, url)   # raises the appropriate CalorieVision error

    if not out_path.exists():
        # yt-dlp may have saved with a slightly different name; try to find it
        candidates = list(out_dir.glob(f"{video_id}.*"))
        mp4_candidates = [c for c in candidates if c.suffix.lower() == ".mp4"]
        if mp4_candidates:
            out_path = mp4_candidates[0]
        elif candidates:
            out_path = candidates[0]
        else:
            raise DownloadError(
                f"Download appeared to succeed but no output file found in {out_dir}"
            )

    size_mb = out_path.stat().st_size / 1_048_576
    print(f"[CalorieVision] Saved -> {out_path}  ({size_mb:.1f} MB)")
    return str(out_path)


def _map_yt_dlp_error(msg: str, url: str) -> None:
    """
    Inspect a yt-dlp error message and raise the most specific CalorieVision
    exception type.  Always raises.
    """
    age_phrases   = ("age", "sign in", "age-restricted", "age restricted")
    priv_phrases  = ("private", "has been removed", "unavailable", "not available",
                     "video is unavailable", "members-only", "deleted")
    net_phrases   = ("connection", "timeout", "network", "errno", "reset by peer",
                     "temporary failure", "name or service not known")

    if any(p in msg for p in age_phrases):
        raise AgeRestrictedError(
            f"Video at {url!r} requires age verification.\n"
            "Pass browser cookies via yt-dlp's --cookies-from-browser option "
            "or supply a cookies.txt file."
        )
    if any(p in msg for p in priv_phrases):
        raise VideoUnavailableError(
            f"Video at {url!r} is private, deleted, or unavailable in your region."
        )
    if any(p in msg for p in net_phrases):
        raise NetworkError(
            f"Network error while downloading {url!r}.\n"
            "Check your internet connection and try again."
        )
    raise DownloadError(
        f"Failed to download {url!r}.\n"
        f"yt-dlp reported: {msg}"
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="download_video",
        description=(
            "Download a YouTube video and save it as MP4 for the "
            "CalorieVision pose extraction pipeline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("url", help="YouTube video URL")
    p.add_argument(
        "--out", "-o",
        default="shared/test-videos",
        metavar="DIR",
        help="Output directory for the downloaded MP4",
    )
    p.add_argument(
        "--max-height", type=int, default=720,
        metavar="PX",
        help="Maximum video height in pixels (e.g. 480, 720, 1080)",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Re-download even if the file already exists",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        path = download_youtube_video(
            args.url,
            output_dir=args.out,
            max_height=args.max_height,
            overwrite=args.overwrite,
        )
        print(f"\nNext step — extract pose keypoints:\n"
              f"  python cv-pipeline/keypoints/extract_keypoints.py \"{path}\" "
              f"--out output.json")
    except DownloadError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

# Clean alias matching real app function convention
download_video = download_youtube_video
