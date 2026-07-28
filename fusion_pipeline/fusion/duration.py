"""
fusion_pipeline/fusion/duration.py
────────────────────────────────────────────────────────────────────────────────
Duration extraction for fused segments.

Primary source: end_time - start_time from the Segment boundary (always available).
Secondary source: parse OCR-readable timer strings ("0:30", "30s", "2:15") if
present in the OCR segment label — used as a cross-check / override when the
OCR timer is more reliable than the scene-cut boundaries.

Usage (library)
---------------
    from fusion_pipeline.fusion.duration import duration_seconds, annotate_durations

    secs = duration_seconds(segment)          # float seconds
    annotated = annotate_durations(segments)  # list of (segment, seconds, source)
"""

from __future__ import annotations

import re
from typing import List, Literal, Tuple

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.schemas import Segment

# ─── Timer regex patterns ─────────────────────────────────────────────────────

# Matches: "0:30", "1:05", "2:15:03"
_COLON_TIMER  = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b")
# Matches: "30s", "45 s", "90sec", "2min"
_SHORT_TIMER  = re.compile(r"\b(\d+)\s*(s(?:ec(?:onds?)?)?|min(?:utes?)?)\b", re.IGNORECASE)

DurationSource = Literal["boundary", "ocr_timer"]


def _parse_timer_string(text: str) -> float | None:
    """
    Try to parse a timer string into seconds.

    Returns None if no timer is detected.
    """
    # Try MM:SS or HH:MM:SS first
    m = _COLON_TIMER.search(text)
    if m:
        parts = [int(g) for g in m.groups() if g is not None]
        if len(parts) == 2:
            minutes, seconds = parts
            return float(minutes * 60 + seconds)
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return float(hours * 3600 + minutes * 60 + seconds)

    # Try Ns / Nmin
    m = _SHORT_TIMER.search(text)
    if m:
        value, unit = m.group(1), m.group(2).lower()
        value = float(value)
        if unit.startswith("m"):
            return value * 60.0
        return value

    return None


def duration_seconds(
    segment: Segment,
    *,
    ocr_label: str | None = None,
) -> Tuple[float, DurationSource]:
    """
    Extract duration from a segment.

    Parameters
    ----------
    segment   : Segment to measure.
    ocr_label : Optional raw OCR text from a co-occurring OCR segment.
                If it contains a parseable timer, that overrides the boundary.

    Returns
    -------
    (seconds, source)
        seconds : duration in seconds (always positive)
        source  : "boundary" or "ocr_timer"
    """
    # Try OCR timer first (more reliable for workout-rep timer videos)
    if ocr_label:
        parsed = _parse_timer_string(ocr_label)
        if parsed is not None and parsed > 0.0:
            return parsed, "ocr_timer"

    # Fallback: boundary arithmetic
    dur = max(0.0, segment.end_time - segment.start_time)
    return dur, "boundary"


def annotate_durations(
    segments: List[Segment],
    *,
    ocr_segments: List[Segment] | None = None,
) -> List[dict]:
    """
    Annotate a list of segments with extracted durations.

    Parameters
    ----------
    segments    : Fused segments to annotate.
    ocr_segments: Optional OCR segments to mine for timer text.

    Returns
    -------
    list[dict]
        Each dict: {"segment": Segment, "duration_seconds": float, "duration_source": str}
    """
    from fusion_pipeline.fusion.fusion import _overlap

    results = []
    for seg in segments:
        # Find best co-occurring OCR text (highest-confidence overlapping OCR)
        ocr_label = None
        if ocr_segments:
            candidates = [
                s for s in ocr_segments
                if _overlap(seg, s) > 0.0
            ]
            if candidates:
                best = max(candidates, key=lambda s: s.confidence)
                ocr_label = best.label

        dur, source = duration_seconds(seg, ocr_label=ocr_label)
        results.append({
            "segment": seg,
            "duration_seconds": round(dur, 3),
            "duration_source": source,
        })

    return results
