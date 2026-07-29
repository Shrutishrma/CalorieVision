"""
fusion_pipeline/ocr/ocr_normalise.py
────────────────────────────────────────────────────────────────────────────────
OCR Text Normalizer.

Maps raw OCR text extracted from video frames (e.g., "30 SQUATS", "PUSH-UPS x15",
"Jumping Jacks 0:45") into the standard 12-class exercise vocabulary (snake_case).

NOTE: This is a minimal implementation scoped to fusion_pipeline/.
Planned ownership belongs to Pose/CV Lead (cv_pipeline/models/ocr_normalise.py).
Reconcile with team upon baseline integration.
"""

from __future__ import annotations

import re
from typing import Optional

EXERCISE_VOCABULARY = [
    "squat",
    "pushup",
    "jumping_jack",
    "lunge",
    "plank",
    "burpee",
    "mountain_climber",
    "high_knees",
    "situp",
    "jump_rope",
    "bicycle_crunch",
    "shoulder_press",
]

# Patterns mapping regexes to canonical exercise names (multi-word first!)
_PATTERNS = [
    (re.compile(r"\bjump(ing)?[\s\-_]*jack(s)?\b", re.IGNORECASE), "jumping_jack"),
    (re.compile(r"\bmountain[\s\-_]*climber(s)?\b", re.IGNORECASE), "mountain_climber"),
    (re.compile(r"\bhigh[\s\-_]*knee(s)?\b", re.IGNORECASE), "high_knees"),
    (re.compile(r"\bjump(ing)?[\s\-_]*rope\b", re.IGNORECASE), "jump_rope"),
    (re.compile(r"\bskip(ping)?[\s\-_]*rope\b", re.IGNORECASE), "jump_rope"),
    (re.compile(r"\bbicycle[\s\-_]*crunch(es)?\b", re.IGNORECASE), "bicycle_crunch"),
    (re.compile(r"\bshoulder[\s\-_]*press\b", re.IGNORECASE), "shoulder_press"),
    (re.compile(r"\boverhead[\s\-_]*press\b", re.IGNORECASE), "shoulder_press"),
    (re.compile(r"\bsquat(s|ting)?\b", re.IGNORECASE), "squat"),
    (re.compile(r"\bpush[\s\-_]*up(s)?\b", re.IGNORECASE), "pushup"),
    (re.compile(r"\blunge(s|ing)?\b", re.IGNORECASE), "lunge"),
    (re.compile(r"\bplank(s|ing)?\b", re.IGNORECASE), "plank"),
    (re.compile(r"\bburpee(s)?\b", re.IGNORECASE), "burpee"),
    (re.compile(r"\bsit[\s\-_]*up(s)?\b", re.IGNORECASE), "situp"),
    (re.compile(r"\bcrunch(es)?\b", re.IGNORECASE), "situp"),
]


def normalise_ocr_text(text: str) -> str:
    """
    Normalise raw OCR text string into one of the 12 exercise classes.
    Returns "unknown" if no pattern matches.

    Parameters
    ----------
    text : str
        Raw OCR text string.

    Returns
    -------
    str
        Canonical snake_case label or "unknown".
    """
    if not text or not isinstance(text, str):
        return "unknown"

    clean_text = text.strip()

    # 1. Exact match if already valid vocabulary string
    lower_text = clean_text.lower()
    if lower_text in EXERCISE_VOCABULARY:
        return lower_text

    # 2. Regex pattern matching
    for pattern, canonical in _PATTERNS:
        if pattern.search(clean_text):
            return canonical

    # 3. Simple fallback substring check
    for canonical in EXERCISE_VOCABULARY:
        parts = canonical.split("_")
        if all(part in lower_text for part in parts):
            return canonical

    return "unknown"
