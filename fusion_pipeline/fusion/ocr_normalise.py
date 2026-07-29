"""
fusion_pipeline/fusion/ocr_normalise.py
────────────────────────────────────────────────────────────────────────────────
Normalizes raw EasyOCR text strings into the canonical 12-class exercise
vocabulary (snake_case) used by CalorieVision.

12 locked exercise classes:
  - squat
  - pushup
  - jumping_jack
  - lunge
  - plank
  - burpee
  - mountain_climber
  - high_knees
  - situp
  - jump_rope
  - bicycle_crunch
  - shoulder_press

Note: This implementation is scoped to fusion_pipeline/ to unblock fusion layer
OCR label alignment, duplicating planned ownership of cv_pipeline/models/ocr_normalise.py.
"""

from __future__ import annotations

import re

VOCABULARY = [
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

# Mapping rules: regex pattern -> snake_case class
_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(bicycle\s*crunch(es)?|bicycles?)\b", re.IGNORECASE), "bicycle_crunch"),
    (re.compile(r"\b(mountain\s*climber(s)?|climbers?)\b", re.IGNORECASE), "mountain_climber"),
    (re.compile(r"\b(jumping\s*jack(s)?|jacks?)\b", re.IGNORECASE), "jumping_jack"),
    (re.compile(r"\b(high\s*knee(s)?)\b", re.IGNORECASE), "high_knees"),
    (re.compile(r"\b(jump\s*rope|jumping\s*rope|rope\s*jumping)\b", re.IGNORECASE), "jump_rope"),
    (re.compile(r"\b(shoulder\s*press|overhead\s*press)\b", re.IGNORECASE), "shoulder_press"),
    (re.compile(r"\b(push\s*-?\s*up(s)?)\b", re.IGNORECASE), "pushup"),
    (re.compile(r"\b(sit\s*-?\s*up(s)?|crunch(es)?)\b", re.IGNORECASE), "situp"),
    (re.compile(r"\b(squat(s)?)\b", re.IGNORECASE), "squat"),
    (re.compile(r"\b(lunge(s)?)\b", re.IGNORECASE), "lunge"),
    (re.compile(r"\b(plank(s)?)\b", re.IGNORECASE), "plank"),
    (re.compile(r"\b(burpee(s)?)\b", re.IGNORECASE), "burpee"),
]


def ocr_normalise(text: str | None) -> str:
    """
    Normalize raw OCR text to one of the 12 locked exercise classes in snake_case.

    Parameters
    ----------
    text : str | None
        Raw text extracted by EasyOCR (e.g. "30 SQUATS", "Push-ups", "SET 1: BURPEES").

    Returns
    -------
    str
        Canonical snake_case class if matched, otherwise stripped lowercased original text
        or "unknown" if empty.
    """
    if not text or not text.strip():
        return "unknown"

    cleaned = text.strip()

    # Rule-based regex matching
    for pattern, canonical_label in _RULES:
        if pattern.search(cleaned):
            return canonical_label

    # Fallback to cleaned lowercase string
    cleaned_lower = cleaned.lower()
    for vocab in VOCABULARY:
        if vocab in cleaned_lower:
            return vocab

    return cleaned_lower
