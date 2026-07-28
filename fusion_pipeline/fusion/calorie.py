"""
fusion_pipeline/fusion/calorie.py
────────────────────────────────────────────────────────────────────────────────
Calorie estimation module for CalorieVision (leMON Team 15).

Formula
-------
    kcal = MET × weight_kg × duration_hours

where MET is from the Compendium of Physical Activities
(Ainsworth et al. 2011, Med. Sci. Sports Exerc. 43(8):1575-1581).

The 12 locked exercise classes are mapped to Compendium codes for three
difficulty tiers (Beginner / Intermediate / Advanced).  Tier differences
model differences in exercise intensity / pace:

  • Beginner:     moderate pace, partial range of motion
  • Intermediate: standard pace, full range of motion
  • Advanced:     high pace, weighted/plyometric variants

Usage (library)
---------------
    from fusion_pipeline.fusion.calorie import estimate_calories, CalorieReport
    report = estimate_calories(fused_segments, weight_kg=70.0)
    print(report.total_kcal["intermediate"])

Usage (CLI)
-----------
    python fusion_pipeline/fusion/calorie.py fused.json --weight 70 --out calories.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.schemas import Segment

# ─── MET table ────────────────────────────────────────────────────────────────
# Source: Ainsworth et al. 2011 Compendium of Physical Activities
# Code   Name                        Compendium code / notes
# ──────────────────────────────────────────────────────────────────────────────
# Tier values represent Beginner / Intermediate / Advanced intensity.

MET_TABLE: Dict[str, Dict[str, float]] = {
    # exercise class        beginner  intermediate  advanced
    "squat":              {"beginner": 3.5, "intermediate": 5.0, "advanced": 6.0},
    "pushup":             {"beginner": 3.8, "intermediate": 5.0, "advanced": 6.5},
    "jumping_jack":       {"beginner": 7.7, "intermediate": 8.0, "advanced": 9.0},
    "lunge":              {"beginner": 3.5, "intermediate": 4.0, "advanced": 5.5},
    "plank":              {"beginner": 2.8, "intermediate": 3.5, "advanced": 4.0},
    "burpee":             {"beginner": 6.0, "intermediate": 8.0, "advanced": 10.0},
    "mountain_climber":   {"beginner": 5.5, "intermediate": 7.0, "advanced": 8.5},
    "high_knees":         {"beginner": 7.0, "intermediate": 8.0, "advanced": 9.5},
    "situp":              {"beginner": 3.5, "intermediate": 4.5, "advanced": 5.5},
    "jump_rope":          {"beginner": 8.8, "intermediate": 10.0, "advanced": 12.3},
    "bicycle_crunch":     {"beginner": 3.0, "intermediate": 4.0, "advanced": 5.0},
    "shoulder_press":     {"beginner": 3.5, "intermediate": 5.0, "advanced": 6.0},
    # Fallback for unknown / scene_cut segments
    "unknown":            {"beginner": 1.5, "intermediate": 1.5, "advanced": 1.5},
}

TIERS = ("beginner", "intermediate", "advanced")
DEFAULT_WEIGHT_KG = 70.0


# ─── Calorie calculation ──────────────────────────────────────────────────────

def kcal_for_segment(
    label:        str,
    duration_secs: float,
    weight_kg:    float,
    tier:         str,
) -> float:
    """
    Compute calories for a single segment.

    Parameters
    ----------
    label         : Exercise class name (must be in MET_TABLE; falls back to "unknown").
    duration_secs : Duration in seconds.
    weight_kg     : Person's body weight in kg.
    tier          : "beginner", "intermediate", or "advanced".

    Returns
    -------
    float  Kilocalories (kcal).
    """
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS!r}, got {tier!r}")

    row  = MET_TABLE.get(label.lower(), MET_TABLE["unknown"])
    met  = row[tier]
    hours = duration_secs / 3600.0
    return round(met * weight_kg * hours, 4)


# ─── Report dataclass ─────────────────────────────────────────────────────────

@dataclass
class SegmentCalorie:
    """Per-segment calorie entry."""
    segment_id:    str
    start_time:    float
    end_time:      float
    label:         str
    duration_secs: float
    confidence:    float
    kcal:          Dict[str, float] = field(default_factory=dict)  # tier → kcal

    def to_dict(self) -> dict:
        return {
            "segment_id":    self.segment_id,
            "start_time":    self.start_time,
            "end_time":      self.end_time,
            "label":         self.label,
            "duration_secs": self.duration_secs,
            "confidence":    self.confidence,
            "kcal":          self.kcal,
        }


@dataclass
class CalorieReport:
    """Full calorie report for a video pipeline run."""
    weight_kg:     float
    segments:      List[SegmentCalorie] = field(default_factory=list)
    total_kcal:    Dict[str, float] = field(default_factory=dict)  # tier → total

    def to_dict(self) -> dict:
        return {
            "weight_kg":    self.weight_kg,
            "total_kcal":   self.total_kcal,
            "total_segments": len(self.segments),
            "segments":     [s.to_dict() for s in self.segments],
        }


# ─── Main estimation function ──────────────────────────────────────────────────

def estimate_calories(
    segments:  List[Segment],
    *,
    weight_kg: float = DEFAULT_WEIGHT_KG,
) -> CalorieReport:
    """
    Estimate calories for all fused segments across all three tiers.

    Parameters
    ----------
    segments   : Fused segment list (from fusion.py output).
    weight_kg  : Body weight in kg.  Default 70 kg.

    Returns
    -------
    CalorieReport
        Contains per-segment kcal and tier totals.
    """
    seg_entries: List[SegmentCalorie] = []
    totals: Dict[str, float] = {tier: 0.0 for tier in TIERS}

    for seg in segments:
        dur = max(0.0, seg.end_time - seg.start_time)
        kcal_per_tier = {
            tier: kcal_for_segment(seg.label, dur, weight_kg, tier)
            for tier in TIERS
        }
        for tier in TIERS:
            totals[tier] = round(totals[tier] + kcal_per_tier[tier], 4)

        seg_entries.append(SegmentCalorie(
            segment_id    = seg.segment_id,
            start_time    = seg.start_time,
            end_time      = seg.end_time,
            label         = seg.label,
            duration_secs = round(dur, 3),
            confidence    = seg.confidence,
            kcal          = kcal_per_tier,
        ))

    return CalorieReport(
        weight_kg   = weight_kg,
        segments    = seg_entries,
        total_kcal  = totals,
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate calories from fused segments using MET values "
            "(Ainsworth et al. 2011 Compendium of Physical Activities)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("fused_json",
                        help="JSON of fused segments (fusion.py output)")
    parser.add_argument("--weight", "-w", type=float, default=DEFAULT_WEIGHT_KG,
                        help="Body weight in kg (used for kcal = MET × kg × h)")
    parser.add_argument("--out", "-o", default=None,
                        help="Output JSON path.  Omit to print to stdout.")
    args = parser.parse_args(argv)

    raw = json.loads(Path(args.fused_json).read_text(encoding="utf-8"))
    items = raw.get("segments", raw) if isinstance(raw, dict) else raw
    segments = [Segment.from_dict(d) for d in items]

    report = estimate_calories(segments, weight_kg=args.weight)

    # ── Human-readable summary to stderr ─────────────────────────────────────
    print(f"\n── Calorie Estimate (weight={args.weight} kg) ──", file=sys.stderr)
    print(f"  {'Tier':<14} {'kcal':>8}", file=sys.stderr)
    print(f"  {'─'*14} {'─'*8}", file=sys.stderr)
    for tier in TIERS:
        print(f"  {tier.capitalize():<14} {report.total_kcal[tier]:>8.2f}", file=sys.stderr)
    print(f"  ({len(segments)} segments)\n", file=sys.stderr)

    out_str = json.dumps(report.to_dict(), indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(out_str, encoding="utf-8")
        print(f"[calorie] Written -> {out}")
    else:
        print(out_str)


if __name__ == "__main__":
    main()
