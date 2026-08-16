"""
cv_pipeline/models/pose_heuristic_classifier.py
────────────────────────────────────────────────────────────────────────────────
Rules-based pose classifier that uses actual joint geometry from MediaPipe landmarks
to determine exercise types. No training data required.
"""
from __future__ import annotations
import math
import numpy as np
from typing import List, Tuple, Dict, Any
from shared.schemas import Segment, Source

def calculate_angle(a: tuple, b: tuple, c: tuple) -> float:
    """Calculate the angle between three points in 2D space (degrees)."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

def get_landmark(landmarks: list, index: int) -> tuple | None:
    if len(landmarks) > index:
        lm = landmarks[index]
        if not lm:
            return None
        if 'x' not in lm or 'y' not in lm:
            return None
        return (lm.get('x', 0), lm.get('y', 0))
    return None

def classify_segment(frames: List[dict]) -> Tuple[str, float]:
    """
    Given a list of frames comprising a single active run, determine the exercise class.
    Returns (label, confidence).
    """
    if not frames:
        return ("unknown", 0.0)

    scores = {
        "squat": 0.0,
        "pushup": 0.0,
        "jumping_jack": 0.0,
        "plank": 0.0,
        "lunge": 0.0,
        "burpee": 0.0,
        "mountain_climber": 0.0,
        "high_knees": 0.0,
        "situp": 0.0,
        "jump_rope": 0.0,
        "bicycle_crunch": 0.0,
        "shoulder_press": 0.0
    }
    
    # We will compute statistics over the sequence to classify it
    min_knee_angle = 180.0
    max_knee_angle = 0.0
    min_hip_y = 1.0
    max_hip_y = 0.0
    horizontal_count = 0
    
    # Analyze all frames to build up signals
    for frame in frames:
        lms = frame.get('landmarks', [])
        if not lms:
            continue
            
        l_shoulder = get_landmark(lms, 11)
        r_shoulder = get_landmark(lms, 12)
        l_hip = get_landmark(lms, 23)
        r_hip = get_landmark(lms, 24)
        l_knee = get_landmark(lms, 25)
        r_knee = get_landmark(lms, 26)
        l_ankle = get_landmark(lms, 27)
        r_ankle = get_landmark(lms, 28)
        
        # Check horizontal vs vertical posture
        if l_shoulder and l_hip and l_ankle:
            # y goes down in image coords
            shoulder_y = (l_shoulder[1] + r_shoulder[1])/2 if r_shoulder else l_shoulder[1]
            hip_y = (l_hip[1] + r_hip[1])/2 if r_hip else l_hip[1]
            ankle_y = (l_ankle[1] + r_ankle[1])/2 if r_ankle else l_ankle[1]
            
            min_hip_y = min(min_hip_y, hip_y)
            max_hip_y = max(max_hip_y, hip_y)
            
            # If shoulder y and ankle y are close, the person is horizontal
            if abs(shoulder_y - ankle_y) < 0.3:
                horizontal_count += 1
                
        # Knee angles
        if l_hip and l_knee and l_ankle:
            l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
            r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle) if (r_hip and r_knee and r_ankle) else l_knee_angle
            avg_knee_angle = (l_knee_angle + r_knee_angle) / 2
            
            min_knee_angle = min(min_knee_angle, avg_knee_angle)
            max_knee_angle = max(max_knee_angle, avg_knee_angle)
            
    total_frames = len(frames)
    if total_frames == 0:
        return ("unknown", 0.0)
        
    horizontal_ratio = horizontal_count / total_frames
    hip_displacement = max_hip_y - min_hip_y
    
    # 1. Horizontal exercises (Plank, Pushup, Mountain Climber)
    if horizontal_ratio > 0.5:
        if hip_displacement > 0.1:
            # High hip movement in horizontal pos -> Mountain Climber or Burpee (if transitioning)
            scores["mountain_climber"] = 0.8
        else:
            # Static horizontal
            scores["plank"] = 0.85
            # If we saw elbow angles changing, it would be a pushup
            scores["pushup"] = 0.75
    else:
        # 2. Vertical exercises (Squat, Jumping Jack, Lunge, High Knees)
        if min_knee_angle < 130 and hip_displacement > 0.1:
            scores["squat"] = 0.85
            scores["lunge"] = 0.7
        else:
            # Upright, less knee bend -> Jumping jack, shoulder press
            scores["jumping_jack"] = 0.7
            scores["shoulder_press"] = 0.65
            
    # VERY naive heuristic fallback for now - we just pick the highest score
    # This guarantees we give a reasonable answer based on posture
    best_label = "squat"
    best_score = 0.0
    for label, score in scores.items():
        if score > best_score:
            best_score = score
            best_label = label
            
    # Default confidence based on heuristics
    confidence = best_score if best_score > 0 else 0.5
    
    return (best_label, confidence)


def classify_sequence(frames: List[dict], active_runs: List[dict]) -> List[Segment]:
    """
    Given the full list of frames and a list of active runs from motion_filter,
    return a list of Segment objects classified by the heuristic classifier.
    """
    segments = []
    
    for i, run in enumerate(active_runs):
        start_time = run.get('start_time', 0.0)
        end_time = run.get('end_time', 0.1)
        run_frames = run.get('frames', [])
        label = run.get('label', 'active')
        
        if label == 'rest':
            seg = Segment(
                segment_id=f"seg_{i}",
                start_time=round(start_time, 2),
                end_time=round(end_time, 2),
                label="rest",
                confidence=1.0,
                source=Source.pose
            )
            segments.append(seg)
        else:
            # It's an active run, classify it
            ex_label, conf = classify_segment(run_frames)
            seg = Segment(
                segment_id=f"seg_{i}",
                start_time=round(start_time, 2),
                end_time=round(end_time, 2),
                label=ex_label,
                confidence=round(conf, 2),
                source=Source.pose
            )
            segments.append(seg)
            
    return segments
