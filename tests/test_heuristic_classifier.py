import pytest
from cv_pipeline.models.pose_heuristic_classifier import classify_segment, calculate_angle, get_landmark

def test_calculate_angle():
    a = (0, 1)
    b = (0, 0)
    c = (1, 0)
    assert calculate_angle(a, b, c) == 90.0
    
    # 180 degrees
    a = (-1, 0)
    b = (0, 0)
    c = (1, 0)
    assert calculate_angle(a, b, c) == 180.0

def test_classify_segment_empty():
    label, conf = classify_segment([])
    assert label == "unknown"
    assert conf == 0.0

def test_classify_segment_squat():
    # Synthetic frames representing a squat
    # We just need to give it landmarks that represent standing, then squatting (knee angle < 130), then standing
    # We also need hip displacement > 0.1
    
    def make_lms(shoulder_y, hip_y, knee_x, knee_y, ankle_y):
        lms = [{} for _ in range(33)]
        lms[11] = {'x': 0.5, 'y': shoulder_y} # L shoulder
        lms[23] = {'x': 0.5, 'y': hip_y} # L hip
        lms[25] = {'x': knee_x, 'y': knee_y} # L knee
        lms[27] = {'x': 0.5, 'y': ankle_y} # L ankle
        return lms

    frames = []
    frames.append({"landmarks": make_lms(0.2, 0.5, 0.5, 0.7, 0.9)}) # Standing
    frames.append({"landmarks": make_lms(0.4, 0.7, 0.7, 0.7, 0.9)}) # Squatting
    frames.append({"landmarks": make_lms(0.2, 0.5, 0.5, 0.7, 0.9)}) # Standing
    
    label, conf = classify_segment(frames)
    assert label == "squat"
    assert conf > 0.5

def test_classify_segment_plank():
    # Synthetic frames representing a plank
    # Body should be horizontal (shoulder y ~= ankle y)
    def make_plank_lms():
        lms = [{} for _ in range(33)]
        lms[11] = {'x': 0.2, 'y': 0.8} # L shoulder
        lms[23] = {'x': 0.5, 'y': 0.8} # L hip
        lms[25] = {'x': 0.7, 'y': 0.8} # L knee
        lms[27] = {'x': 0.9, 'y': 0.8} # L ankle
        return lms

    frames = []
    for _ in range(5):
        frames.append({"landmarks": make_plank_lms()})
        
    label, conf = classify_segment(frames)
    assert label == "plank"
    assert conf > 0.5
