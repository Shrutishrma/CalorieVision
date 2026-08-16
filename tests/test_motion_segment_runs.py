import pytest
from cv_pipeline.motion.motion_filter import segment_runs, ACTIVE_LABEL, REST_LABEL

def _make_frame(timestamp, motion_score):
    return {
        "timestamp": timestamp,
        # We need landmarks for compute_motion_scores to work.
        # But wait, label_motion in motion_filter computes scores based on landmarks.
        # So we have to construct fake landmarks with displacement.
        "landmarks": [{"x": 0.5, "y": 0.5 + motion_score}]
    }

def test_segment_runs_empty():
    assert segment_runs([]) == []

def test_segment_runs_contiguous(monkeypatch):
    # Mock label_motion directly to avoid complex landmark mocking
    import cv_pipeline.motion.motion_filter as mf
    
    frames = [
        {"timestamp": 0.0},
        {"timestamp": 1.0},
        {"timestamp": 2.0},
        {"timestamp": 3.0},
        {"timestamp": 4.0},
        {"timestamp": 5.0},
    ]
    labels = [REST_LABEL, ACTIVE_LABEL, ACTIVE_LABEL, ACTIVE_LABEL, ACTIVE_LABEL, REST_LABEL]
    monkeypatch.setattr(mf, "label_motion", lambda f, threshold: labels)
    
    runs = mf.segment_runs(frames, min_active_secs=2.0, min_gap_secs=1.0)
    assert len(runs) == 1
    assert runs[0]["start_time"] == 1.0
    assert runs[0]["end_time"] == 4.0
    assert len(runs[0]["frames"]) == 4

def test_segment_runs_merge_gap(monkeypatch):
    import cv_pipeline.motion.motion_filter as mf
    
    frames = [
        {"timestamp": 0.0},
        {"timestamp": 1.0}, # active 1 start
        {"timestamp": 2.0}, # active 1 end
        {"timestamp": 3.0}, # gap (rest)
        {"timestamp": 4.0}, # active 2 start
        {"timestamp": 5.0}, # active 2 end
    ]
    labels = [
        REST_LABEL, 
        ACTIVE_LABEL, ACTIVE_LABEL, 
        REST_LABEL, 
        ACTIVE_LABEL, ACTIVE_LABEL
    ]
    monkeypatch.setattr(mf, "label_motion", lambda f, threshold: labels)
    
    # Gap is 2 seconds (from 2.0 to 4.0)
    # With min_gap_secs = 3.0, it should merge
    runs = mf.segment_runs(frames, min_active_secs=1.0, min_gap_secs=3.0)
    assert len(runs) == 1
    assert runs[0]["start_time"] == 1.0
    assert runs[0]["end_time"] == 5.0
    # Frames from index 1 to 5
    assert len(runs[0]["frames"]) == 5

def test_segment_runs_no_merge(monkeypatch):
    import cv_pipeline.motion.motion_filter as mf
    
    frames = [
        {"timestamp": 0.0},
        {"timestamp": 1.0}, # active 1 start
        {"timestamp": 2.0}, # active 1 end
        {"timestamp": 3.0}, # rest
        {"timestamp": 4.0}, # rest
        {"timestamp": 5.0}, # rest
        {"timestamp": 6.0}, # active 2 start
        {"timestamp": 7.0}, # active 2 end
    ]
    labels = [
        REST_LABEL, 
        ACTIVE_LABEL, ACTIVE_LABEL, 
        REST_LABEL, REST_LABEL, REST_LABEL,
        ACTIVE_LABEL, ACTIVE_LABEL
    ]
    monkeypatch.setattr(mf, "label_motion", lambda f, threshold: labels)
    
    # Gap is 4 seconds (from 2.0 to 6.0)
    # With min_gap_secs = 2.0, it should NOT merge
    runs = mf.segment_runs(frames, min_active_secs=1.0, min_gap_secs=2.0)
    assert len(runs) == 2
    assert runs[0]["start_time"] == 1.0
    assert runs[0]["end_time"] == 2.0
    assert runs[1]["start_time"] == 6.0
    assert runs[1]["end_time"] == 7.0
