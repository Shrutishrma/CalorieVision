import json
import pytest
from pathlib import Path
from cv_pipeline.models.lstm_classifier import load_lstm_model, classify_run, KeypointLSTM

@pytest.fixture(scope="module")
def model_and_type():
    m, m_type = load_lstm_model()
    return m, m_type

@pytest.fixture(scope="module")
def real_data():
    dataset_path = Path("dataset/train_data.json")
    labels_path = Path("dataset/train_labels.json")
    if not dataset_path.exists() or not labels_path.exists():
        pytest.skip("dataset/train_data.json not found")
    with open(dataset_path, "r") as f:
        frames = json.load(f)["frames"]
    with open(labels_path, "r") as f:
        labels = json.load(f)
    return frames, labels

def test_model_loading(model_and_type):
    model, m_type = model_and_type
    assert model is not None
    assert m_type in ("keypoint_132", "kinematic_14")

def test_pushup_prediction(model_and_type, real_data):
    model, _ = model_and_type
    frames, labels = real_data
    pushup_frames = [f for f, l in zip(frames, labels) if l == "pushup"]
    assert len(pushup_frames) >= 20
    label, conf = classify_run(pushup_frames[:30], model)
    assert isinstance(label, str) and label != "unknown"
    assert conf > 0.3

def test_squat_prediction(model_and_type, real_data):
    model, _ = model_and_type
    frames, labels = real_data
    squat_frames = [f for f, l in zip(frames, labels) if l == "squat"]
    assert len(squat_frames) >= 20
    label, conf = classify_run(squat_frames[:30], model)
    assert isinstance(label, str) and label != "unknown"
    assert conf > 0.3

def test_plank_prediction(model_and_type, real_data):
    model, _ = model_and_type
    frames, labels = real_data
    plank_frames = [f for f, l in zip(frames, labels) if l == "plank"]
    assert len(plank_frames) >= 20
    label, conf = classify_run(plank_frames[:30], model)
    assert isinstance(label, str) and label != "unknown"
    assert conf > 0.3
