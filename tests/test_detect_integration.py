"""Integration tests that exercise real YOLO weights. Deselected by default
(pytest addopts -m 'not detector'); run with: uv run pytest -m detector -v
"""
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.detector

ROOT = Path(__file__).resolve().parents[1]
REAL_VIDEO = Path(os.environ.get("JUGGLETRACK_TEST_VIDEO", ROOT / "data/raw/af2.mp4"))
LOCAL_WEIGHTS = ROOT / "yolo11n.pt"


def test_yolo_detects_balls_in_real_juggling_video():
    if not REAL_VIDEO.exists():
        pytest.skip("real juggling video not available on this machine")
    from juggletrack.detect.yolo import YOLODetector
    from juggletrack.video.reader import VideoReader

    model = str(LOCAL_WEIGHTS) if LOCAL_WEIGHTS.exists() else "yolo11n.pt"
    det = YOLODetector(model_path=model)
    found = []
    with VideoReader(REAL_VIDEO) as reader:
        for idx, t, frame in reader.frames():
            if idx >= 90:  # first 3 seconds
                break
            found.extend(det.detect(frame, idx, t))
    assert found, "stock YOLO found zero sports balls in 90 frames of juggling"
    for d in found:
        assert 0.0 <= d.x <= 1.0 and 0.0 <= d.y <= 1.0
        assert 0.0 < d.w <= 1.0 and 0.0 < d.h <= 1.0
        assert 0.0 < d.confidence <= 1.0
