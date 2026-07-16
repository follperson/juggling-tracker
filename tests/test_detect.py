import numpy as np

from juggletrack.detect import BallDetector
from juggletrack.detect.fake import FakeDetector
from juggletrack.sim import simulate_cascade


def test_fake_detector_returns_per_frame_detections():
    r = simulate_cascade(n_throws=5, fps=30.0, seed=1)
    det = FakeDetector(r.detections)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    expected_frames = {d.frame_idx for d in r.detections}
    some_frame = min(expected_frames)
    got = det.detect(frame, some_frame, some_frame / 30.0)
    assert got == [d for d in r.detections if d.frame_idx == some_frame]


def test_fake_detector_empty_for_unknown_frame():
    det = FakeDetector([])
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    assert det.detect(frame, 999, 33.3) == []


def test_fake_detector_satisfies_protocol():
    det: BallDetector = FakeDetector([])
    assert callable(det.detect)
