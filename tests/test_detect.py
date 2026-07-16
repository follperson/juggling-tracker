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


def test_detections_from_xywhn_maps_fields():
    import numpy as np

    from juggletrack.detect.yolo import detections_from_xywhn

    xywhn = np.array([[0.5, 0.6, 0.05, 0.08], [0.2, 0.3, 0.04, 0.04]])
    confs = np.array([0.9, 0.15])
    dets = detections_from_xywhn(xywhn, confs, frame_idx=7, t=7 / 30.0)
    assert len(dets) == 2
    d = dets[0]
    assert (d.frame_idx, d.t) == (7, 7 / 30.0)
    assert (d.x, d.y, d.w, d.h) == (0.5, 0.6, 0.05, 0.08)
    assert d.confidence == 0.9


def test_detections_from_xywhn_empty():
    import numpy as np

    from juggletrack.detect.yolo import detections_from_xywhn

    assert detections_from_xywhn(np.zeros((0, 4)), np.zeros(0), 0, 0.0) == []
