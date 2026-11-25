import sys
import types

import numpy as np

from juggletrack.detect import BallDetector
from juggletrack.detect.fake import FakeDetector
from juggletrack.sim import simulate_cascade


def install_fake_yolo_model(monkeypatch, names, cuda=False, mps=False):
    """Install a fake `ultralytics.YOLO` whose instance exposes `names` (as a
    real model would) and records the `classes=` and `device=` kwargs passed to
    `predict`, returning a minimal boxes-less result so `YOLODetector.detect`
    short-circuits. Also installs a fake `torch` reporting the given CUDA and
    MPS availability.
    """
    calls = {}
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeYOLO:
        def __init__(self, model_path):
            calls["model_path"] = model_path
            self.names = names

        def predict(self, frame, **kwargs):
            calls["classes"] = kwargs.get("classes")
            calls["device"] = kwargs.get("device")
            return [types.SimpleNamespace(boxes=None)]

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    return calls


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


def test_resolve_ball_classes_coco():
    from juggletrack.detect.yolo import resolve_ball_classes

    coco_like = {0: "person", 32: "sports ball", 33: "kite"}
    assert resolve_ball_classes(coco_like) == (32,)


def test_resolve_ball_classes_finetuned_single_class():
    from juggletrack.detect.yolo import resolve_ball_classes

    assert resolve_ball_classes({0: "ball"}) == (0,)


def test_resolve_ball_classes_unknown_names_means_no_filter():
    from juggletrack.detect.yolo import resolve_ball_classes

    assert resolve_ball_classes({0: "beanbag"}) is None
    assert resolve_ball_classes({}) is None


def test_yolodetector_resolves_classes_from_model_names(monkeypatch):
    """Constructor wiring: classes should come from the loaded model's own
    `names`, not a hardcoded COCO id — a regression guard for the class-id
    bug fixed alongside resolve_ball_classes.
    """
    from juggletrack.detect.yolo import YOLODetector

    calls = install_fake_yolo_model(monkeypatch, {0: "ball"})
    det = YOLODetector(model_path="fake.pt")
    assert det.classes == (0,)

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    det.detect(frame, 0, 0.0)
    assert calls["classes"] == [0]


def test_yolodetector_passes_none_when_unresolvable(monkeypatch):
    from juggletrack.detect.yolo import YOLODetector

    calls = install_fake_yolo_model(monkeypatch, {0: "cat", 1: "dog"})
    det = YOLODetector(model_path="fake.pt")
    assert det.classes is None

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    det.detect(frame, 0, 0.0)
    assert calls["classes"] is None


def predicted_device(monkeypatch, device=None, cuda=False, mps=False):
    from juggletrack.detect.yolo import YOLODetector

    calls = install_fake_yolo_model(monkeypatch, {0: "ball"}, cuda=cuda, mps=mps)
    det = YOLODetector(model_path="fake.pt", device=device)
    det.detect(np.zeros((64, 64, 3), dtype=np.uint8), 0, 0.0)
    return calls["device"]


def test_yolodetector_defaults_to_mps_when_only_mps_is_available(monkeypatch):
    assert predicted_device(monkeypatch, mps=True) == "mps"


def test_yolodetector_leaves_device_to_ultralytics_when_cuda_is_available(monkeypatch):
    assert predicted_device(monkeypatch, cuda=True, mps=True) is None


def test_yolodetector_leaves_device_unset_without_a_gpu(monkeypatch):
    assert predicted_device(monkeypatch) is None


def test_yolodetector_explicit_device_wins_over_mps(monkeypatch):
    assert predicted_device(monkeypatch, device="cpu", mps=True) == "cpu"
