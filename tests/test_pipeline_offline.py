import json

import pytest

from juggletrack.analyze import analyze_detections
from juggletrack.detect.fake import FakeDetector
from juggletrack.pipeline.offline import (
    analyze_video,
    detect_video,
    load_detections_jsonl,
    save_detections_jsonl,
)
from juggletrack.sim import simulate_cascade
from juggletrack.types import SessionResult
from juggletrack.video.reader import VideoReader
from tests.helpers import write_test_video


@pytest.fixture()
def sim():
    return simulate_cascade(n_throws=12, fps=30.0, noise=0.003, dropout=0.1, seed=1)


@pytest.fixture()
def video_path(tmp_path, sim):
    n_frames = max(d.frame_idx for d in sim.detections) + 1
    path = tmp_path / "cascade.mp4"
    write_test_video(path, n_frames=n_frames, fps=30.0)
    return path


def test_jsonl_roundtrip(tmp_path, sim):
    path = tmp_path / "dets.jsonl"
    save_detections_jsonl(sim.detections, path)
    assert load_detections_jsonl(path) == sim.detections


def test_detect_video_replays_fake_detections(video_path, sim):
    with VideoReader(video_path) as reader:
        dets = detect_video(reader, FakeDetector(sim.detections))
    assert dets == sorted(sim.detections, key=lambda d: (d.frame_idx,))


def test_detect_video_stride_skips_frames(video_path, sim):
    with VideoReader(video_path) as reader:
        dets = detect_video(reader, FakeDetector(sim.detections), stride=2)
    assert dets and all(d.frame_idx % 2 == 0 for d in dets)


def test_analyze_video_end_to_end(tmp_path, video_path, sim):
    out = tmp_path / "out"
    sr = analyze_video(
        video_path, FakeDetector(sim.detections),
        out_dir=out, save_intermediates=True,
    )
    # same event results as feeding detections straight to the core
    direct = analyze_detections(sim.detections)
    assert [r.catches for r in sr.runs] == [r.catches for r in direct.runs]
    assert sr.runs[0].catches == 12
    assert sr.drops == direct.drops
    # meta enrichment
    assert sr.meta["fps"] == pytest.approx(30.0, rel=0.01)
    assert sr.meta["video_path"] == str(video_path)
    assert sr.meta["stride"] == 1
    # artifacts on disk
    written = SessionResult.model_validate(
        json.loads((out / "analysis.json").read_text())
    )
    assert written == sr
    assert load_detections_jsonl(out / "detections.jsonl") == sorted(
        sim.detections, key=lambda d: (d.frame_idx,)
    )
