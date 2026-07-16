import pytest

from juggletrack.analyze import analyze_detections
from juggletrack.pipeline.overlay import render_overlay
from juggletrack.sim import simulate_cascade
from juggletrack.video.reader import VideoReader
from tests.helpers import write_test_video


@pytest.fixture()
def setup(tmp_path):
    sim = simulate_cascade(n_throws=10, fps=30.0, drop_at_throw=6, seed=1)
    n_frames = max(d.frame_idx for d in sim.detections) + 1
    video = tmp_path / "in.mp4"
    write_test_video(video, n_frames=n_frames, fps=30.0)
    session = analyze_detections(sim.detections)
    return sim, video, session, n_frames


def test_overlay_writes_decodable_video(tmp_path, setup):
    sim, video, session, n_frames = setup
    out = tmp_path / "overlay.mp4"
    render_overlay(video, session, out, detections=sim.detections)
    assert out.exists() and out.stat().st_size > 0
    with VideoReader(out) as reader:
        assert reader.info.frame_count == n_frames
        assert (reader.info.width, reader.info.height) == (320, 240)
        # decoding must actually work, not just headers
        assert sum(1 for _ in reader.frames()) == n_frames


def test_overlay_draws_on_frames(tmp_path, setup):
    """Overlay frames must differ from the input (something was drawn)."""
    import numpy as np

    sim, video, session, n_frames = setup
    out = tmp_path / "overlay.mp4"
    render_overlay(video, session, out, detections=sim.detections)
    with VideoReader(video) as a, VideoReader(out) as b:
        fa = next(iter(a.frames()))[2]
        fb = next(iter(b.frames()))[2]
    assert np.abs(fa.astype(int) - fb.astype(int)).sum() > 0


def test_overlay_without_detections(tmp_path, setup):
    _, video, session, _ = setup
    out = tmp_path / "overlay2.mp4"
    render_overlay(video, session, out)
    assert out.exists() and out.stat().st_size > 0
