import numpy as np
import pytest

from juggletrack.analyze import analyze_detections
from juggletrack.pipeline.overlay import render_overlay
from juggletrack.sim import simulate_cascade
from juggletrack.types import Detection
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


def _junk_region(video_path, session, dets, out, dots, junk_frame_idx, x_px, y_px):
    render_overlay(video_path, session, out, detections=dets, dots=dots)
    with VideoReader(out) as reader:
        frame = next(f for i, _, f in reader.frames() if i == junk_frame_idx)
    return frame[max(0, y_px - 6) : y_px + 6, max(0, x_px - 6) : x_px + 6]


def test_overlay_dots_verified_omits_unassigned_junk(tmp_path, setup):
    """A detection nowhere near any fitted arc must not be drawn when
    dots='verified' (the default), but must be drawn when dots='all' --
    this is what keeps junk (eye pupils etc.) out of the default view even
    though the arc gate already rejects it from counts."""
    sim, video, session, n_frames = setup
    junk_frame_idx = 5
    junk = Detection(
        frame_idx=junk_frame_idx, t=junk_frame_idx / 30.0, x=0.02, y=0.02
    )
    dets = [*sim.detections, junk]
    w, h = 320, 240
    x_px, y_px = int(0.02 * w), int(0.02 * h)

    region_all = _junk_region(
        video, session, dets, tmp_path / "all.mp4", "all", junk_frame_idx, x_px, y_px
    )
    region_verified = _junk_region(
        video, session, dets, tmp_path / "verified.mp4", "verified",
        junk_frame_idx, x_px, y_px,
    )

    assert not np.array_equal(region_all, region_verified)


def test_overlay_dots_none_skips_all_dots(tmp_path, setup):
    sim, video, session, n_frames = setup
    out = tmp_path / "none.mp4"
    render_overlay(video, session, out, detections=sim.detections, dots="none")
    assert out.exists() and out.stat().st_size > 0
    with VideoReader(out) as reader:
        assert sum(1 for _ in reader.frames()) == n_frames


def test_overlay_dots_invalid_choice_raises(tmp_path, setup):
    _, video, session, _ = setup
    out = tmp_path / "bad.mp4"
    with pytest.raises(ValueError):
        render_overlay(video, session, out, dots="bogus")
