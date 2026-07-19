import numpy as np
import pytest

from juggletrack.analyze import analyze_detections
from juggletrack.detect.fake import FakeDetector
from juggletrack.pipeline.live import run_live
from juggletrack.sim import simulate_cascade
from tests.helpers import write_test_video


@pytest.fixture()
def sim_video(tmp_path):
    sim = simulate_cascade(n_throws=12, fps=30.0, seed=1)
    n_frames = max(d.frame_idx for d in sim.detections) + 1
    video = tmp_path / "live.mp4"
    write_test_video(video, n_frames=n_frames, fps=30.0)
    return sim, video


def test_run_live_file_source_headless_parity(sim_video):
    sim, video = sim_video
    final, fps = run_live(str(video), FakeDetector(sim.detections), display=False)
    offline = analyze_detections(sim.detections)
    assert final.catches_total == sum(r.catches for r in offline.runs)
    assert final.runs_completed == len(offline.runs)
    assert fps > 0


def test_run_live_max_frames_stops_early(sim_video):
    sim, video = sim_video
    states = []
    final, _ = run_live(
        str(video), FakeDetector(sim.detections), display=False,
        max_frames=30, on_state=lambda s, f: states.append(s),
    )
    assert len(states) == 30
    assert final.t <= 30 / 30.0 + 0.05


def test_run_live_on_state_receives_annotated_frames(sim_video):
    sim, video = sim_video
    captured = []
    run_live(str(video), FakeDetector(sim.detections), display=False,
             max_frames=40, on_state=lambda s, f: captured.append(f.copy()))
    assert captured and all(isinstance(f, np.ndarray) for f in captured)
    # at least one frame after warmup must carry HUD pixels (non-black rows at top)
    assert any(f[:30].sum() > 0 for f in captured[10:])


def test_run_live_bad_source_raises():
    from juggletrack.detect.fake import FakeDetector

    with pytest.raises(ValueError):
        run_live("/nonexistent/video.mp4", FakeDetector([]), display=False)
