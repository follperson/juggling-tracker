import numpy as np
import pytest

from juggletrack.analyze import analyze_detections
from juggletrack.detect.fake import FakeDetector
from juggletrack.pipeline.live import _hud_lines, run_live
from juggletrack.pipeline.realtime import RealtimeState
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
    """T5: the original `f[:30].sum() > 0` check was satisfiable by
    draw_hand_line alone -- hand_line_y defaults to 0.0 before any arcs are
    recognized, which draws a full-width line at row 0 (inside the checked
    band) regardless of whether draw_hud does anything at all. Only check
    frames where hand_line_y is comfortably below the HUD's own row band
    (0.2 * 240 = 48, far above the checked rows 0-29), so a nonzero top
    band can only be attributed to the HUD text."""
    sim, video = sim_video
    captured = []
    run_live(str(video), FakeDetector(sim.detections), display=False,
             max_frames=40,
             on_state=lambda s, f: captured.append((s.hand_line_y, f.copy())))
    assert captured and all(isinstance(f, np.ndarray) for _, f in captured)
    hud_candidates = [f for hl, f in captured if hl > 0.2]
    assert hud_candidates, "no captured frame had a recognized (non-zero-row) hand line"
    assert any(f[:30].sum() > 0 for f in hud_candidates)


def test_run_live_bad_source_raises():
    from juggletrack.detect.fake import FakeDetector

    with pytest.raises(ValueError):
        run_live("/nonexistent/video.mp4", FakeDetector([]), display=False)


def test_hud_lines_idle_shows_totals_not_stale_current_run():
    """S2: idle (no run active) must branch like overlay.py's idle HUD
    ('runs N  catches TOTAL  drops D'), not show a stale 'run N+1 catches 0'
    -- the old single-branch HUD always displayed the "run N" form, so a
    just-closed run momentarily showed catches_current_run=0 (already
    reset) attached to a run label, misleadingly implying zero catches for
    a run that just had some."""
    idle = RealtimeState(t=1.0, catches_total=12, throws_total=12, drops_total=1,
                          runs_completed=2, run_active=False, catches_current_run=0)
    lines = _hud_lines(idle, run_history=[], fps=30.0)
    assert lines == ["runs 2  catches 12  drops 1  fps 30"]


def test_hud_lines_active_shows_current_run_progress():
    active = RealtimeState(t=1.0, catches_total=17, throws_total=18, drops_total=1,
                            runs_completed=2, run_active=True, catches_current_run=5)
    lines = _hud_lines(active, run_history=[], fps=30.0)
    assert lines == ["run 3  catches 5  drops 1  fps 30"]


def test_hud_lines_includes_run_history_when_present():
    """S2: spec §8's 'run history' overlay element -- a compact second line
    of the last few completed runs' catch counts, capped at 5 so a long
    session doesn't grow the line unboundedly."""
    idle = RealtimeState(t=1.0, runs_completed=7, run_active=False)
    lines = _hud_lines(idle, run_history=[3, 12, 8, 9, 15, 4, 10], fps=30.0)
    assert len(lines) == 2
    assert lines[1] == "history: 8, 9, 15, 4, 10"


def test_run_live_hud_run_history_tracks_completed_runs(sim_video):
    """S2 integration: live.py's own run-history bookkeeping (comparing
    consecutive states' runs_completed/catches_current_run) records the
    right per-run catch count as runs close during a real session."""
    sim, video = sim_video
    states = []
    final, _ = run_live(str(video), FakeDetector(sim.detections), display=False,
                         on_state=lambda s, f: states.append(s))
    # single 12-throw cascade -> exactly one run closes, with 12 catches.
    assert final.runs_completed == 1
    assert final.catches_total == 12


def test_run_live_file_source_pts_seam_rebases_like_videoreader(sim_video, monkeypatch):
    """S1: file-source timestamps in run_live must apply the SAME PTS
    health rule as VideoReader.frames() (video/reader.py) -- healthy PTS
    for a while, then a common all-zero "unsupported" sentinel mid-stream,
    must rebase off the last known-good (t, idx) pair rather than jump to
    the absolute idx/fps clock (which can go backward relative to where
    PTS had already drifted, breaking downstream parabola fits). Mirrors
    tests/test_video_reader.py::
    test_frame_timestamps_rebase_at_pts_seam_stay_monotonic's fixture and
    reasoning, applied to the live loop instead of VideoReader."""
    import cv2

    sim, video = sim_video

    class _HealthyThenZeroPtsCap:
        """Reports real-seeming (24fps-rate) PTS for the first 10 frames,
        then the common all-zero sentinel from frame 10 on."""

        def __init__(self, real):
            self._real = real
            self._idx = 0

        def read(self):
            ok, frame = self._real.read()
            if ok:
                self._idx += 1
            return ok, frame

        def get(self, prop_id):
            if prop_id == cv2.CAP_PROP_POS_MSEC:
                if self._idx - 1 < 10:
                    return (self._idx - 1) * (1000.0 / 24.0)
                return 0.0
            return self._real.get(prop_id)

        def __getattr__(self, name):
            return getattr(self._real, name)

    real_ctor = cv2.VideoCapture
    monkeypatch.setattr(cv2, "VideoCapture", lambda src: _HealthyThenZeroPtsCap(real_ctor(src)))

    ts = []
    run_live(str(video), FakeDetector(sim.detections), display=False,
             on_state=lambda s, f: ts.append(s.t))

    assert all(b > a for a, b in zip(ts, ts[1:])), (
        "timestamps must stay strictly monotonic across the PTS seam"
    )
    # The seam itself must continue from the last known-good PTS (0.375 at
    # idx=9), not jump back to the absolute idx/fps clock (10/30=0.3333,
    # which would be a backward jump).
    assert ts[10] == pytest.approx(ts[9] + 1 / 30.0)
    post_seam = ts[10:]
    deltas = [b - a for a, b in zip(post_seam, post_seam[1:])]
    assert all(d == pytest.approx(1 / 30.0) for d in deltas)


def test_run_live_warmup_latency_does_not_drag_down_mean_fps(sim_video, monkeypatch):
    """S3: mean_processing_fps must be warmup-corrected -- a slow FIRST
    detect() call (simulating cap-open/detector-construction latency
    bleeding into the loop) must not drag down the returned mean the way a
    naive from-the-start (idx+1)/elapsed calculation would."""
    import time as time_module

    sim, video = sim_video

    class _SlowFirstFrameDetector:
        def __init__(self, real):
            self._real = real
            self._first = True

        def detect(self, frame, idx, t):
            if self._first:
                self._first = False
                time_module.sleep(0.2)
            return self._real.detect(frame, idx, t)

    final, mean_fps = run_live(
        str(video), _SlowFirstFrameDetector(FakeDetector(sim.detections)),
        display=False, max_frames=30,
    )
    # A naive (idx+1)/elapsed-since-loop-start mean over ~30 fast frames
    # plus one 0.2s outlier would land in the double digits at best; the
    # warmup-corrected mean (excluding that one interval) should be far
    # higher on any machine capable of running this test suite.
    assert mean_fps > 100, f"mean_fps={mean_fps:.1f} -- warmup latency was not excluded"


def test_run_live_file_source_warns_on_short_read(sim_video, monkeypatch, capsys):
    """S6: a file source that ends well short of its own reported frame
    count likely hit a decode problem partway through, not a clean
    end-of-stream -- warn to stderr rather than staying silent."""
    import cv2

    sim, video = sim_video

    class _InflatedFrameCountCap:
        def __init__(self, real):
            self._real = real

        def get(self, prop_id):
            if prop_id == cv2.CAP_PROP_FRAME_COUNT:
                return self._real.get(prop_id) * 10
            return self._real.get(prop_id)

        def __getattr__(self, name):
            return getattr(self._real, name)

    real_ctor = cv2.VideoCapture
    monkeypatch.setattr(cv2, "VideoCapture", lambda src: _InflatedFrameCountCap(real_ctor(src)))

    run_live(str(video), FakeDetector(sim.detections), display=False)
    err = capsys.readouterr().err
    assert "stream may have ended early" in err


def test_run_live_max_frames_stop_does_not_warn(sim_video, monkeypatch, capsys):
    """Contrast with the above: an intentional --max-frames early stop is
    NOT a short-read problem and must not warn."""
    sim, video = sim_video
    run_live(str(video), FakeDetector(sim.detections), display=False, max_frames=10)
    err = capsys.readouterr().err
    assert "stream may have ended early" not in err


def test_run_live_webcam_source_retries_transient_read_failures(monkeypatch):
    """S6: a webcam source gets a bounded retry (not an immediate
    end-of-stream conclusion) on a transient read() failure -- e.g. a
    single dropped USB frame -- so the session doesn't end after one
    hiccup. The proxy below never touches real hardware (source=0 is only
    ever handed to the monkeypatched cv2.VideoCapture constructor)."""
    import cv2

    class _FlakyWebcamCap:
        def __init__(self):
            self._frame = np.zeros((240, 320, 3), dtype=np.uint8)
            self._reads = 0

        def isOpened(self):
            return True

        def read(self):
            self._reads += 1
            if self._reads in (2, 3):  # transient failures, still recoverable
                return False, None
            if self._reads > 25:
                return False, None  # the "camera" is gone for good
            return True, self._frame.copy()

        def get(self, prop_id):
            return 30.0

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda src: _FlakyWebcamCap())
    states = []
    run_live(0, FakeDetector([]), display=False, on_state=lambda s, f: states.append(s))
    assert len(states) > 15, (
        f"only {len(states)} frames processed -- the transient failures at "
        "reads 2-3 should have been retried, not treated as end-of-stream"
    )
