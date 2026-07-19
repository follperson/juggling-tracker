import sys
import subprocess
from collections import defaultdict

from juggletrack.analyze import analyze_detections
from juggletrack.pipeline.realtime import RealtimeAnalyzer, RealtimeConfig
from juggletrack.sim import simulate_cascade
from juggletrack.types import Detection


def shift(dets, dt_s, dframes):
    return [
        Detection(frame_idx=d.frame_idx + dframes, t=d.t + dt_s, x=d.x, y=d.y,
                  w=d.w, h=d.h, confidence=d.confidence)
        for d in dets
    ]


def frames_of(dets):
    by = defaultdict(list)
    for d in dets:
        by[d.frame_idx].append(d)
    return [(idx / 30.0, by.get(idx, [])) for idx in range(max(by) + 1)]


def stream_dets(dets, analyzer):
    states = []
    for t, frame_dets in frames_of(dets):
        states.append(analyzer.feed(frame_dets, t))
    states.append(analyzer.finalize())
    return states


def stream(r, analyzer):
    return stream_dets(r.detections, analyzer)


def test_parity_short_clean_stream():
    r = simulate_cascade(n_throws=12, fps=30.0, noise=0.003, dropout=0.1, seed=1)
    offline = analyze_detections(r.detections)
    final = stream(r, RealtimeAnalyzer())[-1]
    assert final.catches_total == sum(run.catches for run in offline.runs)
    assert final.runs_completed == len(offline.runs)
    assert final.drops_total == len(offline.drops)


def test_parity_long_stream_exercises_trimming():
    r = simulate_cascade(n_throws=24, fps=30.0, noise=0.003, dropout=0.1, seed=2)
    assert r.run_end > 8.0, "fixture must outlast the analysis window"
    offline = analyze_detections(r.detections)
    final = stream(r, RealtimeAnalyzer())[-1]
    off_catches = sum(run.catches for run in offline.runs)
    assert abs(final.catches_total - off_catches) <= 1
    assert final.runs_completed == len(offline.runs)
    assert final.drops_total == len(offline.drops)


def test_parity_drop_stream():
    r = simulate_cascade(n_throws=20, fps=30.0, drop_at_throw=8, seed=1)
    offline = analyze_detections(r.detections)
    final = stream(r, RealtimeAnalyzer())[-1]
    assert final.drops_total == len(offline.drops) == 1
    off_catches = sum(run.catches for run in offline.runs)
    assert abs(final.catches_total - off_catches) <= 1


def test_counters_are_monotonic():
    r = simulate_cascade(n_throws=20, fps=30.0, drop_at_throw=8, seed=1)
    states = stream(r, RealtimeAnalyzer())
    for field in ("catches_total", "throws_total", "drops_total", "runs_completed"):
        vals = [getattr(s, field) for s in states]
        assert vals == sorted(vals), f"{field} regressed: not monotonic"


def test_confirmation_latency_bounded():
    r = simulate_cascade(n_throws=12, fps=30.0, seed=1)
    offline = analyze_detections(r.detections)
    from juggletrack.events.catches import derive_events

    _, catches = derive_events(offline.arcs, offline.hand_line_y)
    cfg = RealtimeConfig()
    analyzer = RealtimeAnalyzer(cfg)
    confirm_t: list[float] = []
    seen = 0
    for t, dets in frames_of(r.detections):
        s = analyzer.feed(dets, t)
        while seen < s.catches_total:
            confirm_t.append(t)
            seen += 1
    bound = cfg.freeze_s + cfg.cadence_frames / 30.0 + 0.25
    for ct, ev in zip(confirm_t, sorted(c.t for c in catches)[: len(confirm_t)]):
        assert ct - ev <= bound, f"catch at {ev:.2f} confirmed {ct - ev:.2f}s late"


def test_analysis_stays_fast_and_reports_timing():
    r = simulate_cascade(n_throws=24, fps=30.0, seed=3)
    states = stream(r, RealtimeAnalyzer())
    timings = [s.last_analysis_ms for s in states if s.last_analysis_ms > 0]
    assert timings, "no analysis timings recorded"
    mean_ms = sum(timings) / len(timings)
    assert mean_ms < 50, f"mean re-analysis {mean_ms:.1f}ms exceeds loose CI bound"


def _two_run_stream(gap_s):
    """Concatenate two independent 10-throw cascades with `gap_s` of silence
    between run 1's end and run 2's start (test_runs.py::shift pattern:
    absolute time-shift the second sim's detections/frame indices)."""
    a = simulate_cascade(n_throws=10, fps=30.0, noise=0.003, dropout=0.1, seed=29)
    b = simulate_cascade(n_throws=10, fps=30.0, noise=0.003, dropout=0.1, seed=30)
    dt_s = gap_s + a.run_end - b.run_start
    dframes = round(dt_s * 30.0)
    return a.detections + shift(b.detections, dt_s, dframes)


def test_two_runs_with_wide_gap_counted_separately():
    """A ~3.0s silence between two real runs sits well outside the run-close
    debounce's merge band (RUN_CLOSE_DEBOUNCE_S combined with freeze_s):
    live and offline must agree on 2 runs."""
    dets = _two_run_stream(gap_s=3.0)
    offline = analyze_detections(dets)
    assert len(offline.runs) == 2, "fixture must offline-segment into 2 runs"

    analyzer = RealtimeAnalyzer()
    final = stream_dets(dets, analyzer)[-1]
    assert final.runs_completed == 2


def test_debounce_merges_narrow_gap_runs_known_tradeoff():
    """KNOWN TRADE-OFF, not desired behavior (see RUN_CLOSE_DEBOUNCE_S's
    docstring in realtime.py): a ~1.0s silence between two real runs sits
    inside the debounce's merge band, so the live analyzer under-counts them
    as a single run even though they are genuinely separate (offline agrees
    there are 2). This pins the CURRENT behavior so a future change to the
    debounce/freeze horizons is a deliberate, visible decision -- flip this
    assertion (to == 2) when freeze_s and the run-close horizon are
    decoupled, not before.

    Pre-condition asserted inline so the pin can't rot silently: if the
    fixture ever stops offline-segmenting into 2 runs, this test's premise is
    gone and it must be revisited rather than trusted at face value.
    """
    dets = _two_run_stream(gap_s=1.0)
    offline = analyze_detections(dets)
    assert len(offline.runs) == 2, "fixture must offline-segment into 2 runs"

    analyzer = RealtimeAnalyzer()
    final = stream_dets(dets, analyzer)[-1]
    assert final.runs_completed == 1, (
        "this pins a documented trade-off (debounce merges narrow real "
        "gaps), not desired behavior"
    )


def test_realtime_module_is_cv2_free():
    code = ("import sys; import juggletrack.pipeline.realtime; "
            "sys.exit(1 if 'cv2' in sys.modules else 0)")
    proc = subprocess.run([sys.executable, "-c", code], cwd="src")
    assert proc.returncode == 0, "importing realtime pulled in cv2"
