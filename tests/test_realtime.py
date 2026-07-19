import sys
import subprocess
from collections import defaultdict

from juggletrack.analyze import analyze_detections
from juggletrack.pipeline.realtime import RealtimeAnalyzer, RealtimeConfig
from juggletrack.sim import simulate_cascade


def frames_of(dets):
    by = defaultdict(list)
    for d in dets:
        by[d.frame_idx].append(d)
    return [(idx / 30.0, by.get(idx, [])) for idx in range(max(by) + 1)]


def stream(r, analyzer):
    states = []
    for t, dets in frames_of(r.detections):
        states.append(analyzer.feed(dets, t))
    states.append(analyzer.finalize())
    return states


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


def test_realtime_module_is_cv2_free():
    code = ("import sys; import juggletrack.pipeline.realtime; "
            "sys.exit(1 if 'cv2' in sys.modules else 0)")
    proc = subprocess.run([sys.executable, "-c", code], cwd="src")
    assert proc.returncode == 0, "importing realtime pulled in cv2"
