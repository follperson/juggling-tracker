import pytest

from juggletrack.arcs.extract import extract_arcs
from juggletrack.events.catches import derive_events
from juggletrack.events.handline import estimate_hand_line
from juggletrack.events.runs import estimate_period, segment_runs
from juggletrack.sim import simulate_cascade
from juggletrack.types import Detection


def pipeline(dets):
    arcs = extract_arcs(dets)
    hl = estimate_hand_line(arcs)
    throws, catches = derive_events(arcs, hl)
    return arcs, hl, throws, catches, segment_runs(arcs, throws, catches, hl)


def shift(dets, dt_s, dframes):
    return [
        Detection(frame_idx=d.frame_idx + dframes, t=d.t + dt_s, x=d.x, y=d.y,
                  w=d.w, h=d.h, confidence=d.confidence)
        for d in dets
    ]


def test_single_clean_run():
    r = simulate_cascade(n_throws=12, fps=30.0, seed=1)
    _, _, throws, _, runs = pipeline(r.detections)
    assert len(runs) == 1
    run = runs[0]
    assert run.throws == 12 and run.catches == 12
    assert run.start_t == pytest.approx(r.run_start, abs=0.05)
    assert run.end_t == pytest.approx(r.run_end, abs=0.05)
    assert run.period_s == pytest.approx(r.params.period_s, abs=0.03)
    assert run.quality > 0.5
    assert run.end_reason == "stop"
    assert estimate_period(throws) == pytest.approx(0.45, abs=0.02)


def test_two_separated_runs():
    a = simulate_cascade(n_throws=10, fps=30.0, seed=1)
    b = simulate_cascade(n_throws=8, fps=30.0, seed=2)
    dets = a.detections + shift(b.detections, 12.0, 360)
    _, _, _, _, runs = pipeline(dets)
    assert len(runs) == 2
    assert runs[0].throws == 10 and runs[1].throws == 8
    assert runs[1].start_t == pytest.approx(12.0 + b.run_start, abs=0.05)


def test_drop_run_ends_at_missed_catch():
    r = simulate_cascade(n_throws=20, fps=30.0, drop_at_throw=8, seed=1)
    _, _, _, _, runs = pipeline(r.detections)
    assert len(runs) == 1
    assert runs[0].end_t == pytest.approx(r.missed_catch_t, abs=0.1)
    assert runs[0].catches == len(r.catch_times)


def test_min_arcs_filters_stray_tosses():
    r = simulate_cascade(n_throws=2, fps=30.0, seed=3)
    _, _, _, _, runs = pipeline(r.detections)
    assert runs == []
