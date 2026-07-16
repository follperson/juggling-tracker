import pytest

from juggletrack.arcs.extract import extract_arcs
from juggletrack.events.catches import derive_events
from juggletrack.events.drops import detect_drops
from juggletrack.events.handline import estimate_hand_line
from juggletrack.events.runs import segment_runs
from juggletrack.sim import simulate_cascade


def pipeline(dets):
    arcs = extract_arcs(dets)
    hl = estimate_hand_line(arcs)
    throws, catches = derive_events(arcs, hl)
    runs = segment_runs(arcs, throws, catches, hl)
    drops, runs = detect_drops(arcs, runs, hl)
    return arcs, hl, drops, runs


def test_drop_detected_with_signals():
    r = simulate_cascade(n_throws=20, fps=30.0, drop_at_throw=8, seed=1)
    _, _, drops, runs = pipeline(r.detections)
    assert len(drops) == 1
    d = drops[0]
    assert len(d.signals) >= 2
    assert "floor_descent" in d.signals
    assert d.t == pytest.approx(r.missed_catch_t, abs=0.1)
    assert runs[0].end_reason == "drop"


def test_clean_stop_is_not_a_drop():
    r = simulate_cascade(n_throws=12, fps=30.0, seed=1)
    _, _, drops, runs = pipeline(r.detections)
    assert drops == []
    assert runs[0].end_reason == "stop"


def test_noisy_clean_run_no_false_drops():
    r = simulate_cascade(n_throws=14, fps=30.0, noise=0.004, dropout=0.15, seed=6)
    _, _, drops, runs = pipeline(r.detections)
    assert drops == []
    assert runs and runs[0].end_reason == "stop"
