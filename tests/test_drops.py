import pytest

from juggletrack.arcs.extract import extract_arcs
from juggletrack.events.catches import derive_events, hand_line_crossings
from juggletrack.events.drops import detect_drops
from juggletrack.events.handline import estimate_hand_line
from juggletrack.events.runs import segment_runs
from juggletrack.sim import simulate_cascade
from juggletrack.types import Arc, Run


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


def _make_floor_candidate(arc_id: int, t_start: float) -> Arc:
    """A hand-crafted flight arc thrown from hand_line=0.65 with g=2.0 (ay=1.0).

    It would normally return to the hand line at dt=1.1 (a catch), but its
    fitted window is extended to dt=1.31 so it keeps falling past the hand
    line toward the floor: y_at(t_end) ~= 0.925 (well past hand_line +
    floor_margin=0.77) with positive (downward) terminal velocity. This is
    exactly the shape that makes floor_descent fire in detect_drops.
    """
    return Arc(
        id=arc_id, t_start=t_start, t_end=t_start + 1.31,
        ay=1.0, by=-1.1, cy=0.65, bx=0.0, cx=0.5,
        n_points=10, rmse=0.01,
    )


def test_continuing_pattern_suppresses_collapse_signal():
    """floor_descent alone (1 of 3 signals) must not become a drop when the
    juggler actually kept the pattern going: a genuine new throw starting
    ~1 period after the candidate's falling hand-line crossing suppresses
    periodicity_collapse, so len(signals) stays at 1 and no DropEvent fires.
    """
    hand_line = 0.65
    period = 0.45
    cand = _make_floor_candidate(1, t_start=10.0)
    _, miss_t = hand_line_crossings(cand, hand_line)

    # A normal juggling arc (rises well above hand_line, so `stays_low` is
    # false and it can never be mistaken for a bounce) that starts shortly
    # after the miss and is caught cleanly -- the pattern continuing.
    continuing = Arc(
        id=2, t_start=miss_t + period, t_end=miss_t + period + 1.1,
        ay=1.0, by=-1.1, cy=0.65, bx=0.0, cx=0.55,
        n_points=10, rmse=0.01,
    )
    run = Run(
        start_t=cand.t_start, end_t=miss_t, catches=1, throws=2,
        arc_ids=[cand.id, continuing.id], period_s=period, end_reason="stop",
    )

    drops, runs = detect_drops([cand, continuing], [run], hand_line)

    assert drops == []
    assert runs[0].end_reason == "stop"


def test_dead_pattern_fires_collapse_signal():
    """Same floor candidate as above, but with no arc starting after the
    miss: the pattern died, so periodicity_collapse joins floor_descent and
    a DropEvent fires with both signals.
    """
    hand_line = 0.65
    period = 0.45
    cand = _make_floor_candidate(1, t_start=10.0)
    _, miss_t = hand_line_crossings(cand, hand_line)

    run = Run(
        start_t=cand.t_start, end_t=miss_t, catches=0, throws=1,
        arc_ids=[cand.id], period_s=period, end_reason="stop",
    )

    drops, runs = detect_drops([cand], [run], hand_line)

    assert len(drops) == 1
    d = drops[0]
    assert d.arc_id == cand.id
    assert d.t == pytest.approx(miss_t)
    assert "floor_descent" in d.signals
    assert "periodicity_collapse" in d.signals
    assert runs[0].end_reason == "drop"


def test_pre_miss_arc_does_not_suppress_collapse():
    """periodicity_collapse must key off *when* the other arc starts, not
    merely whether one exists in the run. A ball already in flight before
    the miss (thrown pre-drop, landing shortly after due to reaction
    latency) is an artifact of the drop itself -- its mere presence must
    not suppress periodicity_collapse. A mutant that suppresses on
    `any(a.id != cand.id for a in run_arcs)` (ignoring timing entirely)
    would pass this arc through and wrongly cancel the signal.
    """
    hand_line = 0.65
    period = 0.45
    cand = _make_floor_candidate(1, t_start=10.0)
    _, miss_t = hand_line_crossings(cand, hand_line)

    # A normal juggling arc (rises well above hand_line, caught cleanly at
    # the hand line so it is not itself a floor candidate) that was already
    # airborne 0.45s *before* the candidate's miss -- i.e. its t_start
    # precedes miss_t, so it falls outside the (miss_t, miss_t +
    # collapse_factor*period] window the real check requires.
    pre_miss = Arc(
        id=2, t_start=miss_t - 0.45, t_end=miss_t - 0.45 + 1.1,
        ay=1.0, by=-1.1, cy=0.65, bx=0.0, cx=0.55,
        n_points=10, rmse=0.01,
    )
    run = Run(
        start_t=cand.t_start, end_t=miss_t, catches=1, throws=2,
        arc_ids=[cand.id, pre_miss.id], period_s=period, end_reason="stop",
    )

    drops, runs = detect_drops([cand, pre_miss], [run], hand_line)

    assert len(drops) == 1
    d = drops[0]
    assert d.arc_id == cand.id
    assert "floor_descent" in d.signals
    assert "periodicity_collapse" in d.signals
    assert runs[0].end_reason == "drop"


def test_late_arc_outside_window_does_not_suppress_collapse():
    """Same idea as the pre-miss case, but for the other edge of the
    window: an arc starting *after* miss_t + collapse_factor*period has
    closed (default collapse_factor=1.5, period=0.45 -> window closes at
    miss_t + 0.675) is too late to count as the pattern continuing, so it
    must not suppress periodicity_collapse either. Only an arc starting
    strictly inside (miss_t, miss_t + collapse_factor*period] is evidence
    the juggler kept going.
    """
    hand_line = 0.65
    period = 0.45
    cand = _make_floor_candidate(1, t_start=10.0)
    _, miss_t = hand_line_crossings(cand, hand_line)

    late = Arc(
        id=2, t_start=miss_t + 0.9, t_end=miss_t + 0.9 + 1.1,
        ay=1.0, by=-1.1, cy=0.65, bx=0.0, cx=0.55,
        n_points=10, rmse=0.01,
    )
    run = Run(
        start_t=cand.t_start, end_t=miss_t, catches=1, throws=2,
        arc_ids=[cand.id, late.id], period_s=period, end_reason="stop",
    )

    drops, runs = detect_drops([cand, late], [run], hand_line)

    assert len(drops) == 1
    d = drops[0]
    assert d.arc_id == cand.id
    assert "floor_descent" in d.signals
    assert "periodicity_collapse" in d.signals
    assert runs[0].end_reason == "drop"
