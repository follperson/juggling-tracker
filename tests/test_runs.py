import pytest

from juggletrack.arcs.extract import extract_arcs
from juggletrack.events.catches import derive_events
from juggletrack.events.handline import estimate_hand_line
from juggletrack.events.periodicity import periodicity_score
from juggletrack.events.runs import estimate_period, segment_runs
from juggletrack.sim import CascadeParams, simulate_cascade
from juggletrack.types import Arc, CatchEvent, Detection, ThrowEvent


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
    # Threshold 0.5 -> 0.4: quality is now scored over the adaptive
    # [0.5p, 1.5p] lag band instead of the fixed (0.2, 1.2) band (turn-4
    # periodicity fix), which legitimately shifts the exact number (0.484
    # here) without weakening the pin -- a clean cascade must still score
    # clearly, unambiguously periodic (junk cohorts measure ~0.1-0.15).
    assert run.quality > 0.4
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


def test_quality_scored_over_arc_span_not_truncated_end_t():
    """A single missed-catch DETECTION (not a real drop) truncates the
    reported end_t to that miss's crossing time, but the group's arcs can
    keep going well past it if the pattern actually continued -- lane 1
    measured a real case (af1 run 2) scoring 0.004 on [start_t, end_t] vs
    0.351 on the arcs' own span. Reproduce the mechanism directly: take a
    full, cleanly periodic 16-throw run and strip the catch event for one
    arc in the middle (simulating an extraction miss) without touching the
    arcs themselves. end_t truncates hard; quality must still reflect the
    real periodic structure across the whole arc span.
    """
    r = simulate_cascade(n_throws=16, fps=30.0, seed=1)
    arcs = extract_arcs(r.detections)
    hl = estimate_hand_line(arcs)
    throws, catches = derive_events(arcs, hl)

    arcs_sorted = sorted(arcs, key=lambda a: a.t_start)
    miss_id = arcs_sorted[3].id
    catches = [c for c in catches if c.arc_id != miss_id]

    runs = segment_runs(arcs, throws, catches, hl)
    assert len(runs) == 1
    run = runs[0]
    # end_t truncated to (approximately) the 4th throw's crossing -- far
    # short of the group's real arc span (16 throws over ~7s).
    assert run.end_t < arcs_sorted[-1].t_end - 2.0
    assert run.quality > 0.3


def test_adaptive_lag_band_scores_meschke_style_slow_period():
    """The in-repo default lag band (0.2, 1.2) can miss a run's true period
    entirely on differently-framed footage (Meschke ground truth: period
    1.19-2.89s, wholly outside (0.2, 1.2)). segment_runs must search around
    the run's OWN estimated period -- [max(0.1, 0.5p), 1.5p] -- not a fixed
    band tuned to one framing. Reproduce with a slow-period simulated
    cascade whose true period sits outside the fixed default band.
    """
    slow = CascadeParams(period_s=1.5, dwell_s=0.9, g=0.25)
    r = simulate_cascade(n_throws=12, fps=30.0, seed=1, params=slow)
    _, _, _, _, runs = pipeline(r.detections)
    assert len(runs) == 1
    run = runs[0]
    assert run.period_s == pytest.approx(1.5, abs=0.05)
    # Confirm this period truly sits outside the fixed default band -- the
    # regression this test pins would be invisible if it didn't.
    assert not (0.2 <= run.period_s <= 1.2)
    assert run.quality > 0.3


def test_unjudgeable_quality_stores_zero():
    """periodicity_score returns None when a window can't be judged (spec
    change: never zero-pad a short window into a fake score). Run.quality is
    a plain float field, so segment_runs must collapse that "can't judge"
    into 0.0 -- distinct in meaning from a judged score of 0.0, but the same
    stored value (documented on Run.quality).

    Hand-built fixture (two throws only, so the run's own period is unknown
    -- len(throws) < 3 -- and quality falls back to the fixed default band
    (0.2, 1.2)): a real 0.30s arc span makes max_lag = min(1.2, 0.15) = 0.15,
    below the default band's lag_lo of 0.2, so the band is empty and the
    score cannot be judged.
    """
    hand_line = 0.65
    arcs = [
        Arc(id=0, t_start=0.0, t_end=0.15, ay=1.0, by=-0.15, cy=hand_line,
            bx=0.1, cx=0.4, n_points=6, rmse=0.001),
        Arc(id=1, t_start=0.15, t_end=0.30, ay=1.0, by=-0.15, cy=hand_line,
            bx=-0.1, cx=0.5, n_points=6, rmse=0.001),
    ]
    throws = [ThrowEvent(t=0.0, x=0.4, arc_id=0), ThrowEvent(t=0.15, x=0.5, arc_id=1)]
    catches = [CatchEvent(t=0.15, x=0.4, arc_id=0), CatchEvent(t=0.30, x=0.5, arc_id=1)]

    runs = segment_runs(arcs, throws, catches, hand_line, min_arcs=2)

    assert len(runs) == 1
    assert runs[0].period_s is None  # <3 throws: no per-run period estimate

    # Confirm this fixture genuinely hits the "band empty" case at the
    # periodicity_score level (not just asserting the collapsed output).
    raw_score, _ = periodicity_score(arcs, runs[0].start_t, runs[0].end_t)
    assert raw_score is None
    assert runs[0].quality == 0.0
