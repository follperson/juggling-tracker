import pytest

from juggletrack.analyze import AnalyzeConfig, analyze_detections
from juggletrack.arcs.extract import extract_arcs
from juggletrack.events.catches import derive_events
from juggletrack.events.drops import is_floor_bound
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


def test_unwitnessed_miss_does_not_truncate_run_span():
    """Plan-5 task 3 (carried forward from plan 3), measured at scale
    (ss42_id_011): one unwitnessed catch mid-run truncated end_t to 38.7s
    while the run's 95 arcs (and its 93 counted catches) span 0..200s. An
    uncaught arc that is NOT floor-bound (i.e. an extraction miss / occlusion,
    not a real drop) must not truncate the span.

    Reproduces the mechanism directly: take a full, cleanly periodic 12-throw
    run and erase every detection belonging to ONE mid-run flight's second
    half (apex through its would-be catch), mirroring the windowed-dropout
    construction in tests/test_realtime.py::
    test_debounce_absorbs_genuine_mid_run_detection_gap. Matched against the
    flight's own analytic trajectory (not just a bare time window) so only
    that one ball's points are removed -- other balls airborne at the same
    moment are untouched. The truncated arc still gets extracted (from its
    surviving ascending half) and still yields a throw, but its fitted
    falling crossing only exists by extrapolating far past its last real
    detection, so derive_events correctly withholds a CatchEvent (see
    catches.py's witnessed-catch gate) -- exactly an "unwitnessed miss", not
    a drop. It ends near its own apex (well above the hand line), so
    is_floor_bound is False.

    Before this fix, segment_runs treated ANY uncaught arc as first_miss and
    truncated end_t to its (extrapolated) hand-line crossing (~3.83s here)
    even though the run's real arcs run to ~6.53s -- a >40% truncation from
    a single unwitnessed miss, matching the measured ss42_id_011 mechanism.
    """
    sim = simulate_cascade(n_throws=12, fps=30.0, seed=1)
    sr = analyze_detections(sim.detections, config=AnalyzeConfig())
    assert len(sr.runs) == 1
    full = sr.runs[0]

    p = sim.params
    target_idx = 5  # a mid-run throw, well clear of both run boundaries
    t0 = sim.throw_times[target_idx]
    hand = target_idx % 2
    x0, x1 = p.hand_x(hand), p.hand_x(1 - hand)
    dur = p.flight_s

    def target_xy(t: float) -> tuple[float, float]:
        dt = t - t0
        y = p.hand_y - p.v0 * dt + 0.5 * p.g * dt * dt
        x = x0 + (x1 - x0) * dt / dur
        return x, y

    # Second half of the flight (apex through the would-be catch, plus a
    # small margin past landing) -- erasing only this ball's points there,
    # not the whole time window, so other simultaneously-airborne balls'
    # arcs are untouched.
    gap_start, gap_end = t0 + dur * 0.5, t0 + dur + 0.05
    gapped = [
        d for d in sim.detections
        if not (
            gap_start <= d.t <= gap_end
            and abs(d.x - target_xy(d.t)[0]) < 1e-6
            and abs(d.y - target_xy(d.t)[1]) < 1e-6
        )
    ]
    assert len(gapped) < len(sim.detections), "fixture must actually remove points"

    sr2 = analyze_detections(gapped, config=AnalyzeConfig())
    assert len(sr2.runs) == 1
    assert sr2.runs[0].catches == full.catches - 1  # exactly the one unwitnessed catch is lost
    assert sr2.runs[0].end_t > 0.9 * full.end_t  # span survives the miss


def test_min_arcs_filters_stray_tosses():
    r = simulate_cascade(n_throws=2, fps=30.0, seed=3)
    _, _, _, _, runs = pipeline(r.detections)
    assert runs == []


def test_quality_scored_over_arc_span_not_truncated_end_t():
    """A genuine floor-bound miss truncates the reported end_t to that
    miss's crossing time, but the group's arcs can keep going well past it
    if the pattern actually continued -- lane 1 measured a real case (af1
    run 2) scoring 0.004 on [start_t, end_t] vs 0.351 on the arcs' own span.
    Reproduce the mechanism directly: take a full, cleanly periodic 16-throw
    run and turn one mid-run arc into a floor-bound miss by extending its
    fitted t_end 0.09s further along its own parabola (past hand_line +
    FLOOR_MARGIN, still descending) and stripping its catch -- WITHOUT
    touching any other arc. end_t truncates hard; quality must still
    reflect the real periodic structure across the whole arc span.

    Plan-5 task 3 update: originally this fixture merely stripped a catch
    EVENT from an otherwise normally-landing arc (an extraction miss, not a
    floor-bound one) to trigger truncation. Task 3 changed segment_runs so
    only a floor-bound uncaught arc (`is_floor_bound`) may truncate end_t --
    an unwitnessed/extraction miss no longer does (see
    test_unwitnessed_miss_does_not_truncate_run_span). The old fixture's
    truncation premise is exactly the bug that fix removes, so this test now
    builds a genuinely floor-bound miss (measured: extending t_end by 0.09s
    lands y_at(t_end) ~= 0.736, comfortably past hand_line + FLOOR_MARGIN
    ~= 0.732, with positive terminal velocity) to keep exercising the
    quality-over-own-span mechanism this test is actually about.
    """
    r = simulate_cascade(n_throws=16, fps=30.0, seed=1)
    arcs = extract_arcs(r.detections)
    hl = estimate_hand_line(arcs)
    throws, catches = derive_events(arcs, hl)

    arcs_sorted = sorted(arcs, key=lambda a: a.t_start)
    target = arcs_sorted[3]
    mutated = target.model_copy(update={"t_end": target.t_end + 0.09})
    assert is_floor_bound(mutated, hl), "fixture must genuinely be floor-bound"
    arcs = [mutated if a.id == target.id else a for a in arcs]
    catches = [c for c in catches if c.arc_id != target.id]

    runs = segment_runs(arcs, throws, catches, hl)
    assert len(runs) == 1
    run = runs[0]
    # end_t truncated to (approximately) the floor-bound arc's own
    # hand-line crossing -- far short of the group's real arc span (16
    # throws over ~7s).
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
