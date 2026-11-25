import pytest

from juggletrack.analyze import AnalyzeConfig, analyze_detections
from juggletrack.arcs.extract import extract_arcs
from juggletrack.events.catches import derive_events, hand_line_crossings
from juggletrack.events.drops import detect_drops, is_floor_bound
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
    arcs, hl, _, _, runs = pipeline(r.detections)
    _, runs = detect_drops(arcs, runs, hl)
    assert len(runs) == 1
    assert runs[0].end_t == pytest.approx(r.missed_catch_t, abs=0.1)
    assert runs[0].end_reason == "drop"
    assert runs[0].catches == len(r.catch_times)


def test_drop_floor_margin_decides_where_a_low_last_throw_ends_the_run():
    """floor_margin is a detect_drops keyword, and through it alone decides
    whether an uncaught last throw is a drop that ends its run. This used to
    be a segment_runs keyword as well, because segment_runs cut runs at
    floor-bound misses itself and the two margins had to be kept in step;
    now only detect_drops ends a run, so there is one margin to tune.

    Constructed directly at the Arc/event level (no sim) so the exact
    threshold crossing is unambiguous: four arcs at hand_line=0.6 with
    ay=0, so `hand_line_crossings` returns None and each arc ends at its
    own t_end. Three are caught at the hand line. The last-thrown arc is
    uncaught and still descending (by=0.2) when it ends at y=0.68 -- below
    `hand_line + 0.05` (0.65) but above `hand_line + FLOOR_MARGIN` (0.70) --
    and no throw follows it. Arc 2 was thrown before it and lands after it.
    At the default margin the arc is not a drop candidate, so the run ends
    where its last arc lands (1.9). At floor_margin=0.05 it is a drop
    (floor_descent and periodicity_collapse), and the run ends there (1.4).
    """
    from juggletrack.types import CatchEvent, ThrowEvent

    hand_line = 0.6

    def mk(id, t_start, t_end, cy, by=0.0):
        return Arc(id=id, t_start=t_start, t_end=t_end, ay=0.0, by=by, cy=cy,
                   bx=0.0, cx=0.5, n_points=10, rmse=0.005)

    arcs = [
        mk(0, 0.0, 0.4, 0.60),
        mk(1, 0.5, 0.9, 0.60),
        mk(2, 0.9, 1.9, 0.60),  # in flight across the miss, caught after it
        mk(3, 1.0, 1.4, 0.60, by=0.2),  # the candidate miss: uncaught, last throw
    ]
    throws = [ThrowEvent(t=a.t_start, x=0.5, arc_id=a.id) for a in arcs]
    catches = [
        CatchEvent(t=a.t_end, x=0.5, arc_id=a.id) for a in arcs if a.id != 3
    ]
    runs = segment_runs(arcs, throws, catches, hand_line)

    drops, default_runs = detect_drops(arcs, runs, hand_line)
    assert drops == []
    assert len(default_runs) == 1
    assert default_runs[0].end_t == pytest.approx(1.9)
    assert default_runs[0].end_reason == "stop"

    drops, tight_runs = detect_drops(arcs, runs, hand_line, floor_margin=0.05)
    assert [d.arc_id for d in drops] == [3]
    assert len(tight_runs) == 1
    assert tight_runs[0].end_t == pytest.approx(1.4)
    assert tight_runs[0].end_reason == "drop"


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
    """A run that ends at a drop can still hold arcs past it: when the
    juggler picks the pattern back up inside the run gap, those throws stay
    in the run's group but fall after end_t. Lane 1 measured a real case
    (af1 run 2) scoring 0.004 on [start_t, end_t] vs 0.351 on the arcs' own
    span. Reproduce the mechanism through segment_runs and detect_drops, the
    path analyze uses: take a full, cleanly periodic 16-throw run, turn one
    mid-run arc into a floor drop by extending its fitted t_end 0.09s along
    its own parabola (measured: y_at(t_end) ~= 0.736, past hand_line +
    FLOOR_MARGIN ~= 0.732, still descending) and stripping its catch, then
    remove the throws of the next two beats so detect_drops sees the pattern
    collapse. The run ends at that drop, yet its quality must match the twin
    run in which that arc was caught, because quality is scored over the
    arcs' own span.
    """
    r = simulate_cascade(n_throws=16, fps=30.0, seed=1)
    arcs = extract_arcs(r.detections)
    hl = estimate_hand_line(arcs)
    throws, catches = derive_events(arcs, hl)

    arcs_sorted = sorted(arcs, key=lambda a: a.t_start)
    target = arcs_sorted[3]
    mutated = target.model_copy(update={"t_end": target.t_end + 0.09})
    assert is_floor_bound(mutated, hl), "fixture must genuinely be floor-bound"
    _, miss_t = hand_line_crossings(mutated, hl)
    paused = {a.id for a in arcs if miss_t < a.t_start <= miss_t + 2 * r.params.period_s}
    assert paused, "fixture must pause the pattern after the miss"
    arcs = [a for a in arcs if a.id not in paused]
    throws = [e for e in throws if e.arc_id not in paused]
    catches = [c for c in catches if c.arc_id not in paused]

    twin = segment_runs(arcs, throws, catches, hl)
    dropped = [mutated if a.id == target.id else a for a in arcs]
    runs = segment_runs(dropped, throws, [c for c in catches if c.arc_id != target.id], hl)
    drops, runs = detect_drops(dropped, runs, hl)
    assert [d.arc_id for d in drops] == [target.id]
    assert len(runs) == 1 and len(twin) == 1
    run = runs[0]
    assert run.arc_ids == twin[0].arc_ids
    assert run.end_reason == "drop"
    # end_t is the drop's own hand-line crossing -- far short of the group's
    # real arc span (16 throws over ~7s).
    assert run.end_t == pytest.approx(miss_t)
    assert run.end_t < arcs_sorted[-1].t_end - 2.0
    assert run.quality > 0.3
    assert run.quality == pytest.approx(twin[0].quality, abs=0.03)


def test_adaptive_lag_band_scores_meschke_style_slow_period():
    """The in-repo default lag band (0.2, 1.2) can miss a run's true period
    entirely on differently-framed footage (Meschke ground truth: period
    1.19-2.89s, wholly outside (0.2, 1.2)). There it does not score low: it
    peaks at its 0.2s edge, which measures near-zero-lag smoothness rather
    than periodicity, and on this fixture scores higher than the true period
    does (measured 0.719 vs 0.570), so a bare quality floor cannot catch it.
    segment_runs must search around the run's OWN estimated period --
    [max(0.1, 0.5p), 1.5p] -- so quality is the score at the true period.
    Reproduce with a slow-period simulated cascade whose true period sits
    outside the fixed default band.
    """
    slow = CascadeParams(period_s=1.5, dwell_s=0.9, g=0.25)
    r = simulate_cascade(n_throws=12, fps=30.0, seed=1, params=slow)
    arcs, hl, _, _, runs = pipeline(r.detections)
    assert len(runs) == 1
    run = runs[0]
    p = run.period_s
    assert p == pytest.approx(1.5, abs=0.05)
    # Confirm this period truly sits outside the fixed default band -- the
    # regression this test pins would be invisible if it didn't.
    assert not (0.2 <= p <= 1.2)
    assert run.quality > 0.3

    def arc_end(a: Arc) -> float:
        crossings = hand_line_crossings(a, hl)
        return crossings[1] if crossings else a.t_end

    group = [a for a in arcs if a.id in run.arc_ids]
    span = (min(a.t_start for a in group), max(arc_end(a) for a in group))
    own, lag = periodicity_score(group, *span, lag_range=(0.5 * p, 1.5 * p))
    fixed, _ = periodicity_score(group, *span, lag_range=(0.2, 1.2))
    assert lag == pytest.approx(1.5, abs=0.1)
    assert abs(fixed - own) > 0.1, "fixture must tell the two bands apart"
    assert run.quality == pytest.approx(own, abs=0.05)


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
