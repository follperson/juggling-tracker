import sys
import subprocess
from collections import defaultdict
from pathlib import Path

import pytest
from pydantic import ValidationError

from juggletrack.analyze import analyze_detections
from juggletrack.pipeline.realtime import RealtimeAnalyzer, RealtimeConfig
from juggletrack.sim import CascadeParams, simulate_cascade
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
    # T7: before zipping (which silently truncates `catches` to
    # len(confirm_t)), assert confirm count is in the ballpark of offline's
    # -- catches near the very end of the stream legitimately aren't
    # confirmed yet by the time the feed loop ends (their freeze horizon
    # hasn't arrived; finalize() would flush them, but this test is
    # specifically about the feed()-time latency bound, so it doesn't call
    # finalize()). Without this, a regression that suppressed nearly ALL
    # confirmations would still pass the loop below vacuously (zip of an
    # empty/near-empty list).
    assert len(confirm_t) >= len(catches) - 5, (
        f"only {len(confirm_t)}/{len(catches)} catches confirmed before stream "
        "end -- too few for the latency-bound check below to be meaningful"
    )
    bound = cfg.freeze_s + cfg.cadence_frames / 30.0 + 0.25
    # T7: lower bound too -- a catch confirmed much EARLIER than
    # freeze_s - cadence/30 (with a little slack for float/frame-boundary
    # slop) would mean the freeze horizon isn't actually being honored.
    bound_lo = cfg.freeze_s - cfg.cadence_frames / 30.0 - 0.05
    for ct, ev in zip(confirm_t, sorted(c.t for c in catches)[: len(confirm_t)]):
        assert bound_lo <= ct - ev <= bound, (
            f"catch at {ev:.2f} confirmed at lag {ct - ev:.2f}s, want "
            f"[{bound_lo:.2f}, {bound:.2f}]"
        )


def test_analysis_stays_fast_and_reports_timing():
    """T3: mean<50ms flaked under load (observed: 1/17 failures across a
    concurrent-process run, clean in isolation) -- a wall-clock timing
    assertion inherently depends on machine load, not just the code under
    test. Switched to median (robust to a handful of slow outlier cycles
    from scheduling jitter, unlike mean) with a load-tolerant bound: 150ms
    is still 3-5x the typically-measured 27-43ms per cycle on this
    machine (see realtime.py's module docstring), so it stays a real
    regression guard (would still catch e.g. an accidental O(n^2) blowup)
    without flaking under CI contention."""
    r = simulate_cascade(n_throws=24, fps=30.0, seed=3)
    states = stream(r, RealtimeAnalyzer())
    timings = sorted(s.last_analysis_ms for s in states if s.last_analysis_ms > 0)
    assert timings, "no analysis timings recorded"
    median_ms = timings[len(timings) // 2]
    assert median_ms < 150, f"median re-analysis {median_ms:.1f}ms exceeds load-tolerant bound"


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

    T2 CORRECTED ATTRIBUTION: this specific 1.0s gap does NOT actually
    exercise RUN_CLOSE_DEBOUNCE_S (0.5s) at all -- 1.0s is comfortably
    inside freeze_s (1.5s), so `live` (via the freeze-horizon liveness
    check in _analyze) never even goes false during the gap; the debounce
    mechanism never gets a chance to engage. The under-count here is
    entirely freeze_s's own liveness horizon doing its job (by design: an
    arc is still "recent" for freeze_s after it ends). The debounce ITSELF
    -- the `_pending_close_t` mechanism that absorbs a genuine liveness
    flap -- is isolated and proven load-bearing by
    `test_debounce_absorbs_genuine_mid_run_detection_gap` below, which
    empirically differs with RUN_CLOSE_DEBOUNCE_S disabled and this test
    does not (verified: disabling it changes nothing here).

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
        "this pins a documented trade-off (freeze-horizon liveness merges "
        "narrow real gaps), not desired behavior"
    )


def test_debounce_absorbs_genuine_mid_run_detection_gap():
    """T2 ORIGINAL INTENT: isolate RUN_CLOSE_DEBOUNCE_S itself, which the
    narrow-gap test above does NOT (see its corrected docstring) -- that
    fixture's 1.0s gap never makes liveness go false in the first place, so
    the debounce never engages there.

    Plan-5 task 3 UPDATE (mechanism shift, measured): this fixture no longer
    isolates the debounce -- Task 3's fix (segment_runs only truncates
    end_t at a floor-bound miss; an uncaught arc that's merely unwitnessed,
    e.g. cut short by this very detector blackout, no longer truncates it)
    already prevents the false split this test used to rely on the
    debounce for. Measured on this exact fixture (gap_dur=1.7s, gap0 at 60%
    through the run): BEFORE task 3, disabling the debounce
    (RUN_CLOSE_DEBOUNCE_S=-1.0) flipped runs_completed from 1 to 2 (the
    false split the debounce existed to absorb). AFTER task 3, disabling
    the debounce on the SAME fixture still gives 1 -- no false split occurs
    at all, debounce or not.

    Why: `strict_live` reads `r.end_t` from segment_runs' run objects, which
    are recomputed from the FULL arc history every cycle. During this
    blackout, at least one other ball is always still mid-flight (3-ball
    cascade, period 0.45s << flight duration 1.1s -- overlap is structural,
    not incidental to this fixture), so its arc gets fit from only its
    real, pre-blackout points and its *analytic* hand-line crossing
    (extrapolated past those points, same fit-then-solve `arc_end` used
    everywhere in runs.py) lands later, inside or past the blackout. Since
    that arc is uncaught-but-not-floor-bound (it ends mid-air, nowhere near
    the hand line), task 3 lets it flow into the `max(arc_end(a) for a in
    group)` fallback instead of truncating end_t to some earlier miss --
    keeping the run's end_t "fresh" enough that `end_t > now - freeze_s`
    never goes false during this gap. An exhaustive re-check (gap position
    from 30%-97% through the run, gap_dur from 1.55s-1.95s) found no
    parameter combination in this fixture family that still reproduces the
    pre-task-3 false split -- the overlap that rescues it is structural to
    any continuous multi-ball cascade at this period/flight ratio, not a
    property of this one gap's placement.

    This test is kept (rather than deleted) as regression coverage for
    task 3's benefit itself: both assertions below now hold with or
    without the debounce, which is exactly what "fewer premature closes"
    (the task's stated goal) looks like from the realtime engine's side.
    If the "without debounce" control ever regresses to 2 here, task 3's
    protection has been lost for this mechanism and needs re-diagnosis --
    not a debounce-band widening.

    Offline (which has no debounce concept, and free-runs segment_runs' own
    fixed gap_factor*period grouping threshold -- 1.3*0.45 =~ 0.585s, far
    below this gap) still segments this into 2 runs; that's expected and
    is not what this test is checking. This test is specifically about
    RealtimeAnalyzer's own liveness (now dominated by task 3's fix, not the
    debounce) for this gap shape.

    Plan 5 task 2: per-frame duplicate-box clustering briefly threatened
    this test (a confidence-descending-sort-only merge rule let a genuine
    sub-0.03 crossing between two distinct balls here flip runs_completed
    from 1 to 2), fixed by strict-lower-confidence absorption -- two real
    balls tied at confidence=1.0 never merge, so clustering is a no-op on
    this fixture at the AnalyzeConfig default. Verified directly; reverted
    the transient cluster_merge_dist=0.0 override.

    Task 3b UPDATE (re-verified, not just re-asserted): re-ran both
    assertions above against the actual post-task-3 code and they still
    hold -- for THIS fixture's specific shape, task 3's end_t fix (not the
    debounce) is what prevents the false split, and the debounce is
    provably a no-op here. That does NOT generalize to "the debounce is
    dead code": it isolates one structural rescue (a 3-ball OVERLAPPING
    cascade always has a second ball mid-flight during the gap, whose own
    arc_end extrapolates past it) that this fixture happens to have and a
    slower, non-overlapping cascade does not. Two independent lines of
    evidence show the debounce is still load-bearing once that rescue is
    absent: (1) `test_debounce_absorbs_gap_in_no_overlap_cascade` below,
    a deliberately non-overlapping fixture (period_s > flight_s, one ball
    airborne at a time) where killing the debounce still flips a genuine
    single run to a false 2, post-task-3; (2) field replays on real
    footage with the debounce killed (task-3b report,
    .superpowers/sdd/task-3b-report.md): ss3_id_016 runs_completed
    9 (matches offline) -> 70; af2 2 (matches offline) -> 3; catches/drops
    totals unchanged in both cases, isolating the effect to run-counting.
    """
    r = simulate_cascade(n_throws=20, fps=30.0, noise=0.003, dropout=0.1, seed=5)
    cfg = RealtimeConfig()
    gap_dur = 1.7
    assert cfg.freeze_s < gap_dur < cfg.freeze_s + RealtimeAnalyzer.RUN_CLOSE_DEBOUNCE_S, (
        "gap_dur must sit strictly inside (freeze_s, freeze_s + debounce) for "
        "this repro to originally have isolated the debounce (pre-task-3)"
    )
    gap0 = r.run_start + (r.run_end - r.run_start) * 0.6
    dets = [d for d in r.detections if not (gap0 <= d.t < gap0 + gap_dur)]

    final = stream_dets(dets, RealtimeAnalyzer(cfg))[-1]
    assert final.runs_completed == 1, (
        f"this mid-run gap must still be counted as one run; got {final.runs_completed}"
    )

    class _NoDebounce(RealtimeAnalyzer):
        RUN_CLOSE_DEBOUNCE_S = -1.0

    without_debounce = stream_dets(dets, _NoDebounce(cfg))[-1]
    assert without_debounce.runs_completed == 1, (
        "post-task-3: the run-span fix (not the debounce) now prevents the "
        f"false split on this fixture even with the debounce disabled; got "
        f"{without_debounce.runs_completed} -- a return to 2 here means task "
        "3's end_t protection has regressed for this mechanism, not that the "
        "debounce needs restoring"
    )


def test_debounce_absorbs_gap_in_no_overlap_cascade():
    """Task 3b: re-isolates RUN_CLOSE_DEBOUNCE_S under CORRECT (post-task-3)
    span semantics, after the review found the debounce delete-proof again
    on `test_debounce_absorbs_genuine_mid_run_detection_gap`'s own fixture
    (see that test's Task 3b UPDATE paragraph). That fixture's rescue is
    structural to any OVERLAPPING cascade: a second ball is always
    mid-flight during the gap, and task 3's fix lets its arc_end extrapolate
    past the gap, so liveness never actually goes false there -- the
    debounce never gets a chance to matter. This fixture removes that
    rescue by construction: `CascadeParams(period_s=2.0, dwell_s=4.5)` makes
    `flight_s = 3*period_s - dwell_s = 1.5s < period_s`, so at most ONE ball
    is ever airborne at a time (verified: `p.period_s > p.flight_s`) -- there
    is no second, still-flying arc anywhere near the gap whose extrapolated
    end could rescue liveness. With the single flight that straddles the cut
    evicted along with everything else in [gap0, gap0+gap_dur), liveness
    genuinely has nothing to hold onto for the gap's own ~2s duration.

    gap_dur=1.9s sits in (freeze_s=1.5, freeze_s+RUN_CLOSE_DEBOUNCE_S+1.0)
    as the task specified, and (verified by direct sweep, not assumed) in
    the middle of a comfortably wide plateau (1.75-1.92s at this exact
    gap0) where this fixture's premise holds -- not a knife-edge: gap_dur
    below ~1.93s gives offline=1/on=1/off=2 consistently; above it the
    picture changes (both on and off give 2, a separate live/offline
    disagreement unrelated to the debounce, out of scope here).

    Offline (single whole-video pass over the identical gapped detections)
    still recognizes ONE continuous run -- catches=7, throws=7, matching
    every real throw/catch in this 7-throw fixture, 0 drops -- because
    segment_runs' own gap tolerance (`gap_factor * period`, computed from
    the run's own ~2s period once all arcs are visible) comfortably exceeds
    this gap once both sides of it are in view. Live has to decide whether
    to close the run WHILE still inside the gap, before the far side's arcs
    exist to prove that same bridging -- that timing gap is exactly what
    the debounce buys: enough delay to let live-and-recover cancel a
    premature close instead of committing it.

    MEASURED, both directions: WITH the debounce (`on`, current code, RED
    would mean this regresses) matches offline at runs_completed=1. WITHOUT
    it (`off`, RUN_CLOSE_DEBOUNCE_S=-1.0) false-splits to 2 -- confirming the
    debounce, not task 3's fix, is what's load-bearing for THIS shape.
    catches_total/drops_total are identical (7/0) in both variants, isolating
    the divergence to run-counting alone, same as the sibling fixture above.
    """
    p = CascadeParams(period_s=2.0, dwell_s=4.5, n_balls=3)
    assert p.period_s > p.flight_s, (
        "fixture must be non-overlapping (at most one ball airborne at a "
        f"time); got period_s={p.period_s} <= flight_s={p.flight_s}"
    )
    r = simulate_cascade(n_throws=7, fps=30.0, params=p, seed=7)
    cfg = RealtimeConfig()
    gap_dur = 1.9
    assert cfg.freeze_s < gap_dur < cfg.freeze_s + RealtimeAnalyzer.RUN_CLOSE_DEBOUNCE_S + 1.0, (
        "gap_dur must sit inside (freeze_s, freeze_s + debounce + 1.0) per "
        "the task-3b flap-hunt brief"
    )
    gap0 = r.run_start + (r.run_end - r.run_start) * 0.4
    dets = [d for d in r.detections if not (gap0 <= d.t < gap0 + gap_dur)]

    offline = analyze_detections(dets)
    assert len(offline.runs) == 1, "fixture must offline-segment into 1 continuous run"
    off_catches = sum(run.catches for run in offline.runs)

    with_debounce = stream_dets(dets, RealtimeAnalyzer(cfg))[-1]
    assert with_debounce.runs_completed == 1, (
        f"debounce must absorb this genuine no-overlap detection gap; got "
        f"{with_debounce.runs_completed}"
    )
    assert with_debounce.catches_total == off_catches

    class _NoDebounce(RealtimeAnalyzer):
        RUN_CLOSE_DEBOUNCE_S = -1.0

    without_debounce = stream_dets(dets, _NoDebounce(cfg))[-1]
    assert without_debounce.runs_completed == 2, (
        "without the debounce this genuine single run must false-split -- if "
        f"this now gives 1, the debounce has become dead code for this "
        f"mechanism too and this test's premise needs revisiting; got "
        f"{without_debounce.runs_completed}"
    )
    assert without_debounce.catches_total == off_catches


def test_realtime_module_is_cv2_free():
    # T8: cwd="src" was relative to whatever directory pytest happened to be
    # invoked FROM, not this test file's location -- fragile (silently
    # wrong, not a clean failure) if the suite is ever run from elsewhere.
    src_dir = Path(__file__).resolve().parents[1] / "src"
    code = ("import sys; import juggletrack.pipeline.realtime; "
            "sys.exit(1 if 'cv2' in sys.modules else 0)")
    proc = subprocess.run([sys.executable, "-c", code], cwd=src_dir)
    assert proc.returncode == 0, "importing realtime pulled in cv2"


def _three_run_stream(gap_s, noise, dropout):
    """Concatenate three independent 12-throw cascades with `gap_s` of
    silence between each pair (same shift pattern as `_two_run_stream`,
    extended to a third run). Each run individually lasts well under
    `RealtimeConfig.window_s` (8s, ~6.5s per run here), but the whole
    stream (~24s) spans far longer than the window.

    `noise`/`dropout` matter empirically, not cosmetically: at the
    near-noiseless levels `_two_run_stream` uses (0.003/0.1), this fixture
    reproduces NO divergence at all pre-fix (verified) -- three short,
    clean, gapped runs never give the sliding window a reason to
    misbehave. Measured (see docs/superpowers/plans/
    2026-07-19-plan4-bench-findings.md's §9 evidence appendix) that
    noise=0.015/dropout=0.15 is enough real-detector-like jitter to
    reproduce the same class of window-boundary run-churn the field bench
    found on ss3_id_016 (offline 9 runs vs live 46), just at a much
    smaller, fast, deterministic scale suitable for a unit test."""
    a = simulate_cascade(n_throws=12, fps=30.0, noise=noise, dropout=dropout, seed=41)
    b = simulate_cascade(n_throws=12, fps=30.0, noise=noise, dropout=dropout, seed=42)
    c = simulate_cascade(n_throws=12, fps=30.0, noise=noise, dropout=dropout, seed=43)

    dt_b = gap_s + a.run_end - b.run_start
    b_shifted = shift(b.detections, dt_b, round(dt_b * 30.0))
    b_run_end = b.run_end + dt_b

    dt_c = gap_s + b_run_end - c.run_start
    c_shifted = shift(c.detections, dt_c, round(dt_c * 30.0))

    return a.detections + b_shifted + c_shifted


def test_parity_very_long_stream():
    """The field-bench failure (docs/superpowers/plans/
    2026-07-19-plan4-bench-findings.md §3, §8):
    ss3_id_016 (205s of real footage, runs up to ~14s) turned 9 offline
    runs/39 catches/0 drops into 46 runs/306 catches/42 phantom drops live.
    This reproduces the same class of failure -- window-boundary run churn
    -- with three ~6.5s synthetic runs (each individually well under
    window_s=8) separated by ~3s gaps, spanning ~24s total (>> window_s),
    at just enough real-detector-like noise (see `_three_run_stream`) to
    make segment_runs' from-scratch-every-cycle re-derivation flap.

    MEASURED, not assumed (both directions reproduced and pinned here):
    pre-fix-wave-1 (dd01055) this fixture inflated runs_completed by +2 and
    catches_total by +4 (offline 3/23/0 -> live 5/27/0). Post-wave-1,
    runs_completed and drops_total matched offline exactly and catches_total
    sat at +4 over. RE-MEASURED after THIS fix wave (E1 event-bearing
    sticky-open gate, E2 unified hand line, E3 run-membership catch gate):
    runs_completed and drops_total still match offline exactly; catches_total
    is UNCHANGED at +4 over (live=27) -- none of E1/E2/E3 touch this
    fixture's specific gap, confirming (not merely asserting) it comes from a
    separate, broader re-extraction-instability mechanism (local-window vs
    whole-video arc segmentation) that this wave's fixes don't target. The
    band is signed, not absolute: live has only ever been measured
    OVER-counting on this fixture, never under, across both fix waves --
    an absolute-value band would silently accept a live UNDER-count that
    would actually be a new, different regression. See the findings doc's
    §8 (Wave 2/3) and §9 evidence appendix for the much larger version of
    this same gap on real ss3_id_016 footage and why it's out of scope for
    this design.

    Plan 5 task 2: per-frame duplicate-box clustering briefly threatened
    this test (a confidence-descending-sort-only merge rule let a genuine
    sub-0.03 crossing here inflate offline 23 -> 25), fixed by strict-lower-
    confidence absorption -- two real balls tied at confidence=1.0 never
    merge, so clustering is a no-op on this fixture at the AnalyzeConfig
    default. Verified directly; reverted the transient
    cluster_merge_dist=0.0 override on both configs below."""
    dets = _three_run_stream(gap_s=3.0, noise=0.015, dropout=0.15)
    offline = analyze_detections(dets)
    total_span = max(d.t for d in dets) - min(d.t for d in dets)
    assert total_span > 20.0, "fixture must span well beyond window_s=8"
    assert len(offline.runs) == 3, "fixture must offline-segment into 3 runs"
    assert len(offline.drops) == 0, "fixture has no drops by construction"
    off_catches = sum(run.catches for run in offline.runs)
    assert off_catches == 23, "pinned offline baseline for this fixture; revisit if sim.py changes"

    analyzer = RealtimeAnalyzer()
    final = stream_dets(dets, analyzer)[-1]

    assert final.runs_completed == len(offline.runs), (
        f"live={final.runs_completed} offline={len(offline.runs)}"
    )
    assert 0 <= final.catches_total - off_catches <= 4, (
        f"live={final.catches_total} offline={off_catches} -- live must OVER-count "
        "by at most 4 here (never under-count; see docstring for the measured "
        "post-fix-wave gap this pins)"
    )
    assert final.drops_total == len(offline.drops) == 0, (
        f"live={final.drops_total} offline={len(offline.drops)}"
    )


def _low_hop_dets(t_start, t_end, fps=30.0, y0=0.85, g=2.22, x0=0.5, vx=0.05, hop_dur=0.75):
    """Synthetic dropped-ball floor junk: a chain of shallow ballistic
    "hops" (ay=g/2 inside g_range, apex height 0.85-0.156=0.694, well below
    a ~0.63 hand line so derive_events' apex guard rejects every one of
    them as a throw) spanning [t_start, t_end) continuously, with slow x
    drift so the static-detection pre-filter (extract_arcs.
    filter_static_detections) doesn't discard the whole chain as background
    clutter. Reproduces the domain-native case of a dropped ball bouncing
    on the floor during the silence between two real runs -- confirmed
    (see test_bounce_junk_in_gap_does_not_merge_runs) to extract as clean,
    real Arc objects that never produce a throw/catch/run."""
    dets = []
    t = t_start
    while t < t_end:
        dur = min(hop_dur, t_end - t)
        if dur < 0.2:
            break
        v0 = g * dur / 2.0
        n = int(dur * fps)
        for i in range(n + 1):
            dt = i / fps
            if dt > dur:
                break
            tt = t + dt
            y = y0 - v0 * dt + 0.5 * g * dt * dt
            x = x0 + vx * (tt - t_start)
            dets.append(Detection(frame_idx=round(tt * fps), t=tt,
                                   x=min(max(x, 0.0), 1.0), y=min(y, 0.97)))
        t += dur
    return dets


def test_bounce_junk_in_gap_does_not_merge_runs():
    """E1 (reviewer's repro P3): sticky-open liveness must only treat
    EVENT-BEARING arcs (apex clears the hand line, i.e. produced a derived
    throw) as proof "the pattern is still going" -- not ANY arc in the
    window. A dropped ball bouncing on the floor during the silence
    between two real runs is exactly this kind of junk: it fits a clean
    ballistic parabola (ay inside g_range) but never rises above hand
    height, so derive_events' apex guard rejects every throw from it.
    Pre-fix, `recent_activity` counted these junk arcs too, so the first
    run never looked "dead" and the debounce never fired -- the two runs
    merged into one (measured pre-fix: this exact fixture gave live
    runs_completed=1 vs offline's 2; the no-junk control below already
    gives 2/2, isolating the effect to the junk arcs specifically).
    """
    a = simulate_cascade(n_throws=10, fps=30.0, noise=0.003, dropout=0.1, seed=29)
    b = simulate_cascade(n_throws=10, fps=30.0, noise=0.003, dropout=0.1, seed=30)
    gap_s = 3.0
    dt_s = gap_s + a.run_end - b.run_start
    dframes = round(dt_s * 30.0)
    bounce_dets = _low_hop_dets(a.run_end + 0.2, a.run_end + gap_s - 0.2)
    dets = a.detections + bounce_dets + shift(b.detections, dt_s, dframes)

    offline = analyze_detections(dets)
    assert len(offline.runs) == 2, "fixture must offline-segment into 2 runs despite the junk"
    off_catches = sum(r.catches for r in offline.runs)

    analyzer = RealtimeAnalyzer()
    final = stream_dets(dets, analyzer)[-1]
    assert final.runs_completed == 2, (
        f"live={final.runs_completed} (bounce-junk arcs must not bridge the gap)"
    )
    assert final.catches_total == off_catches

    # Control: the identical two-run/3.0s-gap fixture WITHOUT the junk
    # arcs already counts 2/2 (test_two_runs_with_wide_gap_counted_
    # separately) -- confirming the junk, not the gap itself, was the
    # pre-fix failure mode.
    dets_ctrl = a.detections + shift(b.detections, dt_s, dframes)
    offline_ctrl = analyze_detections(dets_ctrl)
    assert len(offline_ctrl.runs) == 2
    final_ctrl = stream_dets(dets_ctrl, RealtimeAnalyzer())[-1]
    assert final_ctrl.runs_completed == 2


def test_catches_below_min_arcs_are_not_confirmed_live():
    """E3 repro: live used to confirm every derived catch regardless of run
    membership, while offline's parity reference (sum of Run.catches,
    runs.py:96) counts only catches whose arc is inside a segment_runs
    group (min_arcs=3 gate). A clean 2-throw stream -- below the gate --
    has 0 offline run-catches but used to give live catches_total=2."""
    r = simulate_cascade(n_throws=2, fps=30.0, seed=1)
    offline = analyze_detections(r.detections)
    assert len(offline.runs) == 0, "fixture must stay below the min_arcs gate offline"
    off_catches = sum(run.catches for run in offline.runs)
    assert off_catches == 0

    final = stream(r, RealtimeAnalyzer())[-1]
    assert final.catches_total == off_catches == 0


def test_config_rejects_window_too_small_for_freeze_envelope():
    """E4 (reviewer's repro P2): window_s below the freeze+flight envelope
    used to silently zero out nearly every count (measured: the standard
    drop fixture went from offline 10 catches/1 drop to live 1 catch/0
    drops with window_s=2.5, no error or warning). RealtimeConfig must
    reject this at construction instead of accepting it silently."""
    with pytest.raises(ValidationError):
        RealtimeConfig(window_s=2.5)


def test_config_accepts_default():
    RealtimeConfig()  # must not raise


@pytest.mark.parametrize("kwargs", [
    {"window_s": -1.0},
    {"window_s": 0.0},
    {"cadence_frames": 0},
    {"freeze_s": -1.0},
    {"drop_freeze_s": -1.0},
    {"hand_line_ema_alpha": 0.0},
    {"hand_line_ema_alpha": 1.5},
    {"edge_pad": -5.0},
    {"event_match_tol": -0.01},
])
def test_config_rejects_invalid_fields(kwargs):
    """E4: the remaining individual field constraints from the finding's
    suggested fix (window_s > 0, cadence_frames >= 1, freeze_s/
    drop_freeze_s >= 0, 0 < hand_line_ema_alpha <= 1).

    Re-review follow-up (MINOR): a negative `edge_pad` silently empties
    `truncated_ids` in `_analyze` (every arc's `t_start` trivially satisfies
    `t_start < left_edge + edge_pad` less often, or for a large-enough
    negative value, never) -- disabling the left-edge guard with no error;
    a negative `event_match_tol` would make `_confirm`'s `abs(et - k) <=
    tol` comparison never match anything with `tol < 0` (an always-false
    dedup gate), silently disabling dedup entirely. RED repro:
    `RealtimeConfig(edge_pad=-5.0)` constructed with no error before this
    fix (verified: `pydantic.ValidationError` was NOT raised)."""
    with pytest.raises(ValidationError):
        RealtimeConfig(**kwargs)


def test_feed_after_finalize_raises():
    """E5: feed() after finalize() used to silently resume analysis with no
    way to ever flush the resumed tail again (finalize() is idempotent --
    a second call is a no-op once _finalized is set), permanently losing
    events inside the final freeze window and any run re-opened after the
    "final" state. Make the contract explicit."""
    analyzer = RealtimeAnalyzer()
    analyzer.feed([], 0.0)
    analyzer.finalize()
    with pytest.raises(RuntimeError):
        analyzer.feed([], 1.0)


def test_finalize_recognizes_run_completing_only_in_final_window():
    """E6 repro: finalize's flush pass (freeze=-1) used the SAME -1 for the
    run-liveness horizon too (`now - freeze == now + 1.0`, unsatisfiable),
    so a run whose completing arcs were never visible during an earlier
    periodic feed() cycle could never open at all and was silently dropped
    from runs_completed entirely (not merely mis-timed). cadence_frames is
    set absurdly high here so _analyze runs exactly once, at finalize() --
    the only place this run's arcs are ever seen."""
    r = simulate_cascade(n_throws=3, fps=30.0, seed=7)
    offline = analyze_detections(r.detections)
    assert len(offline.runs) == 1, "fixture must offline-segment into 1 run"

    analyzer = RealtimeAnalyzer(RealtimeConfig(cadence_frames=1000))
    for t, dets in frames_of(r.detections):
        analyzer.feed(dets, t)
    final = analyzer.finalize()
    assert final.runs_completed == 1, f"live={final.runs_completed} (run silently dropped)"


def test_buffer_prune_evicts_even_when_first_det_has_large_t():
    """E7: feed() used `buffer[0].t < cut` as an early-exit proxy for "does
    anything need evicting", assuming the buffer stays t-sorted. A single
    out-of-order detection with an anomalously large t at buffer[0] (a
    plausible real-world glitch: a detector emitting slightly reordered
    timestamps) defeats that proxy and used to defer eviction of
    everything behind it indefinitely, growing the buffer unboundedly.
    window_s=2.0 (smallest that still clears E4's envelope validator with
    correspondingly small freeze/drop_freeze/edge_pad -- this test only
    cares about the buffer-eviction mechanism, not a realistic envelope)."""
    cfg = RealtimeConfig(window_s=2.0, edge_pad=0.1, freeze_s=0.1, drop_freeze_s=0.1)
    analyzer = RealtimeAnalyzer(cfg)
    analyzer.feed([Detection(frame_idx=0, t=100.0, x=0.5, y=0.5)], 0.0)
    n_frames = 150
    for i in range(1, n_frames):
        analyzer.feed([Detection(frame_idx=i, t=i / 30.0, x=0.5, y=0.5)], i / 30.0)
    # window_s=2.0 at t~=4.97s: only the last ~2.0s of real detections (plus
    # the one spurious far-future point) should survive -- NOT the full
    # {n_frames}-frame history the old early-exit would have retained
    # forever (buffer[0].t=100.0 never satisfies `< cut`, so the old code
    # never even looked at the rest of the buffer).
    assert len(analyzer._buffer) < n_frames // 2, (
        f"buffer holds {len(analyzer._buffer)} dets; eviction was deferred"
    )


def test_confirm_does_not_dedup_within_same_cycle():
    """E8: two distinct events derived in the SAME analysis cycle that land
    within event_match_tol of each other must both be kept. Pre-fix,
    `_confirm` treated the second's closeness to the first (added earlier in
    the SAME call, since it checked the mutating `known` list instead of a
    prior-cycle snapshot) as a duplicate and dropped it.

    NOT claimed "by construction" that one derive_events()/detect_drops()
    pass never emits two synthetic times for the same physical event (the
    findings doc's §9 evidence appendix measured the opposite: 2-5 mutually
    time-overlapping arc candidates per cycle for one physical flight, each
    with a distinct arc id) -- this is a narrower, MEASURED claim about
    real inter-event spacing clearing event_match_tol, plus a fix to a
    same-call ordering bug that holds regardless.

    MEASURED closest real inter-event spacing (plan step 4(c); a one-off,
    not-committed scratch script ran offline analyze_detections +
    derive_events over this file's own realtime-parity fixtures -- the
    12-throw sim, both _two_run_stream gaps, and _three_run_stream -- and
    took the minimum consecutive gap between sorted event times):

        fixture                                    throw-throw  catch-catch  pooled
        12-throw sim (seed=1, no noise)             0.450s       0.450s       0.167s
        12-throw sim (noise=0.003/dropout=0.1)      0.447s       0.447s       0.166s
        two-run stream, gap_s=3.0                   0.447s       0.446s       0.165s
        two-run stream, gap_s=1.0                   0.447s       0.446s       0.165s
        _three_run_stream (noise=0.015/dropout=0.15) 0.415s      0.430s       0.139s

    `_confirm` dedups each kind ("throw"/"catch"/"drop") against its OWN
    list independently, so the load-bearing comparison is the SAME-KIND
    (throw-throw / catch-catch) minimum, not pooled: the closest real
    same-kind pair measured across all these fixtures is 0.415s
    (_three_run_stream throw-throw) against event_match_tol=0.15 -- a
    ~0.265s (~2.8x) margin, comfortably clear. The pooled (throw-vs-catch)
    minimum does dip below tol on the noisiest fixture (0.139s < 0.15s), but
    that pairing is cross-kind and never enters the same dedup comparison,
    so it does not threaten a real collision under this mechanism."""
    analyzer = RealtimeAnalyzer()
    analyzer._confirm("catch", [1.000, 1.100], horizon=10.0)
    assert analyzer._confirmed["catch"] == [1.000, 1.100]


def test_confirm_still_dedups_across_cycles():
    """Companion to the above: an event re-derived in a LATER cycle within
    tol of one already confirmed in an EARLIER cycle must still be treated
    as the same physical event -- this is the mechanism the dedup exists
    for (e.g. a re-fit shouldn't double a real confirmation across
    cycles), and E8's fix must not break it."""
    analyzer = RealtimeAnalyzer()
    analyzer._confirm("catch", [1.000], horizon=10.0)
    analyzer._confirm("catch", [1.05], horizon=10.0)  # later cycle, same physical event
    assert analyzer._confirmed["catch"] == [1.000]


def test_left_edge_guard_prevents_phantom_catches_from_truncated_refit():
    """T1: the left-edge guard (dd01055) was otherwise delete-proof -- no
    existing fixture failed with edge_pad=0 (verified: every parity test
    above passes identically either way). This one does: on this specific
    noisy long-stream fixture, a truncated re-fit near the window's
    trailing edge manufactures phantom catches that the guard's event
    filtering suppresses. Verified this is the isolating mechanism (not a
    side effect of some other fix): with the guard active (default
    edge_pad=0.5), catches_total matches offline EXACTLY; with it disabled
    (edge_pad=0.0) on the IDENTICAL detections, catches_total picks up 3
    phantom catches.

    Plan 5 task 2: per-frame duplicate-box clustering briefly threatened
    this test's pinned numbers (confidence-descending-sort-only merging let
    a genuine crossing here slip through), fixed by strict-lower-confidence
    absorption -- two real balls tied at confidence=1.0 never merge, so
    clustering is a no-op on this fixture at the AnalyzeConfig default.
    Verified directly; reverted the transient cluster_merge_dist=0.0
    override on both configs below."""
    r = simulate_cascade(n_throws=24, fps=30.0, noise=0.01, dropout=0.1, seed=4)
    offline = analyze_detections(r.detections)
    off_catches = sum(run.catches for run in offline.runs)
    assert off_catches == 20, "pinned offline baseline for this fixture"

    guarded = stream(r, RealtimeAnalyzer(RealtimeConfig(edge_pad=0.5)))[-1]
    assert guarded.catches_total == off_catches, (
        f"left-edge guard active: expected {off_catches} catches, "
        f"got {guarded.catches_total}"
    )

    unguarded = stream(r, RealtimeAnalyzer(RealtimeConfig(edge_pad=0.0)))[-1]
    assert unguarded.catches_total == off_catches + 3, (
        "this pins the CURRENT measured effect of disabling the guard "
        f"(edge_pad=0.0); got {unguarded.catches_total} -- if this changes, the "
        "guard's isolating repro needs to be re-measured, not just loosened"
    )


def test_hand_line_ema_blends_across_cycles():
    """T4: EMA blending arithmetic, direct check (zero coverage before).
    cadence_frames=1 forces _analyze to run on every feed() call, so the
    buffer this test independently re-analyzes (via analyze_detections,
    exactly as _analyze itself does) is identical to what _analyze just
    used internally. Asserts state.hand_line_y always equals
    alpha*raw + (1-alpha)*previous_ema (first non-empty cycle: raw, no
    prior -- and empty-window cycles must leave the EMA, and therefore
    hand_line_y, untouched)."""
    r = simulate_cascade(n_throws=12, fps=30.0, seed=1)
    cfg = RealtimeConfig(cadence_frames=1)
    analyzer = RealtimeAnalyzer(cfg)
    ema_expected = None
    for t, dets in frames_of(r.detections):
        state = analyzer.feed(dets, t)
        result = analyze_detections(list(analyzer._buffer), cfg.analyze)
        if not result.arcs:
            if ema_expected is not None:
                assert state.hand_line_y == pytest.approx(ema_expected)
            continue
        raw = result.hand_line_y
        ema_expected = (
            raw if ema_expected is None
            else cfg.hand_line_ema_alpha * raw + (1.0 - cfg.hand_line_ema_alpha) * ema_expected
        )
        assert state.hand_line_y == pytest.approx(ema_expected)


def test_finalize_clears_run_active_and_catches_current_run():
    """T4: finalize()'s force-close path patches _last_state's run_active
    and catches_current_run (see finalize()'s own comment on why) -- direct
    assertion, zero coverage before."""
    r = simulate_cascade(n_throws=12, fps=30.0, seed=1)
    analyzer = RealtimeAnalyzer()
    for t, dets in frames_of(r.detections):
        analyzer.feed(dets, t)
    final = analyzer.finalize()
    assert final.run_active is False
    assert final.catches_current_run == 0


def test_duplicate_injection_inherits_clustering_via_analyze_detections():
    """Plan 5 task 2: RealtimeAnalyzer has no clustering logic of its own --
    it inherits per-frame duplicate-box clustering entirely through the
    single call to analyze_detections() inside _analyze (see realtime.py).
    Feeding a duplicate-injected stream (same clone-injection recipe as
    test_cluster.py::test_duplicate_injection_does_not_inflate_catches, which
    measured the pre-fix offline inflation at ss3_id_086's 24 -> 57 catches)
    must leave the LIVE final catches_total within +/-1 of the clean live
    baseline, confirming the fix's one integration point covers the
    realtime window path too, not just offline. Measured on this exact
    fixture with the shipped strict-lower-confidence absorption +
    merge_dist=0.023: clean=12, dirty=12 -- passing with margin, matching
    the offline measurement in test_cluster.py."""
    import numpy as np
    r = simulate_cascade(n_throws=12, fps=30.0, seed=3)
    rng = np.random.default_rng(7)
    clones = []
    for d in r.detections:
        for _ in range(int(rng.integers(2, 4))):
            clones.append(d.model_copy(update={
                "x": d.x + float(rng.uniform(-0.02, 0.02)),
                "y": d.y + float(rng.uniform(-0.02, 0.02)),
                "confidence": max(0.05, d.confidence * float(rng.uniform(0.3, 0.9))),
            }))
    clean_final = stream_dets(r.detections, RealtimeAnalyzer())[-1]
    dirty_final = stream_dets(r.detections + clones, RealtimeAnalyzer())[-1]
    assert abs(dirty_final.catches_total - clean_final.catches_total) <= 1
