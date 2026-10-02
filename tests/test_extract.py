import numpy as np
import pytest

from juggletrack.arcs.extract import (
    _EM_TIME_MARGIN,
    _em_assign_refit,
    _link_fragments,
    _merge_pass,
    _split_ballistic,
    dedup_parallel_arcs,
    extract_arcs,
    filter_static_detections,
)
from juggletrack.arcs.fit import fit_arc, points_array, x_residuals, y_residuals
from juggletrack.sim import simulate_cascade
from juggletrack.types import Arc, Detection


def match_arcs_to_flights(arcs, result, tol):
    """Return the arcs that match ground-truth (throw, catch) windows 1:1."""
    matched = []
    for th in result.throw_times:
        ca = th + result.params.flight_s
        hits = [a for a in arcs if abs(a.t_start - th) < tol and abs(a.t_end - ca) < tol]
        matched.append((th, hits))
    return matched


def test_clean_run_yields_one_arc_per_throw():
    r = simulate_cascade(n_throws=12, fps=30.0, seed=1)
    arcs = extract_arcs(r.detections)
    assert len(arcs) == 12
    for th, hits in match_arcs_to_flights(arcs, r, tol=0.06):
        assert len(hits) == 1, f"throw at {th} matched {len(hits)} arcs"
    p = r.params
    for a in arcs:
        assert a.ay == pytest.approx(p.g / 2.0, rel=0.05)
    assert arcs == sorted(arcs, key=lambda a: a.t_start)
    assert [a.id for a in arcs] == list(range(12))


def test_noise_and_dropout_still_recovers_all_arcs():
    r = simulate_cascade(n_throws=12, fps=30.0, noise=0.004, dropout=0.15, seed=2)
    arcs = extract_arcs(r.detections)
    assert len(arcs) == 12
    for th, hits in match_arcs_to_flights(arcs, r, tol=0.10):
        assert len(hits) == 1


def test_linker_survives_three_frame_gaps():
    """Turn-4 diagnosis: domain-shifted/outdoor footage doesn't drop so many
    detections overall that ``min_points`` fails -- it clusters the drops
    into 3-4 consecutive-frame gaps that the old 0.12s ``link_max_dt`` budget
    can't bridge (0.12s tolerated zero consecutive misses at 24fps
    stride-2), costing 60% of the outdoor holdout's real misses.

    Reproduce the same failure mode synthetically: a 12-throw cascade at
    24fps (the outdoor clip's approximate frame rate) with a 3-consecutive-
    frame detection gap injected twice per flight (mid-ascent, mid-descent;
    fractional offsets seeded for reproducibility), surgically removing only
    the targeted flight's own points (matched by recomputing its analytic
    position) so other simultaneously-airborne balls in the cascade are left
    untouched.

    Pins both sides of the fix: the legacy 0.12s budget (explicit override,
    stable regardless of the shipped default) stays starved below the 11/12
    recovery bar -- if this ever stops failing, the fixture no longer
    reproduces the diagnosed gap and needs revisiting -- while the shipped
    default must clear that bar.
    """
    fps = 24.0
    dt = 1.0 / fps
    r = simulate_cascade(n_throws=12, fps=fps, seed=2)
    p = r.params
    rng = np.random.default_rng(2)

    def flight_pos(i: int, t: float) -> tuple[float, float]:
        t0 = r.throw_times[i]
        hand = i % 2
        x0, x1 = p.hand_x(hand), p.hand_x(1 - hand)
        dtt = t - t0
        x = x0 + (x1 - x0) * dtt / p.flight_s
        y = p.hand_y - p.v0 * dtt + 0.5 * p.g * dtt * dtt
        return x, y

    gapped = list(r.detections)
    for i in range(12):
        # mid-ascent and mid-descent windows, each jittered within its band
        for frac in (0.25 + 0.1 * rng.random(), 0.55 + 0.1 * rng.random()):
            center = r.throw_times[i] + frac * p.flight_s
            frame0 = round(center / dt)
            drop_ts = [(frame0 + k) * dt for k in range(3)]  # 3 consecutive frames
            keep = []
            for d in gapped:
                hit = any(
                    abs(d.t - tt) < 1e-6
                    and abs(d.x - flight_pos(i, tt)[0]) < 1e-4
                    and abs(d.y - flight_pos(i, tt)[1]) < 1e-4
                    for tt in drop_ts
                )
                if not hit:
                    keep.append(d)
            gapped = keep

    starved = extract_arcs(gapped, link_max_dt=0.12)
    assert len(starved) < 11, (
        f"expected the legacy 0.12s budget to still be starved by these gaps "
        f"(got {len(starved)}/12); fixture no longer reproduces the diagnosed failure"
    )

    recovered = extract_arcs(gapped)  # shipped default
    assert len(recovered) >= 11, f"only recovered {len(recovered)}/12 arcs"


def test_false_positives_do_not_create_arcs():
    r = simulate_cascade(n_throws=12, fps=30.0, false_positives_per_frame=0.5, seed=3)
    arcs = extract_arcs(r.detections)
    assert len(arcs) == 12


def test_shuffle_invariance():
    """Property from the spec: results independent of detection ordering/identity."""
    r = simulate_cascade(n_throws=10, fps=30.0, noise=0.002, seed=4)
    arcs_a = extract_arcs(r.detections)
    rng = np.random.default_rng(0)
    shuffled = list(r.detections)
    rng.shuffle(shuffled)
    arcs_b = extract_arcs(shuffled)
    assert arcs_a == arcs_b


def test_drop_scenario_arc_reaches_floor():
    r = simulate_cascade(n_throws=20, fps=30.0, drop_at_throw=8, seed=1)
    arcs = extract_arcs(r.detections)
    p = r.params
    floor_arcs = [a for a in arcs if a.y_at(a.t_end) > p.hand_y + 0.15]
    # the dropped flight continues to the floor; the bounce may add one more
    assert 1 <= len(floor_arcs) <= 2
    main = min(floor_arcs, key=lambda a: a.t_start)
    assert main.t_end == pytest.approx(r.drop_t, abs=0.08)


def test_held_balls_produce_no_arcs():
    r = simulate_cascade(n_throws=8, fps=30.0, include_held=True, seed=5)
    arcs = extract_arcs(r.detections)
    assert len(arcs) == 8  # held-ball (stationary) detections must not become arcs


def test_merge_does_not_fuse_crossing_balls():
    """Regression for the merge gate fusing opposite-direction balls.

    Two fragments from *different* balls can sit on the same y-corridor
    (both pass through similar heights around the same time) while moving
    in opposite x-directions -- the signature of two balls crossing paths
    mid-air, not one continuous flight. Before the x-gate, `_merge_pass`
    accepted a union based on y-rmse alone, which fuses such fragments into
    a single (wrong) arc whenever the shared parabola fits well in y.

    Construct fragment A (t in [0.0, 0.4], x rising at +0.15/s) and
    fragment B (t in [0.53, 0.93], x falling at -0.15/s), a ~0.13s gap
    apart, both sampling y from one shared parabola with its apex
    (y=0.35) at t=0.465 (mid-gap) and both fragments' outer endpoints at
    y=0.65. In y alone this looks exactly like one ball thrown, crossing
    another mid-flight, and landing -- but the opposite x-velocities mean
    it is really two different balls. The result must not contain a
    single arc spanning both fragments' time ranges.
    """
    fps = 30.0
    dt = 1.0 / fps
    apex_t, apex_y = 0.465, 0.35
    ay = 0.30 / apex_t**2  # so y(0) == y(2*apex_t) == 0.65

    def y_at(tt: float) -> float:
        return ay * (tt - apex_t) ** 2 + apex_y

    frame_a = [i * dt for i in range(13)]  # 0.0 .. 0.4, rising x
    frame_b = [0.53 + i * dt for i in range(13)]  # 0.53 .. 0.93, falling x

    dets = []
    idx = 0
    for tt in frame_a:
        dets.append(Detection(frame_idx=idx, t=tt, x=0.3 + 0.15 * tt, y=y_at(tt)))
        idx += 1
    for tt in frame_b:
        dets.append(Detection(frame_idx=idx, t=tt, x=0.42 - 0.15 * (tt - 0.53), y=y_at(tt)))
        idx += 1

    arcs = extract_arcs(dets)

    assert not any(a.t_start < 0.4 and a.t_end > 0.53 for a in arcs), (
        "a single arc spans both fragments -- opposite-direction balls were fused"
    )


@pytest.mark.filterwarnings("ignore::numpy.exceptions.RankWarning")
def test_dense_same_timestamp_clusters_do_not_crash():
    """Regression: extract_arcs used to crash end-to-end with
    numpy.linalg.LinAlgError on real dense footage.

    Detections below are a delta-debug-minimized (69 -> 18 points) slice of
    real detections captured off af2.mp4 at imgsz=960 (frames 860-870,
    t~28.62-28.95s): several frames there each carry >=2-3 overlapping
    candidate ball boxes (a real, dense-detection pattern, not synthetic
    noise). Pre-fix, some EM-refit/merge iteration inside extract_arcs
    isolated a same-timestamp cluster as a fit candidate, and fit_arc's
    weighted np.polyfit crashed with LinAlgError instead of raising the
    documented ValueError for unfittable input -- see tests/test_fit.py's
    test_fit_rejects_all_same_timestamp for the minimal unit-level case.
    The exact output (arcs may legitimately be empty; this slice is too
    short/sparse to pass the usual min_points/min_duration gates) doesn't
    matter here -- only that the call completes without raising.

    T9 (final-review fix wave): this fixture's dense, near-degenerate
    point clusters are EXPECTED to poorly-condition np.polyfit (that's
    what "same-timestamp cluster" means numerically) -- np.polyfit's own
    RankWarning firing here is the well-conditioned-input assumption
    correctly not holding, not a bug this test is checking for. Suppressed
    at the source (this one test, via a marker) rather than globally, so a
    RankWarning anywhere else in the suite still surfaces normally.
    """
    dets = [
        Detection(frame_idx=860, t=28.620932, x=0.758729, y=0.596699, confidence=0.255003),
        Detection(frame_idx=860, t=28.620932, x=0.759045, y=0.599219, confidence=0.105573),
        Detection(frame_idx=861, t=28.654212, x=0.743005, y=0.599745, confidence=0.488518),
        Detection(frame_idx=861, t=28.654212, x=0.749257, y=0.600747, confidence=0.072931),
        Detection(frame_idx=862, t=28.687492, x=0.729324, y=0.590641, confidence=0.112499),
        Detection(frame_idx=862, t=28.687492, x=0.736447, y=0.593838, confidence=0.052603),
        Detection(frame_idx=863, t=28.720773, x=0.723885, y=0.570815, confidence=0.105532),
        Detection(frame_idx=865, t=28.787333, x=0.632862, y=0.567148, confidence=0.185014),
        Detection(frame_idx=865, t=28.787333, x=0.632417, y=0.563010, confidence=0.087624),
        Detection(frame_idx=866, t=28.820613, x=0.632732, y=0.581252, confidence=0.320277),
        Detection(frame_idx=866, t=28.820613, x=0.629926, y=0.580358, confidence=0.089623),
        Detection(frame_idx=866, t=28.820613, x=0.634149, y=0.581015, confidence=0.084602),
        Detection(frame_idx=867, t=28.853893, x=0.638207, y=0.596839, confidence=0.301228),
        Detection(frame_idx=867, t=28.853893, x=0.632152, y=0.596847, confidence=0.289990),
        Detection(frame_idx=867, t=28.853893, x=0.632994, y=0.597118, confidence=0.063283),
        Detection(frame_idx=869, t=28.920453, x=0.640977, y=0.607052, confidence=0.190243),
        Detection(frame_idx=870, t=28.953734, x=0.651686, y=0.602629, confidence=0.434234),
        Detection(frame_idx=870, t=28.953734, x=0.649421, y=0.604342, confidence=0.133543),
    ]

    arcs = extract_arcs(dets)  # must not raise
    assert isinstance(arcs, list)


def test_x_residuals_flag_cross_ball_points():
    r = simulate_cascade(n_throws=1, fps=60.0, seed=0)
    arr = points_array(r.detections)
    arc = fit_arc(arr)
    # a point on the arc's y-parabola but at a wrong x (another ball's position)
    t_mid = (arc.t_start + arc.t_end) / 2
    impostor = np.array([[t_mid, arc.x_at(t_mid) + 0.2, arc.y_at(t_mid), 1.0]])
    res = x_residuals(arc, impostor)
    assert res[0] == pytest.approx(0.2, abs=1e-6)
    assert x_residuals(arc, arr).max() < 0.01  # true points fit x tightly


def test_gravity_prune_kills_chimera_curvature():
    from juggletrack.arcs.extract import _gravity_prune
    from juggletrack.types import Arc

    def arc_with_ay(i, ay):
        return Arc(id=i, t_start=float(i), t_end=float(i) + 1.0, ay=ay, by=-1.1,
                   cy=0.65, bx=0.1, cx=0.4, n_points=20, rmse=0.005)

    arcs = [arc_with_ay(i, 1.0 + 0.03 * i) for i in range(5)] + [arc_with_ay(9, 2.2)]
    kept = _gravity_prune(arcs)
    assert {a.id for a in kept} == {0, 1, 2, 3, 4}
    # fewer than 4 arcs: untouched even with an outlier
    few = [arc_with_ay(0, 1.0), arc_with_ay(1, 2.2)]
    assert _gravity_prune(few) == few


def test_catch_accuracy_seed_sweep():
    """Spec §1 target: catch count within ±1 on >=90% of runs.

    Measured history: pre-hardening baseline 19/20 (Plan 2's merge/witnessed-catch
    fixes had already closed the older 16/20 gap); post-hardening 18/20 with a
    strictly safer failure mode (fused arcs now rejected rather than silently
    netting out); seeds 4/8 are the known residual crossing-fusion gap.

    Plan 5 task 2 (per-frame duplicate-box clustering, default-on in
    AnalyzeConfig, cluster_merge_dist=0.03): briefly regressed this to
    17/20 (seed 2, catches 12->9) because the initial confidence-descending
    sort let a genuine sub-0.03 real-ball crossing merge whenever it landed
    on a tied-confidence pair. Fixed by strict-lower-confidence absorption
    (detect/cluster.py): an anchor at or below a detection's own confidence
    is never an eligible merge target, so two real balls tied at
    confidence=1.0 (every raw sim detection) never merge, period. Verified
    directly: clustering is now a no-op on every seed in this sweep (and
    all 20 in the drift-cohort sweep below) -- 18/20 restored exactly.
    """
    from juggletrack.analyze import analyze_detections

    ok = 0
    failures = []
    for seed in range(20):
        r = simulate_cascade(n_throws=12, fps=30.0, noise=0.004, dropout=0.15, seed=seed)
        sr = analyze_detections(r.detections)
        total = sum(run.catches for run in sr.runs)
        if abs(total - 12) <= 1:
            ok += 1
        else:
            failures.append((seed, total, len(sr.runs)))
    assert ok >= 18, f"catch accuracy {ok}/20 below 90% target; failures: {failures}"


def _inject_static_cluster(fps, n_frames, cx, cy, jitter, density, seed, confidence=0.9):
    """Fake a persistent background false positive: ``density`` detections per
    frame near (cx, cy), jittered by +/-``jitter``, for ``n_frames`` frames.

    (cx, cy) = (0.91, 0.71) is used by the tests below rather than the literal
    (0.9, 0.7) from the field diagnosis: 0.9 sits (to floating-point) exactly
    on a cell=0.03 grid line, so a tight +/-0.005 jitter around it straddles
    two bins -- confirmed empirically to sometimes leave a short-lived,
    under-threshold remainder in one of the split bins on a short sim. Nudging
    off the grid line keeps the injected cluster inside one bin, matching the
    "one static cluster" scenario the field failure actually describes.
    """
    rng = np.random.default_rng(seed)
    dets = []
    for i in range(n_frames):
        t = i / fps
        for _ in range(density):
            x = cx + float(rng.uniform(-jitter, jitter))
            y = cy + float(rng.uniform(-jitter, jitter))
            dets.append(Detection(frame_idx=i, t=t, x=x, y=y, confidence=confidence))
    return dets


def test_static_filter_removes_persistent_cluster():
    """A background false positive parked at one spot for the whole video is
    dropped; real (sweeping) detections survive untouched.

    n_throws=1: with more throws, the ball's own flight retraces the same
    handful of spatial bins on every throw (juggling is periodic -- every
    same-hand throw starts/ends at the exact same hand position), so a real
    bin's own time envelope can legitimately span nearly the whole video too.
    Confirmed empirically (n_throws>=2 already loses >2% of real detections
    to this, and by n_throws=6 loses 100%) -- this is why a single-flight sim
    is used here rather than a longer cascade.
    """
    r = simulate_cascade(n_throws=1, fps=30.0, seed=1)
    real = list(r.detections)
    fps = 30.0
    t_max = max(d.t for d in real)
    n_frames = int(t_max * fps) + 1
    cx, cy = 0.91, 0.71
    static = _inject_static_cluster(fps, n_frames, cx, cy, jitter=0.005, density=1, seed=42)
    contaminated = real + static

    filtered = filter_static_detections(contaminated)

    survivors_near_cluster = [
        d for d in filtered if abs(d.x - cx) < 0.05 and abs(d.y - cy) < 0.05
    ]
    assert survivors_near_cluster == [], "static cluster should be fully removed"

    kept_real = sum(1 for d in filtered if d in real)
    assert kept_real >= 0.99 * len(real)
    # Pinned: this sim is dwell-free enough that the filter costs nothing.
    assert kept_real == len(real)


def test_static_cluster_no_longer_poisons_extraction():
    """Regression for the diagnosed field failure: a persistent static
    high-confidence cluster made the greedy linker in ``_link_fragments``
    absorb real ball detections into junk fragments, corrupting the
    extracted arc count. With ``filter_static_detections`` running first,
    contaminated detections yield the same arc count as clean ones.

    Density here (100 detections/frame, jitter +/-0.04) is far denser/looser
    than the 1-per-frame, +/-0.005 filter test above -- deliberately, to
    reproduce actual poisoning (mirroring the field case's 711 detections
    over 22s at one spot): a lower-density, tight-jitter cluster like the one
    above turns out to be spatially isolated enough from this sim's ball
    path that the greedy linker never competes for it regardless of density,
    so it never poisons extraction even pre-fix. This denser/looser
    construction does: verified pre-fix (filter_static_detections not called
    in extract_arcs) this gives 3 arcs, not 1.
    """
    r = simulate_cascade(n_throws=1, fps=30.0, seed=1)
    real = list(r.detections)
    clean_arcs = extract_arcs(real)
    assert len(clean_arcs) == 1

    fps = 30.0
    t_max = max(d.t for d in real)
    n_frames = int(t_max * fps) + 1
    static = _inject_static_cluster(
        fps, n_frames, cx=0.91, cy=0.71, jitter=0.04, density=100, seed=0
    )
    contaminated = real + static

    poisoned_arcs = extract_arcs(contaminated)
    assert len(poisoned_arcs) == len(clean_arcs)


def test_held_balls_survive_filter():
    """Legitimate hand dwell must never be mistaken for a static false
    positive: it's far under max_span_s=1.5s (hand dwell ~0.2-0.5s; the
    single-throw sim here has no next throw to end the final hold, so it
    runs to the ~0.5s lead-out tail instead -- still well under 1.5s)."""
    r = simulate_cascade(n_throws=1, fps=30.0, include_held=True, seed=5)
    dets = list(r.detections)
    assert filter_static_detections(dets) == dets


def test_slow_drift_junk_cohort_known_gap():
    """GAP CLOSED (turn-4 drift-cohort run gate, events/validate.py): the
    framing fix (gravity floor ay 0.25 -> 0.05, commit 73b0b0f) admitted
    SELF-consistent slow-drift junk cohorts that the old absolute floor
    rejected, because _gravity_prune is median-relative and cannot
    invalidate a cohort that agrees with itself -- arc EXTRACTION still
    fits these three smooth drift paths as plausible arcs (asserted below).

    analyze_detections now drops this run entirely (0 runs), even though its
    3 arcs still exist in sr.arcs (the gate removes runs, not arcs -- see
    analyze._events_from_arcs).

    Note the paths must be spatially SEPARATED and drift briskly: slow
    drifters sharing spatial bins are already killed by
    filter_static_detections (verified while building this fixture), so
    the residual gap is specifically well-separated smooth movers.
    """
    from juggletrack.analyze import analyze_detections
    from juggletrack.types import Detection

    fps = 30.0
    dets = []
    for k in range(3):
        t0 = 0.5 + k * 0.7
        v0, a = 0.12, 0.1  # apex rise v0^2/(4a) = 0.036; fitted ay ~= a
        for i in range(int(1.2 * fps)):
            dt = i / fps
            dets.append(Detection(
                frame_idx=int((t0 + dt) * fps), t=t0 + dt,
                x=0.15 + 0.22 * k + 0.15 * dt,
                y=0.6 - v0 * dt + a * dt * dt,
            ))
    sr = analyze_detections(dets)
    assert len(sr.arcs) == 3, "drift paths still fit as plausible arcs (extraction is unchanged)"
    assert sr.runs == [], "drift-cohort run gate must reject this cohort's run"


def test_split_flight_gets_stitched_back_into_one_arc():
    """Turn-4 split-stitch post-pass (final pass inside extract_arcs): a
    brief mid-flight detection dropout can make the linker/EM machinery mint
    two sequential arcs for what was really one continuous ball flight.

    Reproduce directly: one clean throw, 6 consecutive frames deleted from
    the middle. Empirically the smallest gap that still splits under the
    shipped linker budget (link_max_dt=0.18s, link_max_dist=0.08) -- a 4-5
    frame gap gets bridged by the linker itself before ever reaching the
    stitch pass (verified before writing this test: extract_arcs on this
    same fixture with a 4- or 5-frame gap already returns 1 arc; 6 frames
    is where it splits into 2, which the stitch pass must then repair).
    """
    r = simulate_cascade(n_throws=1, fps=30.0, seed=1)
    dets = sorted(r.detections, key=lambda d: d.t)
    mid = len(dets) // 2
    gapped = dets[:mid] + dets[mid + 6 :]

    arcs = extract_arcs(gapped)

    assert len(arcs) == 1
    assert arcs[0].t_start == pytest.approx(r.throw_times[0], abs=0.05)
    assert arcs[0].t_end == pytest.approx(r.throw_times[0] + r.params.flight_s, abs=0.05)


def test_split_stitch_does_not_fuse_crossing_balls():
    """The split-stitch pass must respect the same crossing-balls signature
    _merge_pass already guards against: opposite-direction x-velocity across
    the gap. Same fixture as test_merge_does_not_fuse_crossing_balls (0.13s
    gap, within the stitch pass's own 0.25s gap budget) -- confirms the new
    post-pass doesn't reopen that hole from a different angle."""
    fps = 30.0
    dt = 1.0 / fps
    apex_t, apex_y = 0.465, 0.35
    ay = 0.30 / apex_t**2

    def y_at(tt: float) -> float:
        return ay * (tt - apex_t) ** 2 + apex_y

    frame_a = [i * dt for i in range(13)]
    frame_b = [0.53 + i * dt for i in range(13)]

    dets = []
    idx = 0
    for tt in frame_a:
        dets.append(Detection(frame_idx=idx, t=tt, x=0.3 + 0.15 * tt, y=y_at(tt)))
        idx += 1
    for tt in frame_b:
        dets.append(Detection(frame_idx=idx, t=tt, x=0.42 - 0.15 * (tt - 0.53), y=y_at(tt)))
        idx += 1

    arcs = extract_arcs(dets)

    assert not any(a.t_start < 0.4 and a.t_end > 0.53 for a in arcs), (
        "a single arc spans both fragments -- opposite-direction balls were fused"
    )


def _mk_arc(id, t_start, t_end, *, ay=0.0, by=0.0, cy=0.5, bx=0.0, cx=0.5,
            n_points=10, rmse=0.005):
    return Arc(id=id, t_start=t_start, t_end=t_end, ay=ay, by=by, cy=cy,
               bx=bx, cx=cx, n_points=n_points, rmse=rmse)


def test_dedup_parallel_arcs_keeps_better_witnessed_near_duplicate():
    """Plan 5 task 2b: two arcs tracing near-identical trajectories (offset
    0.01 in both x and y) over a fully-overlapping window are the same
    physical flight seen through duplicate detection boxes -- exactly the
    box-clustering scenario this stage moves to the arc/trajectory level,
    since on REAL footage duplicate echoes and genuine crossings can't be
    told apart from box confidence alone (see
    docs/superpowers/plans/2026-08-03-meschke-validation-findings.md).
    The better-witnessed arc (higher n_points) must be the one that
    survives, not merely 'one of them'."""
    a = _mk_arc(0, 0.0, 1.0, ay=1.0, by=-1.0, cy=0.60, bx=0.1, cx=0.40,
                n_points=20, rmse=0.005)
    b = _mk_arc(1, 0.0, 1.0, ay=1.0, by=-1.0, cy=0.61, bx=0.1, cx=0.41,
                n_points=8, rmse=0.01)
    out = dedup_parallel_arcs([a, b], overlap_frac=0.5, traj_tol=0.03)
    assert [o.id for o in out] == [0]


def test_dedup_parallel_arcs_keeps_both_crossing_arcs():
    """Two arcs whose x(t) lines cross once mid-window but diverge sharply
    at the edges are two distinct real balls that happened to cross paths,
    not duplicate boxes of one ball -- the mean separation over the WHOLE
    overlap window stays large even though it is ~0 at the single crossing
    instant, so neither may be dropped even though the pair fully overlaps
    in time."""
    a = _mk_arc(0, 0.0, 2.0, bx=0.2, cx=0.3, n_points=20, rmse=0.01)
    b = _mk_arc(1, 0.0, 2.0, bx=-0.2, cx=0.7, n_points=15, rmse=0.01)
    out = dedup_parallel_arcs([a, b], overlap_frac=0.5, traj_tol=0.02)
    assert {o.id for o in out} == {0, 1}


def test_dedup_parallel_arcs_keeps_sequential_non_overlapping_arcs():
    """Two arcs from the same ball on different throws, back to back with
    no temporal overlap, must both survive regardless of trajectory shape
    -- dedup only ever compares arcs that are actually airborne at the same
    time (an identical-shape check here would be meaningless: sequential
    throws of the same juggling pattern legitimately retrace similar
    parabolas)."""
    a = _mk_arc(0, 0.0, 1.0, ay=1.0, by=-1.0, cy=0.6, bx=0.1, cx=0.4,
                n_points=20, rmse=0.005)
    b = _mk_arc(1, 1.5, 2.5, ay=1.0, by=-1.0, cy=0.6, bx=0.1, cx=0.4,
                n_points=20, rmse=0.005)
    out = dedup_parallel_arcs([a, b], overlap_frac=0.5, traj_tol=0.03)
    assert {o.id for o in out} == {0, 1}


def test_dedup_parallel_arcs_keeps_both_at_shipped_crossing_floor():
    """IMPORTANT (2b review): pin the shipped dedup operating point against
    the measured field crossing floor, not just the mechanism-level margin
    `test_dedup_parallel_arcs_keeps_both_crossing_arcs` above already checks
    (that test's crossing pair measures mean separation 0.24 against
    traj_tol=0.02 -- a 12x margin, useful for isolating the mechanism, but
    nowhere near the real operating point).

    The shipped default is `traj_tol=0.15`
    (`juggletrack.arcs.extract.ARC_DEDUP_TRAJ_TOL`, wired through
    `AnalyzeConfig.arc_dedup_traj_tol`). The measured field crossing floor
    -- `ss531_id_989`'s genuine duplicate-vs-crossing arc pairs, the
    tightest real crossing in the 22-video Meschke validation set -- is
    mean separation 0.164 (`.superpowers/sdd/task-2b-report.md` §5/§9),
    only ~9% above 0.15. Construct a synthetic crossing pair whose mean
    per-sample `|dx|+|dy|` over the full overlap window is exactly that
    0.164 floor (two lines through a shared midpoint, opposite slopes,
    sampled at the default 5 points -- same shape as the mechanism test
    above, retuned) and assert BOTH arcs survive `dedup_parallel_arcs` at
    the actual `AnalyzeConfig` defaults (read from a real `AnalyzeConfig()`
    instance, not hardcoded here, so this test tracks the shipped knobs if
    they're ever retuned) -- so any future tol bump past the 0.164 floor
    fails this test loudly instead of silently starting to swallow real
    crossings.
    """
    from juggletrack.analyze import AnalyzeConfig

    cfg = AnalyzeConfig()
    # Two lines crossing at the overlap window's midpoint (t=1 of [0, 2]):
    # mean |dx| over 5 evenly-spaced samples works out to 1.2*m for a
    # crossing slope +/-m through cx=0.5 -+ m (verified: (0.164/1.2)*1.2 ==
    # 0.164). y is identical on both arcs (ay=by=0), so mean |dx|+|dy| ==
    # mean |dx| == 0.164 exactly.
    m = 0.164 / 1.2
    a = _mk_arc(0, 0.0, 2.0, bx=m, cx=0.5 - m, n_points=20, rmse=0.005)
    b = _mk_arc(1, 0.0, 2.0, bx=-m, cx=0.5 + m, n_points=15, rmse=0.01)
    out = dedup_parallel_arcs(
        [a, b], overlap_frac=cfg.arc_dedup_overlap_frac, traj_tol=cfg.arc_dedup_traj_tol,
    )
    assert {o.id for o in out} == {0, 1}


def test_dedup_parallel_arcs_validates_knobs():
    a = _mk_arc(0, 0.0, 1.0)
    with pytest.raises(ValueError, match="overlap_frac"):
        dedup_parallel_arcs([a], overlap_frac=-0.01, traj_tol=0.02)
    with pytest.raises(ValueError, match="overlap_frac"):
        dedup_parallel_arcs([a], overlap_frac=1.01, traj_tol=0.02)
    with pytest.raises(ValueError, match="traj_tol"):
        dedup_parallel_arcs([a], overlap_frac=0.5, traj_tol=-0.01)
    # boundary values stay valid
    assert dedup_parallel_arcs([a], overlap_frac=0.0, traj_tol=0.0) == [a]
    assert dedup_parallel_arcs([a], overlap_frac=1.0, traj_tol=0.0) == [a]


# Frozen copies of the extraction stages as they were before the fast paths
# in extract.py. Those paths must reproduce these exactly (Arc == compares
# every float), with fit_arc still the only producer of a returned Arc.


def _ref_split_ballistic(arr, idxs, resid_tol):
    pieces = []
    cur = []
    for i in idxs:
        cur.append(i)
        if len(cur) >= 4:
            arc = fit_arc(arr[cur])
            x_bad = np.max(x_residuals(arc, arr[cur])) > 2 * resid_tol
            if arc.rmse > resid_tol or x_bad:
                pieces.append(cur[:-1])
                cur = [i]
    if len(cur) >= 4:
        pieces.append(cur)
    return [p for p in pieces if len(p) >= 4]


def _ref_em_assign_refit(arr, arcs, resid_tol):
    if not arcs:
        return []
    t = arr[:, 0]
    best_res = np.full(len(arr), np.inf)
    best_arc = np.full(len(arr), -1, dtype=int)
    for k, arc in enumerate(arcs):
        in_span = (t >= arc.t_start - _EM_TIME_MARGIN) & (t <= arc.t_end + _EM_TIME_MARGIN)
        res_both = np.maximum(y_residuals(arc, arr), x_residuals(arc, arr))
        res = np.where(in_span, res_both, np.inf)
        better = res < best_res
        best_res[better] = res[better]
        best_arc[better] = k
    best_arc[best_res > 2 * resid_tol] = -1

    out = []
    for k in range(len(arcs)):
        member = np.where(best_arc == k)[0]
        if len(member) >= 3:
            try:
                out.append(fit_arc(arr[member]))
            except ValueError:
                continue
    return out


def _ref_merge_pass(arr, arcs, resid_tol):
    arcs = sorted(arcs, key=lambda a: a.t_start)
    t = arr[:, 0]
    n = len(arcs)
    used = [False] * n
    merged = []
    for i in range(n):
        if used[i]:
            continue
        a = arcs[i]
        best_j, best_union = None, None
        for j in range(i + 1, n):
            if used[j]:
                continue
            b = arcs[j]
            if b.t_start - a.t_end >= 0.2:
                break
            if a.bx * b.bx < 0 and abs(a.bx) > 0.02 and abs(b.bx) > 0.02:
                continue
            sel = ((t >= a.t_start) & (t <= a.t_end)) | ((t >= b.t_start) & (t <= b.t_end))
            pts = arr[sel]
            keep_a = y_residuals(a, pts) < 2 * resid_tol
            keep_b = y_residuals(b, pts) < 2 * resid_tol
            pts = pts[keep_a | keep_b]
            if len(pts) < 3:
                continue
            try:
                union = fit_arc(pts)
            except ValueError:
                continue
            if union.rmse > resid_tol:
                continue
            dt = pts[:, 0] - union.t_start
            x_pred = union.bx * dt + union.cx
            x_rmse = float(np.sqrt(np.mean((x_pred - pts[:, 1]) ** 2)))
            if x_rmse > 2 * resid_tol:
                continue
            if best_union is None or union.rmse < best_union.rmse:
                best_j, best_union = j, union
        if best_j is not None:
            used[i] = used[best_j] = True
            merged.append(best_union)
        else:
            merged.append(a)
    return merged


def _sorted_points(dets):
    """The (t, x, y, confidence) array exactly as extract_arcs orders it."""
    arr = points_array(dets)
    return arr[np.lexsort((arr[:, 3], arr[:, 2], arr[:, 1], arr[:, 0]))]


def _crossing_cloud():
    """Two balls on one y-parabola with opposite x-velocities, crossing at t=0.5."""
    dets = []
    for f in range(31):
        t = f / 30.0
        y = 0.65 - 1.2 * t + 1.2 * t * t
        dets.append(Detection(frame_idx=f, t=t, x=0.3 + 0.3 * t, y=y))
        dets.append(Detection(frame_idx=f, t=t, x=0.6 - 0.3 * t, y=y + 0.001))
    return dets


def _duplicate_box_cloud():
    """One to three jittered boxes per detection in the same frame, as dense detectors emit."""
    rng = np.random.default_rng(7)
    base = simulate_cascade(n_throws=8, fps=30.0, noise=0.002, seed=7).detections
    dets = []
    for d in base:
        for _ in range(int(rng.integers(1, 4))):
            dets.append(d.model_copy(update={
                "x": d.x + float(rng.normal(0, 0.003)),
                "y": d.y + float(rng.normal(0, 0.003)),
                "confidence": float(rng.uniform(0.05, 1.0)),
            }))
    return dets


def _low_confidence_cloud():
    """Mostly tiny or zero weights: weighted fits are dominated by a few points."""
    rng = np.random.default_rng(11)
    base = simulate_cascade(n_throws=10, fps=30.0, noise=0.004, dropout=0.1, seed=11).detections
    conf = rng.choice([0.0, 1e-6, 1e-3, 0.02, 0.3, 1.0], size=len(base),
                      p=[0.08, 0.1, 0.2, 0.2, 0.2, 0.22])
    return [d.model_copy(update={"confidence": float(c)}) for d, c in zip(base, conf)]


def _random_cloud():
    """Uniform junk on a 30 fps grid, several points sharing most timestamps."""
    rng = np.random.default_rng(13)
    frames = rng.integers(0, 120, size=500)
    return [
        Detection(frame_idx=int(f), t=f / 30.0, x=float(rng.uniform(0.2, 0.8)),
                  y=float(rng.uniform(0.2, 0.8)), confidence=float(rng.uniform(0.01, 1.0)))
        for f in frames
    ]


_CLOUDS = {
    "clean": lambda: simulate_cascade(n_throws=12, fps=30.0, seed=1).detections,
    "noisy": lambda: simulate_cascade(
        n_throws=12, fps=30.0, noise=0.004, dropout=0.15, seed=2).detections,
    "false-positives": lambda: simulate_cascade(
        n_throws=12, fps=30.0, noise=0.003, false_positives_per_frame=0.5, seed=3).detections,
    "drop": lambda: simulate_cascade(n_throws=12, fps=30.0, drop_at_throw=6, seed=4).detections,
    "crossing": _crossing_cloud,
    "duplicate-boxes": _duplicate_box_cloud,
    "low-confidence": _low_confidence_cloud,
    "random": _random_cloud,
}


def _window_arcs(arr, n, seed):
    """Arcs fit to random time windows of ``arr``: nested, overlapping, touching
    and disjoint spans, many mixing points from several balls."""
    rng = np.random.default_rng(seed)
    t = arr[:, 0]
    arcs = []
    while len(arcs) < n:
        t0 = rng.uniform(t[0], t[-1])
        win = arr[(t >= t0) & (t <= t0 + rng.uniform(0.05, 0.8))]
        if len(win) >= 3 and win[-1, 0] > win[0, 0]:
            arcs.append(fit_arc(win))
    return arcs


def _fragment(seed, n=24, y_noise=0.004, x_noise=0.004, weights=(0.05, 1.0), switch_at=None):
    """One ball's points in strictly increasing t (as _link_fragments emits them),
    optionally jumping onto a second ball's x-line at ``switch_at`` (a crossing)."""
    rng = np.random.default_rng(seed)
    t = 10.0 + np.cumsum(rng.uniform(1 / 60, 1 / 12, n))
    dt = t - t[0]
    y = 0.7 - rng.uniform(1.0, 2.5) * dt + rng.uniform(2.0, 6.0) * dt**2
    x = 0.3 + rng.uniform(-0.4, 0.4) * dt
    if switch_at is not None:
        x[switch_at:] = 0.6 - rng.uniform(0.2, 0.4) * dt[switch_at:]
    lo, hi = weights
    w = rng.uniform(lo, hi, n)
    return np.column_stack([
        t, x + rng.normal(0, x_noise, n), y + rng.normal(0, y_noise, n), w,
    ])


_FRAGMENTS = (
    [_fragment(s) for s in range(20)]
    + [_fragment(100 + s, n=60, y_noise=0.01, x_noise=0.01) for s in range(10)]
    + [_fragment(200 + s, switch_at=12) for s in range(10)]
    + [_fragment(300 + s, y_noise=0.002, weights=(1e-12, 1e-3)) for s in range(10)]
    + [_fragment(400 + s, y_noise=0.0, x_noise=0.0) for s in range(5)]
)


@pytest.mark.filterwarnings("ignore::numpy.exceptions.RankWarning")
@pytest.mark.parametrize("cloud", sorted(_CLOUDS))
@pytest.mark.parametrize("resid_tol", [0.005, 0.02, 0.08])
def test_split_matches_reference_on_linked_fragments(cloud, resid_tol):
    arr = _sorted_points(_CLOUDS[cloud]())
    for frag in _link_fragments(arr, 0.18, 0.08):
        assert _split_ballistic(arr, frag, resid_tol) == _ref_split_ballistic(arr, frag, resid_tol)


@pytest.mark.filterwarnings("ignore::numpy.exceptions.RankWarning")
@pytest.mark.parametrize("resid_tol", [0.001, 0.004, 0.02])
def test_split_matches_reference_on_synthetic_fragments(resid_tol):
    for arr in _FRAGMENTS:
        idxs = list(range(len(arr)))
        assert _split_ballistic(arr, idxs, resid_tol) == _ref_split_ballistic(arr, idxs, resid_tol)


def _prefix_stats(arr):
    """fit_arc's (rmse, max |x residual|) of every prefix of ``arr`` with >= 4 points."""
    out = []
    for k in range(4, len(arr) + 1):
        arc = fit_arc(arr[:k])
        out.append((arc.rmse, float(np.max(x_residuals(arc, arr[:k])))))
    return out


@pytest.mark.parametrize("stat", ["y-rmse", "x-max"])
def test_split_matches_reference_when_a_statistic_ties_its_threshold(stat):
    """Set resid_tol so the reference statistic of one prefix lands exactly on
    its threshold (rmse <= resid_tol keeps, max |x residual| <= 2*resid_tol
    keeps), then one ulp either side. The closed-form statistic differs from
    fit_arc's in the last bits, so only the margin plus the fit_arc fallback
    can keep these decisions identical."""
    for seed in range(30):
        if stat == "y-rmse":
            arr = _fragment(500 + seed, y_noise=0.005, x_noise=1e-4)
            tie = max(r for r, _ in _prefix_stats(arr))
        else:
            arr = _fragment(600 + seed, y_noise=1e-4, x_noise=0.005)
            tie = max(x for _, x in _prefix_stats(arr)) / 2
        idxs = list(range(len(arr)))
        below = np.nextafter(tie, 0.0)
        assert _ref_split_ballistic(arr, idxs, below) != _ref_split_ballistic(arr, idxs, tie)
        for resid_tol in (below, tie, np.nextafter(tie, np.inf)):
            assert (
                _split_ballistic(arr, idxs, resid_tol)
                == _ref_split_ballistic(arr, idxs, resid_tol)
            ), (seed, resid_tol)


def _split_flight(seed, *, y_step, x_step):
    """One flight cut into two arcs whose halves disagree by ``y_step`` in y and
    ``x_step`` in x. Each arc fits its own half almost exactly, so every point
    passes _merge_pass's keep filter and the union's misfit is the step alone.
    Even seeds leave a gap between the halves, odd seeds overlap them."""
    rng = np.random.default_rng(seed)
    t = 20.0 + np.arange(28) / 30.0
    cut = 20.45
    lo_end, hi_start = (cut - 0.04, cut + 0.04) if seed % 2 == 0 else (cut + 0.06, cut - 0.06)
    half_a, half_b = t <= lo_end, t >= hi_start
    dt = t - t[0]
    y = 0.7 - 1.6 * dt + 4.0 * dt**2 + rng.normal(0, 1e-4, len(t))
    x = 0.3 + 0.15 * dt + rng.normal(0, 1e-4, len(t))
    pts_a = np.column_stack([t, x, y, rng.uniform(0.2, 1.0, len(t))])[half_a]
    pts_b = np.column_stack([t, x + x_step, y + y_step, rng.uniform(0.2, 1.0, len(t))])[half_b]
    arr = _sorted_points([
        Detection(frame_idx=0, t=tt, x=xx, y=yy, confidence=c)
        for tt, xx, yy, c in np.vstack([pts_a, pts_b])
    ])
    return arr, [fit_arc(pts_a), fit_arc(pts_b)]


def _union_stats(arr, arcs):
    """fit_arc's (rmse, unweighted x-rmse) of the union _merge_pass forms from two arcs."""
    a, b = arcs
    t = arr[:, 0]
    pts = arr[((t >= a.t_start) & (t <= a.t_end)) | ((t >= b.t_start) & (t <= b.t_end))]
    union = fit_arc(pts)
    dt = pts[:, 0] - union.t_start
    return union.rmse, float(np.sqrt(np.mean((union.bx * dt + union.cx - pts[:, 1]) ** 2)))


@pytest.mark.parametrize("stat", ["y-rmse", "x-rmse"])
def test_merge_matches_reference_when_a_statistic_ties_its_threshold(stat):
    """Set resid_tol so the union's reference statistic lands exactly on its
    acceptance bound (rmse <= resid_tol, x-rmse <= 2*resid_tol), then one ulp
    either side. Only the margin keeps the closed-form pre-reject from
    rejecting a union fit_arc would accept."""
    for seed in range(30):
        if stat == "y-rmse":
            arr, arcs = _split_flight(seed, y_step=0.01, x_step=0.0)
            tie = _union_stats(arr, arcs)[0]
        else:
            arr, arcs = _split_flight(seed, y_step=0.0, x_step=0.02)
            tie = _union_stats(arr, arcs)[1] / 2
        assert len(_ref_merge_pass(arr, arcs, tie)) == 1
        assert len(_ref_merge_pass(arr, arcs, np.nextafter(tie, 0.0))) == 2
        for resid_tol in (np.nextafter(tie, 0.0), tie, np.nextafter(tie, np.inf)):
            assert _merge_pass(arr, arcs, resid_tol) == _ref_merge_pass(arr, arcs, resid_tol), (
                seed, resid_tol,
            )


@pytest.mark.filterwarnings("ignore::numpy.exceptions.RankWarning")
def test_merge_matches_reference_on_two_timestamp_unions():
    """Duplicate boxes can leave a union with only two distinct timestamps.
    polyfit's rank-deficient fit then accepts unions that the normal equations
    cannot even solve, so these must reach fit_arc's own acceptance test."""
    merged = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        t = np.repeat([5.0, 5.0 + 1 / 30], rng.integers(2, 5, size=2))
        pts = np.column_stack([
            t,
            0.4 + 0.3 * (t - 5.0) + rng.normal(0, 0.002, len(t)),
            0.6 - 0.9 * (t - 5.0) + rng.normal(0, 0.002, len(t)),
            rng.uniform(0.05, 1.0, len(t)),
        ])
        arr = _sorted_points([
            Detection(frame_idx=0, t=tt, x=xx, y=yy, confidence=c) for tt, xx, yy, c in pts
        ])
        first = np.flatnonzero(arr[:, 0] == 5.0)
        second = np.flatnonzero(arr[:, 0] > 5.0)
        arcs = [
            fit_arc(arr[np.concatenate([rng.choice(first, 2, replace=False), rng.choice(second, 1)])])
            for _ in range(3)
        ]
        expected = _ref_merge_pass(arr, arcs, 0.02)
        assert _merge_pass(arr, arcs, 0.02) == expected
        merged += len(expected) < len(arcs)
    assert merged > 0


@pytest.mark.filterwarnings("ignore::numpy.exceptions.RankWarning")
@pytest.mark.parametrize("cloud", sorted(_CLOUDS))
@pytest.mark.parametrize("resid_tol", [0.005, 0.02, 0.08])
def test_em_and_merge_match_reference_through_extraction(cloud, resid_tol):
    arr = _sorted_points(_CLOUDS[cloud]())
    arcs = []
    for frag in _link_fragments(arr, 0.18, 0.08):
        for seed in _split_ballistic(arr, frag, resid_tol):
            try:
                arcs.append(fit_arc(arr[seed]))
            except ValueError:
                continue
    for _ in range(3):
        refit = _em_assign_refit(arr, arcs, resid_tol)
        assert refit == _ref_em_assign_refit(arr, arcs, resid_tol)
        merged = _merge_pass(arr, refit, resid_tol)
        assert merged == _ref_merge_pass(arr, refit, resid_tol)
        arcs = merged


def test_em_claim_window_includes_points_exactly_at_the_margin():
    arr = _sorted_points(simulate_cascade(n_throws=1, fps=30.0, seed=0).detections)
    arc = fit_arc(arr)
    lo, hi = arc.t_start - _EM_TIME_MARGIN, arc.t_end + _EM_TIME_MARGIN
    edges = [lo, hi, np.nextafter(lo, -np.inf), np.nextafter(hi, np.inf)]
    extra = np.array([[t, arc.x_at(t), arc.y_at(t), 1.0] for t in edges])
    arr = _sorted_points([
        Detection(frame_idx=0, t=t, x=x, y=y, confidence=c)
        for t, x, y, c in np.vstack([arr, extra])
    ])

    refit = _em_assign_refit(arr, [arc], 0.02)

    assert refit == _ref_em_assign_refit(arr, [arc], 0.02)
    assert [(a.t_start, a.t_end, a.n_points) for a in refit] == [(lo, hi, arc.n_points + 2)]


@pytest.mark.filterwarnings("ignore::numpy.exceptions.RankWarning")
@pytest.mark.parametrize("cloud", sorted(_CLOUDS))
@pytest.mark.parametrize("resid_tol", [0.02, 0.2])
def test_em_and_merge_match_reference_on_arbitrary_spans(cloud, resid_tol):
    arr = _sorted_points(_CLOUDS[cloud]())
    for seed in range(3):
        arcs = _window_arcs(arr, 25, seed)
        assert _em_assign_refit(arr, arcs, resid_tol) == _ref_em_assign_refit(arr, arcs, resid_tol)
        assert _merge_pass(arr, arcs, resid_tol) == _ref_merge_pass(arr, arcs, resid_tol)
