from juggletrack.analyze import analyze_detections
from juggletrack.events.validate import is_drift_cohort
from juggletrack.sim import CascadeParams, simulate_cascade
from juggletrack.types import Arc, Detection


def _arc(t_start, dur, bx, cx, ay=0.1, by=-0.12, cy=0.6):
    return Arc(
        id=0, t_start=t_start, t_end=t_start + dur, ay=ay, by=by, cy=cy,
        bx=bx, cx=cx, n_points=36, rmse=0.005,
    )


def test_pinned_junk_cohort_is_flagged():
    """The pinned test_slow_drift_junk_cohort_known_gap shape: 3 arcs marching
    unidirectionally, same sign dx, no alternation -- dom2=1.0, mono=1.0,
    alt2=0.0. Exactly the shape clause A targets."""
    arcs = [
        _arc(0.5, 1.2, bx=0.15, cx=0.15 + 0.22 * 0),
        _arc(1.2, 1.2, bx=0.15, cx=0.15 + 0.22 * 1),
        _arc(1.9, 1.2, bx=0.15, cx=0.15 + 0.22 * 2),
    ]
    assert is_drift_cohort(arcs) is True


def test_alternating_cascade_is_not_flagged():
    """A real 2-hand cascade alternates dx sign every throw (ball goes left,
    then right, then left...) -- dom2 stays far below 1.0 and alt2 is high,
    so clause A must not fire."""
    arcs = [
        _arc(0.0, 1.1, bx=0.16, cx=0.41),
        _arc(0.45, 1.1, bx=-0.16, cx=0.59),
        _arc(0.90, 1.1, bx=0.16, cx=0.41),
        _arc(1.35, 1.1, bx=-0.16, cx=0.59),
        _arc(1.80, 1.1, bx=0.16, cx=0.41),
        _arc(2.25, 1.1, bx=-0.16, cx=0.59),
    ]
    assert is_drift_cohort(arcs) is False


def test_near_vertical_arcs_protected_by_dx_min():
    """Near-vertical throws (siteswap-4 "columns" style: bx ~ 0) must not be
    misread as a drift cohort just because tiny sign noise happens to have a
    dominant sign -- DX_MIN filters them out of "meaningful" entirely, so
    dom2 falls back to 0.0 (no meaningful arcs)."""
    arcs = [_arc(float(i) * 0.5, 0.45, bx=0.001 * (1 if i % 2 == 0 else -1), cx=0.5)
            for i in range(6)]
    assert is_drift_cohort(arcs) is False


def test_too_few_arcs_for_mono_is_not_flagged():
    """mono needs >= 3 points; fewer than that can't be judged monotonic, so
    the gate must not fire regardless of dom2/alt2 (matches min_arcs=3 being
    the pre-existing floor for a run to exist at all)."""
    arcs = [_arc(0.0, 1.0, bx=0.15, cx=0.1), _arc(1.0, 1.0, bx=0.15, cx=0.3)]
    assert is_drift_cohort(arcs) is False


def test_drift_cohort_gate_wired_into_analyze_detections():
    """FLIP of the pinned known-gap fixture (tests/test_extract.py::
    test_slow_drift_junk_cohort_known_gap): analyze_detections must now
    reject the junk cohort's run entirely."""
    fps = 30.0
    dets = []
    for k in range(3):
        t0 = 0.5 + k * 0.7
        v0, a = 0.12, 0.1
        for i in range(int(1.2 * fps)):
            dt = i / fps
            dets.append(Detection(
                frame_idx=int((t0 + dt) * fps), t=t0 + dt,
                x=0.15 + 0.22 * k + 0.15 * dt,
                y=0.6 - v0 * dt + a * dt * dt,
            ))
    sr = analyze_detections(dets)
    assert sr.runs == []
    # The arcs themselves are untouched -- only the run is rejected.
    assert len(sr.arcs) == 3


def test_seed_sweep_regression_at_least_18_of_20():
    """Bake-off regression requirement: the drift-cohort gate must not
    collaterally damage real noisy simulated cascades."""
    ok = 0
    for seed in range(20):
        r = simulate_cascade(n_throws=12, fps=30.0, noise=0.004, dropout=0.15, seed=seed)
        sr = analyze_detections(r.detections)
        total = sum(run.catches for run in sr.runs)
        if abs(total - 12) <= 1:
            ok += 1
    assert ok >= 18


def test_low_g_run_survives_drift_gate():
    r = simulate_cascade(n_throws=12, fps=30.0, seed=1, params=CascadeParams(g=0.35))
    sr = analyze_detections(r.detections)
    assert len(sr.runs) == 1
    assert abs(sr.runs[0].catches - 12) <= 1


def test_near_vertical_columns_pattern_survives_drift_gate():
    """ss441-style single-run scenario (near-vertical siteswap-4 columns):
    hand_sep=0 makes every flight's net x displacement ~0, the exact case
    DX_MIN exists to protect end to end through the real pipeline."""
    columns = CascadeParams(hand_sep=0.0)
    r = simulate_cascade(n_throws=12, fps=30.0, seed=1, params=columns)
    sr = analyze_detections(r.detections)
    assert len(sr.runs) == 1
    assert abs(sr.runs[0].catches - 12) <= 1
