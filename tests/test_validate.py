import pytest

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
    collaterally damage real noisy simulated cascades.

    Identical sweep to test_extract.py::test_catch_accuracy_seed_sweep, which
    documents Plan 5 task 2's per-frame duplicate-box clustering: briefly
    regressed this to 17/20 under confidence-descending-sort-only merging,
    restored to 18/20 by strict-lower-confidence absorption (equal-confidence
    detections, e.g. two real balls crossing at confidence=1.0, never merge)
    -- see that test's docstring for the measured root cause."""
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


# Fitted on real af2 juggling (7.1-9.1 s) by the stock COCO detector, which
# finds too few throws for the cascade to visibly alternate.
# Each row is (t_start, t_end, ay, by, cy, bx, cx).
AF2_STOCK_COHORT = [
    (7.1220, 7.8208, 0.9381, -0.5772, 0.4778, 0.1494, 0.5694),
    (7.8541, 8.2535, 1.4412, -0.6132, 0.4521, 0.1801, 0.5838),
    (8.4864, 9.0855, 1.1122, -0.5784, 0.4483, 0.2618, 0.6096),
]


@pytest.mark.xfail(strict=True, reason="drift gate deletes real same-direction runs")
def test_real_same_direction_run_survives_drift_gate():
    """Three real throws that all travel right while their starts step right.
    They match the junk shape on direction and monotonicity, but their x-ranges
    overlap because the balls keep returning to one pattern."""
    fps = 30.0
    dets = [
        Detection(frame_idx=f, t=f / fps, x=bx * (f / fps - t0) + cx,
                  y=ay * (f / fps - t0) ** 2 + by * (f / fps - t0) + cy)
        for t0, t1, ay, by, cy, bx, cx in AF2_STOCK_COHORT
        for f in range(round(t0 * fps), round(t1 * fps) + 1)
    ]
    sr = analyze_detections(dets)
    assert [(r.catches, r.throws) for r in sr.runs] == [(3, 3)]


# Runs the gate rejected on real footage, as (t_start, t_end, bx, cx). af2 and
# pxl come from the stock COCO detector; the 2026-07-16 PXL clips come from the
# motion detector. Each was checked by eye against the video frames.
REAL_JUGGLING_COHORTS = {
    "af2_stock": [(t0, t1, bx, cx) for t0, t1, _, _, _, bx, cx in AF2_STOCK_COHORT],
    "pxl_stock": [
        (6.0571, 6.4231, 0.1954, 0.5030),
        (6.3566, 6.6894, 0.0392, 0.5125),
        (6.6561, 7.2219, 0.4671, 0.5540),
    ],
    "pxl_191622045_motion": [
        (27.7840, 28.8238, -0.2454, 0.6729),
        (28.4495, 29.6141, -0.0969, 0.5459),
        (29.1981, 30.0716, -0.2936, 0.5086),
    ],
}


@pytest.mark.xfail(strict=True, reason="drift gate deletes real same-direction runs")
@pytest.mark.parametrize("name", REAL_JUGGLING_COHORTS)
def test_real_juggling_cohorts_are_not_flagged(name):
    arcs = [_arc(t0, t1 - t0, bx=bx, cx=cx) for t0, t1, bx, cx in REAL_JUGGLING_COHORTS[name]]
    assert is_drift_cohort(arcs) is False


def test_real_junk_cohort_is_flagged():
    """Motion-detector streaks on PXL_20260716_182734164, 19.3-22.4 s, scattered
    across the frame as a second person walks in. Each arc lands on new ground."""
    arcs = [_arc(t0, t1 - t0, bx=bx, cx=cx) for t0, t1, bx, cx in [
        (19.3104, 20.0474, -0.0088, 0.4464),
        (19.3841, 20.4896, 0.0281, 0.6081),
        (21.5952, 22.3691, 0.0642, 0.9394),
    ]]
    assert is_drift_cohort(arcs) is True
