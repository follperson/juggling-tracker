import numpy as np
import pytest

from juggletrack.analyze import analyze_detections
from juggletrack.events.validate import _sweeps, is_drift_cohort
from juggletrack.sim import CascadeParams, simulate_cascade
from juggletrack.types import Arc, Detection


def _arc(t_start, dur, bx, cx, ay=0.1, by=-0.12, cy=0.6):
    return Arc(
        id=0, t_start=t_start, t_end=t_start + dur, ay=ay, by=by, cy=cy,
        bx=bx, cx=cx, n_points=36, rmse=0.005,
    )


def test_pinned_junk_cohort_is_flagged():
    arcs = [
        _arc(0.5, 1.2, bx=0.15, cx=0.15 + 0.22 * 0),
        _arc(1.2, 1.2, bx=0.15, cx=0.15 + 0.22 * 1),
        _arc(1.9, 1.2, bx=0.15, cx=0.15 + 0.22 * 2),
    ]
    assert is_drift_cohort(arcs) is True


def test_alternating_cascade_is_not_flagged():
    """Every term rejects it: signs alternate, x-origins don't march, x-ranges overlap."""
    arcs = [
        _arc(0.0, 1.1, bx=0.16, cx=0.41),
        _arc(0.45, 1.1, bx=-0.16, cx=0.59),
        _arc(0.90, 1.1, bx=0.16, cx=0.41),
        _arc(1.35, 1.1, bx=-0.16, cx=0.59),
        _arc(1.80, 1.1, bx=0.16, cx=0.41),
        _arc(2.25, 1.1, bx=-0.16, cx=0.59),
    ]
    assert is_drift_cohort(arcs) is False


def test_mixed_sign_sweep_is_not_flagged():
    """Marches (mono ~0.93) and sweeps, so only the middle arc's opposite
    sign (dom2 = 1/3) keeps the gate from firing."""
    arcs = [
        _arc(0.0, 1.0, bx=0.10, cx=0.10),
        _arc(1.0, 1.0, bx=-0.10, cx=0.35),
        _arc(2.0, 1.0, bx=0.10, cx=0.40),
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


AF2_STOCK_ARCS = [
    Arc(id=0, t_start=7.1220, t_end=7.8208, ay=0.9381, by=-0.5772, cy=0.4778,
        bx=0.1494, cx=0.5694, n_points=19, rmse=0.0082),
    Arc(id=1, t_start=7.8541, t_end=8.2535, ay=1.4412, by=-0.6132, cy=0.4521,
        bx=0.1801, cx=0.5838, n_points=13, rmse=0.0006),
    Arc(id=2, t_start=8.4864, t_end=9.0855, ay=1.1122, by=-0.5784, cy=0.4483,
        bx=0.2618, cx=0.6096, n_points=19, rmse=0.0085),
]

REAL_JUGGLING_COHORTS = {
    "af2_stock": AF2_STOCK_ARCS,
    "pxl_stock": [
        _arc(6.0571, 0.3660, bx=0.1954, cx=0.5030),
        _arc(6.3566, 0.3328, bx=0.0392, cx=0.5125),
        _arc(6.6561, 0.5658, bx=0.4671, cx=0.5540),
    ],
    "pxl_191622045_motion": [
        _arc(27.7840, 1.0398, bx=-0.2454, cx=0.6729),
        _arc(28.4495, 1.1646, bx=-0.0969, cx=0.5459),
        _arc(29.1981, 0.8735, bx=-0.2936, cx=0.5086),
    ],
}

PXL_182734164_MOTION_JUNK = [
    _arc(19.3104, 0.7370, bx=-0.0088, cx=0.4464),
    _arc(19.3841, 1.1055, bx=0.0281, cx=0.6081),
    _arc(21.5952, 0.7739, bx=0.0642, cx=0.9394),
]


def test_real_same_direction_run_survives_drift_gate():
    fps = 30.0
    dets = [
        Detection(frame_idx=f, t=f / fps, x=a.x_at(f / fps), y=a.y_at(f / fps))
        for a in AF2_STOCK_ARCS
        for f in range(round(a.t_start * fps), round(a.t_end * fps) + 1)
    ]
    sr = analyze_detections(dets)
    assert [(r.catches, r.throws) for r in sr.runs] == [(3, 3)]


@pytest.mark.parametrize("name", REAL_JUGGLING_COHORTS)
def test_real_juggling_cohorts_are_not_flagged(name):
    assert is_drift_cohort(REAL_JUGGLING_COHORTS[name]) is False


def test_live_window_of_one_hand_throws_is_not_flagged():
    """Real juggling that the pre-sweep gate deleted in live mode.

    Provenance: outputs/meschke-val/ss42_id_010/detections.jsonl (shipped v3
    detector) replayed through RealtimeAnalyzer on its recorded clock at
    commit 324ad61. At feed time 8.967 s the 8 s window held one run of these
    four arcs, and is_drift_cohort returned True. The Meschke labels show
    siteswap 42 there: one hand throws two balls in a fountain while the
    other holds the third. Every dx is negative and cx rises steadily, but
    the x-ranges overlap, so the cohort does not sweep.
    """
    arcs = [
        Arc(id=0, t_start=1.0000, t_end=2.9667, ay=0.1380, by=-0.0437, cy=0.2903,
            bx=-0.0236, cx=0.3521, n_points=61, rmse=0.0095),
        Arc(id=1, t_start=2.2333, t_end=5.1667, ay=0.1292, by=-0.2898, cy=0.5144,
            bx=-0.0282, cx=0.4031, n_points=89, rmse=0.0180),
        Arc(id=2, t_start=3.7667, t_end=7.0333, ay=0.1755, by=-0.5994, cy=0.7973,
            bx=-0.0517, cx=0.4107, n_points=99, rmse=0.0079),
        Arc(id=3, t_start=5.9000, t_end=8.9667, ay=0.1869, by=-0.6156, cy=0.8030,
            bx=-0.0676, cx=0.4290, n_points=93, rmse=0.0062),
    ]
    assert is_drift_cohort(arcs) is False


def test_real_junk_cohort_is_flagged():
    assert is_drift_cohort(PXL_182734164_MOTION_JUNK) is True


@pytest.mark.xfail(strict=True, reason="known gap: a pattern that translates faster than "
                   "its throws move sideways sweeps when only one ball is detected")
def test_panning_juggler_with_one_ball_detected_is_not_flagged():
    hand_sep, flight, period, pan = 0.18, 1.1, 0.45, 0.2
    arcs = []
    for k in (0, 3, 6):
        t, from_right = k * period, k % 2 == 1
        bx = (-hand_sep if from_right else hand_sep) / flight + pan
        arcs.append(_arc(t, flight, bx=bx, cx=0.05 + pan * t + (hand_sep if from_right else 0)))
    assert is_drift_cohort(arcs) is False


@pytest.mark.xfail(strict=True, reason="known gap: junk escapes when two of its pieces "
                   "overlap in x")
def test_junk_with_two_overlapping_pieces_is_flagged():
    arcs = [*PXL_182734164_MOTION_JUNK, _arc(21.70, 0.60, bx=0.06, cx=0.955)]
    assert is_drift_cohort(arcs) is True


def test_stacking_twin_of_junk_fixture_is_not_flagged():
    arcs = [_arc(0.5 + 0.7 * k, 1.2, bx=0.15, cx=0.15 + 0.15 * k) for k in range(3)]
    assert is_drift_cohort(arcs) is False


def test_leftward_junk_fixture_is_flagged():
    arcs = [_arc(0.5 + 0.7 * k, 1.2, bx=-0.15, cx=0.85 - 0.22 * k) for k in range(3)]
    assert is_drift_cohort(arcs) is True


def _spans(ranges, leftward):
    lo, hi = np.array(ranges, dtype=float).T
    return (hi, lo - hi) if leftward else (lo, hi - lo)


@pytest.mark.parametrize("ranges, leftward, expected", [
    pytest.param([(.15, .33), (.37, .55), (.59, .77)], False, True, id="disjoint_every_step"),
    pytest.param([(.67, .85), (.45, .63), (.23, .41)], True, True, id="leftward_sweep"),
    pytest.param([(0, .25), (.25, .5), (.5, .75)], False, False, id="touching_is_overlap"),
    pytest.param([(.10, .30), (.35, .50), (.45, .60)], False, False, id="later_step_overlaps"),
    pytest.param([(.10, .30), (.25, .40), (.70, .80)], False, False, id="earlier_step_overlaps"),
    pytest.param([(0, .1), (.3, .4), (.15, .2), (.6, .7), (.8, .9)], False, False,
                 id="overlaps_hull_not_previous_arc"),
])
def test_sweeps_requires_every_arc_to_clear_the_hull(ranges, leftward, expected):
    assert _sweeps(*_spans(ranges, leftward)) is expected
