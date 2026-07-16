import numpy as np
import pytest

from juggletrack.arcs.extract import extract_arcs
from juggletrack.sim import simulate_cascade
from juggletrack.types import Detection


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
