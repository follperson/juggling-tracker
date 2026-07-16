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
