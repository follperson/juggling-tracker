import numpy as np
import pytest

from juggletrack.arcs.extract import extract_arcs
from juggletrack.sim import simulate_cascade


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
