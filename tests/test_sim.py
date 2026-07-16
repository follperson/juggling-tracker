from collections import Counter

import numpy as np
import pytest

from juggletrack.sim import CascadeParams, simulate_cascade


def test_derived_params():
    p = CascadeParams()
    assert p.flight_s == pytest.approx(3 * 0.45 - 0.25)  # 1.10
    assert p.v0 == pytest.approx(2.0 * 1.10 / 2.0)       # 1.10


def test_clean_run_ground_truth():
    r = simulate_cascade(n_throws=12, fps=30.0, seed=1)
    assert len(r.throw_times) == 12
    assert len(r.catch_times) == 12
    p = r.params
    for th, ca in zip(r.throw_times, r.catch_times):
        assert ca == pytest.approx(th + p.flight_s)
    assert r.throw_times[1] - r.throw_times[0] == pytest.approx(p.period_s)
    assert r.drop_t is None and r.missed_catch_t is None
    assert r.run_start == pytest.approx(r.throw_times[0])
    assert r.run_end == pytest.approx(r.catch_times[-1])


def test_detections_geometry():
    r = simulate_cascade(n_throws=12, fps=30.0, seed=1)
    p = r.params
    assert len(r.detections) > 0
    xs = np.array([d.x for d in r.detections])
    ys = np.array([d.y for d in r.detections])
    assert xs.min() >= 0.0 and xs.max() <= 1.0
    assert ys.min() >= 0.0 and ys.max() <= 1.0
    # airborne balls never go below hand level in a clean run
    assert ys.max() <= p.hand_y + 0.02
    # apex reached: hand_y - v0^2/(2g) = 0.65 - 0.3025
    assert ys.min() == pytest.approx(p.hand_y - p.v0**2 / (2 * p.g), abs=0.02)


def test_airborne_ball_count_bounded():
    r = simulate_cascade(n_throws=20, fps=30.0, seed=1)
    from collections import Counter

    per_frame = Counter(d.frame_idx for d in r.detections)
    assert max(per_frame.values()) <= 3  # never more than 3 balls
    # cascade with flight 1.1s / period 0.45s keeps 2-3 airborne mid-run
    mid = [c for f, c in per_frame.items() if 3.0 < f / 30.0 < 6.0]
    assert min(mid) >= 2


def test_drop_injection():
    r = simulate_cascade(n_throws=20, fps=30.0, drop_at_throw=8, seed=1)
    assert r.missed_catch_t == pytest.approx(r.throw_times[8] + r.params.flight_s)
    assert r.drop_t is not None and r.drop_t > r.missed_catch_t
    # juggler stops: fewer throws than requested
    assert len(r.throw_times) < 20
    # dropped ball produces floor-region detections
    floor_dets = [d for d in r.detections if d.y > r.params.hand_y + 0.1]
    assert len(floor_dets) > 0
    assert r.run_end == pytest.approx(r.missed_catch_t)


def test_noise_dropout_determinism():
    a = simulate_cascade(n_throws=10, noise=0.005, dropout=0.2, seed=7)
    b = simulate_cascade(n_throws=10, noise=0.005, dropout=0.2, seed=7)
    assert a.detections == b.detections  # same seed → identical
    clean = simulate_cascade(n_throws=10, seed=7)
    assert len(a.detections) < len(clean.detections)  # dropout removed some


def test_false_positives():
    r = simulate_cascade(n_throws=10, false_positives_per_frame=0.5, seed=3)
    clean = simulate_cascade(n_throws=10, seed=3)
    assert len(r.detections) > len(clean.detections)


def test_include_held_emits_stationary_hand_detections():
    """include_held=True is the ground-truth oracle later tasks rely on: held
    balls must show up as stationary detections pinned to a hand position."""
    fps = 30.0
    r_held = simulate_cascade(n_throws=8, fps=fps, include_held=True, seed=5)
    r_air = simulate_cascade(n_throws=8, fps=fps, include_held=False, seed=5)
    p = r_held.params

    assert len(r_held.detections) > len(r_air.detections)

    held_per_frame = Counter(d.frame_idx for d in r_held.detections)
    air_per_frame = Counter(d.frame_idx for d in r_air.detections)

    # mid-run frames where held balls fill airborne gaps up to the full
    # 3-ball count (airborne alone dips below 3 during hand dwell time).
    mid_frames = [f for f in held_per_frame if 2.0 <= f / fps <= 4.0]
    assert mid_frames
    fill_in_frames = [
        f for f in mid_frames
        if held_per_frame[f] == 3 and air_per_frame.get(f, 0) < 3
    ]
    assert fill_in_frames, "expected held detections to fill airborne gaps to 3 mid-run"

    # Every detection in the held run that isn't explainable as an airborne
    # detection (per-frame (x, y) multiset diff) must sit exactly at a hand.
    hand_positions = [(p.hand_x(0), p.hand_y), (p.hand_x(1), p.hand_y)]
    held_by_frame: dict[int, Counter] = {}
    air_by_frame: dict[int, Counter] = {}
    for d in r_held.detections:
        held_by_frame.setdefault(d.frame_idx, Counter())[(d.x, d.y)] += 1
    for d in r_air.detections:
        air_by_frame.setdefault(d.frame_idx, Counter())[(d.x, d.y)] += 1

    checked_any = False
    for f_idx, held_counts in held_by_frame.items():
        extra = held_counts - air_by_frame.get(f_idx, Counter())
        for (x, y) in extra:
            checked_any = True
            assert any(
                x == pytest.approx(hx) and y == pytest.approx(hy)
                for hx, hy in hand_positions
            ), f"extra held detection at ({x}, {y}) on frame {f_idx} is not a hand position"
    assert checked_any, "expected at least one held-only detection to verify"
