"""Where a run ends, as a caller of analyze_detections sees it.

A run ends at its first confirmed drop. A run with no confirmed drop ends
where its last arc lands, whatever low catches the juggler recovered from on
the way.
"""
import math

import pytest

from juggletrack.analyze import analyze_detections
from juggletrack.events.catches import derive_events
from juggletrack.events.drops import detect_drops, is_floor_bound
from juggletrack.events.runs import segment_runs
from juggletrack.sim import CascadeParams, simulate_cascade
from juggletrack.types import Arc, Detection

FPS = 30.0
SLOW = CascadeParams(period_s=1.5, dwell_s=0.9, g=0.25)


def with_low_catch(sim, k: int, depth: float = 0.11) -> list[Detection]:
    """`sim`'s detections, with flight `k` caught `depth` below the hand line.

    The ball keeps falling past the hand line until the hand stops it, so its
    arc ends floor-bound and gets no catch, while the juggler carries on.
    """
    p = sim.params
    t0 = sim.throw_times[k]
    x0, x1 = p.hand_x(k % 2), p.hand_x(1 - k % 2)
    extra: list[Detection] = []
    y, i = p.hand_y, 0
    while y < p.hand_y + depth:
        i += 1
        dt = p.flight_s + i / FPS
        y = p.hand_y - p.v0 * dt + 0.5 * p.g * dt * dt
        extra.append(Detection(frame_idx=round((t0 + dt) * FPS), t=t0 + dt,
                               x=x0 + (x1 - x0) * dt / p.flight_s, y=y))
    return sim.detections + extra


def reflex_drop(drop_at: int, reaction_s: float = 1.0) -> tuple[list[Detection], float]:
    """A fast cascade whose flight `drop_at` is missed, falls to the floor and
    bounces once. The juggler still makes the throws due within `reaction_s`
    of the miss, catches them, and stops. Returns the detections and the time
    of the missed catch.
    """
    p = CascadeParams()
    missed_t = 0.5 + drop_at * p.period_s + p.flight_s
    t0s = [t for t in (0.5 + i * p.period_s for i in range(60)) if t <= missed_t + reaction_s]
    dt_floor = (p.v0 + math.sqrt(p.v0**2 + 2.0 * p.g * (p.floor_y - p.hand_y))) / p.g
    t_floor = t0s[drop_at] + dt_floor
    v_bounce = p.restitution * (p.g * dt_floor - p.v0)
    bounce_s = 2.0 * v_bounce / p.g

    def flight(i: int, dt: float) -> tuple[float, float]:
        x0, x1 = p.hand_x(i % 2), p.hand_x(1 - i % 2)
        return x0 + (x1 - x0) * dt / p.flight_s, p.hand_y - p.v0 * dt + 0.5 * p.g * dt * dt

    x_floor, _ = flight(drop_at, dt_floor)
    t_last = max(t0s[-1] + p.flight_s, t_floor + bounce_s) + 0.5
    dets: list[Detection] = []
    for f in range(int(t_last * FPS) + 1):
        t = f / FPS
        pos = [
            flight(i, t - t0) for i, t0 in enumerate(t0s)
            if 0.0 <= t - t0 <= (dt_floor if i == drop_at else p.flight_s)
        ]
        db = t - t_floor
        if 0.0 < db <= bounce_s:
            pos.append((x_floor, p.floor_y - v_bounce * db + 0.5 * p.g * db * db))
        elif db > bounce_s:
            pos.append((x_floor, p.floor_y))
        dets += [Detection(frame_idx=f, t=t, x=x, y=y) for x, y in pos]
    return dets, missed_t


@pytest.mark.parametrize("params", [None, SLOW], ids=["fast", "slow"])
def test_recovered_floor_bound_miss_keeps_the_run_span(params):
    """A low catch is not the end of a run. Flight 4 falls past the hand line
    before the hand stops it, so its arc is floor-bound and uncaught, yet the
    juggler keeps throwing and catching. There is no drop, and the run ends
    with its last catch, not at the low one (measured on ss441_id_013: 120 of
    the run's 135 catches fell after an end_t cut at a miss that a new throw
    followed 0.12 s later).

    On the slow cascade the next throw comes 0.9 s after the low catch:
    inside 1.5 of the run's own 1.5 s periods, outside 1.5 of the 0.5 s
    fallback period. So this case also fails if the drop check stops using
    the run's own period.
    """
    sim = simulate_cascade(n_throws=12, fps=FPS, seed=1, params=params)
    sr = analyze_detections(with_low_catch(sim, 4))

    assert len(sr.runs) == 1
    run = sr.runs[0]
    _, catches = derive_events(sr.arcs, sr.hand_line_y)
    caught = {c.arc_id for c in catches}
    misses = [
        a for a in sr.arcs
        if a.id in run.arc_ids and a.id not in caught and is_floor_bound(a, sr.hand_line_y)
    ]
    assert len(misses) == 1, "fixture must leave one floor-bound uncaught arc"
    assert sr.drops == []
    assert run.end_reason == "stop"
    assert run.end_t == pytest.approx(sim.run_end, abs=0.05)


@pytest.mark.parametrize(
    "params, seed, noise",
    [(None, 1, 0.0), (SLOW, 8, 0.004), (SLOW, 10, 0.004), (SLOW, 2, 0.008), (SLOW, 8, 0.008)],
    ids=["fast", "slow-s8-n.004", "slow-s10-n.004", "slow-s2-n.008", "slow-s8-n.008"],
)
def test_run_that_ends_in_a_drop_ends_at_the_missed_catch(params, seed, noise):
    """The simulated juggler misses catch 9, the ball hits the floor and no
    throw follows. On the noisy slow cascade a ball thrown 0.6 s before the
    miss can be first tracked after it, which must not carry the run past
    the drop.
    """
    sim = simulate_cascade(n_throws=24, fps=FPS, seed=seed, noise=noise,
                           drop_at_throw=9, params=params)
    assert max(sim.throw_times) < sim.missed_catch_t

    sr = analyze_detections(sim.detections)

    assert len(sr.drops) == 1
    assert sr.drops[0].t == pytest.approx(sim.missed_catch_t, abs=0.1)
    assert len(sr.runs) == 1
    assert sr.runs[0].end_reason == "drop"
    assert sr.runs[0].end_t == pytest.approx(sim.missed_catch_t, abs=0.1)


@pytest.mark.parametrize("drop_at", [8, 14])
def test_floor_drop_followed_by_two_caught_throws_ends_the_run_at_the_drop(drop_at):
    """A ball is missed, hits the floor and bounces; the juggler makes the
    two throws already under way, catches them and stops. The drop is
    reported, and the run it ended is tagged drop and ends there.
    """
    dets, missed_t = reflex_drop(drop_at)

    sr = analyze_detections(dets)

    assert len(sr.drops) == 1
    assert sr.drops[0].t == pytest.approx(missed_t, abs=0.1)
    assert len(sr.runs) == 1
    assert sr.runs[0].end_reason == "drop"
    assert sr.runs[0].end_t == pytest.approx(missed_t, abs=0.1)


HAND_LINE, PERIOD, FLIGHT_S = 0.65, 0.45, 1.1


def flight(i: int, overshoot: float = 0.0) -> Arc:
    """Thrown at i * PERIOD from the hand line, back at it FLIGHT_S later;
    tracked for `overshoot` seconds past that."""
    t0 = i * PERIOD
    return Arc(id=i, t_start=t0, t_end=t0 + FLIGHT_S + overshoot, ay=1.0, by=-FLIGHT_S,
               cy=HAND_LINE, bx=0.0, cx=0.5, n_points=10, rmse=0.01)


def floor_bounce(id: int, fallen: Arc) -> Arc:
    """The bounce of the ball whose arc `fallen` ends on its way to the floor."""
    t = fallen.t_end
    return Arc(id=id, t_start=t + 0.05, t_end=t + 0.35, ay=1.0, by=-0.2, cy=0.9,
               bx=0.0, cx=0.5, n_points=6, rmse=0.01)


def test_bounce_drop_after_a_recovered_miss_is_reported_and_ends_the_run():
    """A low catch the juggler recovers from, then later a ball that falls to
    the floor and bounces while the pattern goes on. The bounce drop is
    reported and ends the run.

    This test and the next show the one way drop output differs from before
    runs ended at their first confirmed drop. segment_runs used to cut a run
    at its first floor-bound miss, recovered or not, and detect_drops looks
    for a bounce only up to 1.6 s past the run's end_t, so a later bounce
    drop in the same run was never reported. No saved session and no
    simulated stream in the fuzz family shows it.
    """
    arcs = [flight(i) for i in range(16)]
    arcs[2] = flight(2, overshoot=0.2)  # low catch; the next throw is already up
    arcs[10] = flight(10, overshoot=0.25)
    arcs.append(floor_bounce(100, arcs[10]))

    throws, catches = derive_events(arcs, HAND_LINE)
    assert {e.arc_id for e in throws} - {c.arc_id for c in catches} == {2, 10}
    runs = segment_runs(arcs, throws, catches, HAND_LINE)
    drops, runs = detect_drops(arcs, runs, HAND_LINE)

    assert [d.arc_id for d in drops] == [10]
    assert "bounce" in drops[0].signals
    assert len(runs) == 1
    assert runs[0].end_reason == "drop"
    assert runs[0].end_t == pytest.approx(10 * PERIOD + FLIGHT_S)


def test_run_with_two_drops_ends_at_the_first():
    """Two balls fall to the floor and bounce in one run while the pattern
    goes on. Both drops are reported, and the run ends at the first one.
    """
    arcs = [flight(i) for i in range(16)]
    arcs[4] = flight(4, overshoot=0.25)
    arcs[10] = flight(10, overshoot=0.25)
    arcs += [floor_bounce(100, arcs[4]), floor_bounce(101, arcs[10])]

    throws, catches = derive_events(arcs, HAND_LINE)
    runs = segment_runs(arcs, throws, catches, HAND_LINE)
    drops, runs = detect_drops(arcs, runs, HAND_LINE)

    assert [d.arc_id for d in drops] == [4, 10]
    assert len(runs) == 1
    assert runs[0].end_reason == "drop"
    assert runs[0].end_t == pytest.approx(4 * PERIOD + FLIGHT_S)


@pytest.mark.parametrize(
    "kw, low_catch",
    [
        (dict(seed=1), None),
        (dict(seed=1, drop_at_throw=9), None),
        (dict(seed=6, noise=0.004, dropout=0.1), None),
        (dict(seed=3, noise=0.004, dropout=0.1, drop_at_throw=9), None),
        (dict(seed=1), 4),
        (dict(seed=1, noise=0.004, params=SLOW), None),
        (dict(seed=8, noise=0.004, drop_at_throw=9, params=SLOW), None),
        (dict(seed=2, noise=0.008, drop_at_throw=9, params=SLOW), None),
        (dict(seed=1, params=SLOW), 4),
    ],
    ids=[
        "fast", "fast-drop", "fast-noisy", "fast-noisy-drop", "fast-low-catch",
        "slow-noisy", "slow-noisy-drop", "slow-noisier-drop", "slow-low-catch",
    ],
)
def test_run_is_tagged_drop_exactly_when_a_drop_sits_on_its_arcs(kw, low_catch):
    """A run is tagged drop exactly when a drop sits on one of its arcs, and
    then it ends at the first such drop. Any other run ends where its last
    arc lands, so none of its catches comes after its end.
    """
    sim = simulate_cascade(n_throws=24, fps=FPS, **kw)
    dets = sim.detections if low_catch is None else with_low_catch(sim, low_catch)

    sr = analyze_detections(dets)

    assert sr.runs
    assert bool(sr.drops) == (sim.missed_catch_t is not None), "fixture must drop exactly when told"
    _, catches = derive_events(sr.arcs, sr.hand_line_y)
    for run in sr.runs:
        ids = set(run.arc_ids)
        own = [d.t for d in sr.drops if d.arc_id in ids]
        if own:
            assert run.end_reason == "drop"
            assert run.end_t == pytest.approx(min(own))
        else:
            assert run.end_reason != "drop"
            assert [c.t for c in catches if c.arc_id in ids and c.t > run.end_t] == []
