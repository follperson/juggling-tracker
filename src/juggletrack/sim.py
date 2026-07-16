"""Deterministic 3-ball-cascade simulator: ground truth for the event core.

Down-positive normalized coordinates. A flight is y(dt) = hand_y - v0*dt + g*dt²/2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel

from juggletrack.types import Detection


class CascadeParams(BaseModel):
    n_balls: int = 3
    period_s: float = 0.45
    dwell_s: float = 0.25
    hand_y: float = 0.65
    hand_sep: float = 0.18
    center_x: float = 0.5
    g: float = 2.0
    floor_y: float = 0.92
    restitution: float = 0.35

    @property
    def flight_s(self) -> float:
        return self.n_balls * self.period_s - self.dwell_s

    @property
    def v0(self) -> float:
        return self.g * self.flight_s / 2.0

    def hand_x(self, hand: int) -> float:
        """hand 0 = left, 1 = right."""
        side = -1.0 if hand == 0 else 1.0
        return self.center_x + side * self.hand_sep / 2.0


class SimResult(BaseModel):
    detections: list[Detection]
    throw_times: list[float]
    catch_times: list[float]
    missed_catch_t: float | None
    drop_t: float | None
    run_start: float
    run_end: float
    params: CascadeParams


@dataclass
class _Flight:
    t0: float
    x0: float
    x1: float
    dur: float
    caught: bool

    def pos(self, t: float, p: CascadeParams) -> tuple[float, float]:
        dt = t - self.t0
        y = p.hand_y - p.v0 * dt + 0.5 * p.g * dt * dt
        x = self.x0 + (self.x1 - self.x0) * dt / self.dur
        return x, y


def simulate_cascade(
    n_throws: int = 20,
    fps: float = 30.0,
    params: CascadeParams | None = None,
    noise: float = 0.0,
    dropout: float = 0.0,
    drop_at_throw: int | None = None,
    include_held: bool = False,
    false_positives_per_frame: float = 0.0,
    seed: int = 0,
) -> SimResult:
    p = params or CascadeParams()
    rng = np.random.default_rng(seed)
    t_first = 0.5  # lead-in before first throw

    # --- schedule flights -------------------------------------------------
    missed_catch_t: float | None = None
    if drop_at_throw is not None:
        missed_catch_t = t_first + drop_at_throw * p.period_s + p.flight_s

    flights: list[_Flight] = []
    throw_times: list[float] = []
    catch_times: list[float] = []
    for i in range(n_throws):
        t0 = t_first + i * p.period_s
        if missed_catch_t is not None and t0 > missed_catch_t:
            break  # juggler noticed the drop and stopped throwing
        hand = i % 2
        caught = drop_at_throw is None or i != drop_at_throw
        flights.append(_Flight(t0=t0, x0=p.hand_x(hand), x1=p.hand_x(1 - hand),
                               dur=p.flight_s, caught=caught))
        throw_times.append(t0)
        if caught:
            catch_times.append(t0 + p.flight_s)

    # --- dropped-ball extension: fall to floor, one bounce, rest ----------
    drop_t: float | None = None
    drop_segments: list[tuple[float, float, _Flight]] = []  # (t_from, t_to, flight-like)
    rest_from: float | None = None
    rest_x: float | None = None
    if drop_at_throw is not None:
        f = flights[drop_at_throw]
        # solve hand_y - v0*dt + g*dt²/2 = floor_y for dt > flight_s
        disc = p.v0**2 + 2.0 * p.g * (p.floor_y - p.hand_y)
        dt_floor = (p.v0 + math.sqrt(disc)) / p.g
        drop_t = f.t0 + dt_floor
        vx = (f.x1 - f.x0) / f.dur
        x_floor = f.x0 + vx * dt_floor
        drop_segments.append((f.t0 + f.dur, drop_t, f))  # continuation below hand line
        # bounce: rises from floor with v_b, returns after 2*v_b/g
        vy_floor = -p.v0 + p.g * dt_floor            # downward speed at impact
        v_b = p.restitution * vy_floor
        bounce_dur = 2.0 * v_b / p.g
        bounce = _Flight(t0=drop_t, x0=x_floor, x1=x_floor + vx * bounce_dur * 0.3,
                         dur=bounce_dur, caught=False)
        rest_from = drop_t + bounce_dur
        rest_x = bounce.x1
        drop_segments.append((drop_t, rest_from, _BounceSeg(bounce, v_b, p)))

    run_start = throw_times[0]
    run_end = missed_catch_t if missed_catch_t is not None else catch_times[-1]
    t_max = (drop_t + 1.0 if drop_t is not None else run_end) + 0.5

    # --- render detections frame by frame ----------------------------------
    detections: list[Detection] = []
    n_frames = int(t_max * fps) + 1
    for f_idx in range(n_frames):
        t = f_idx / fps
        positions: list[tuple[float, float]] = []
        for fl in flights:
            if fl.t0 <= t <= fl.t0 + fl.dur:
                positions.append(fl.pos(t, p))
        for t_from, t_to, seg in drop_segments:
            if t_from < t <= t_to:
                positions.append(seg.pos(t, p))
        if rest_from is not None and t > rest_from:
            positions.append((rest_x, p.floor_y))
        if include_held:
            for hand in (0, 1):
                if _hand_holds_ball(t, hand, flights, p):
                    positions.append((p.hand_x(hand), p.hand_y))
        n_fp = rng.poisson(false_positives_per_frame)
        for _ in range(n_fp):
            positions.append((float(rng.uniform()), float(rng.uniform())))

        for x, y in positions:
            if dropout > 0.0 and rng.random() < dropout:
                continue
            if noise > 0.0:
                x += float(rng.normal(0.0, noise))
                y += float(rng.normal(0.0, noise))
            x, y = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
            detections.append(Detection(frame_idx=f_idx, t=t, x=x, y=y))

    return SimResult(
        detections=detections, throw_times=throw_times, catch_times=catch_times,
        missed_catch_t=missed_catch_t, drop_t=drop_t,
        run_start=run_start, run_end=run_end, params=p,
    )


class _BounceSeg:
    """Bounce parabola starting at the floor with upward speed v_b."""

    def __init__(self, fl: _Flight, v_b: float, p: CascadeParams):
        self.fl, self.v_b, self.p = fl, v_b, p

    def pos(self, t: float, p: CascadeParams) -> tuple[float, float]:
        dt = t - self.fl.t0
        y = p.floor_y - self.v_b * dt + 0.5 * p.g * dt * dt
        x = self.fl.x0 + (self.fl.x1 - self.fl.x0) * dt / self.fl.dur
        return x, min(y, p.floor_y)


def _hand_holds_ball(t: float, hand: int, flights: list[_Flight], p: CascadeParams) -> bool:
    """A hand holds a ball between catching one flight and throwing the next.

    ``next_throw`` must be searched for starting at ``last_catch``, not at the
    query time ``t``: a hand may have already rethrown the ball it caught
    before ``t`` arrives, and its schedule's next *upcoming* throw (relative to
    ``t``) would then belong to a later, unrelated catch. Anchoring the search
    at ``last_catch`` finds that specific catch's own next throw.
    """
    last_catch = None
    for fl in flights:
        thrown_from = 0 if fl.x0 < p.center_x else 1
        caught_by = 1 - thrown_from
        if fl.caught and caught_by == hand and fl.t0 + fl.dur <= t:
            last_catch = max(last_catch or -1.0, fl.t0 + fl.dur)
    if last_catch is None:
        return False
    next_throw = None
    for fl in flights:
        thrown_from = 0 if fl.x0 < p.center_x else 1
        if thrown_from == hand and fl.t0 >= last_catch:
            next_throw = min(next_throw or 1e9, fl.t0)
    return next_throw is None or t < next_throw
