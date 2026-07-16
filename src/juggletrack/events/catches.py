"""Sub-frame throw/catch events from arc coefficients (spec §4).

throw = rising crossing of the hand line (smaller quadratic root);
catch = falling crossing (larger root), only if the arc actually terminates
near the hand line — an arc that keeps descending toward the floor is a drop
candidate and yields no catch.
"""
from __future__ import annotations

import math

from juggletrack.types import Arc, CatchEvent, ThrowEvent


def derive_events(
    arcs: list[Arc],
    hand_line: float,
    *,
    min_apex_above: float = 0.05,
    floor_margin: float = 0.10,
) -> tuple[list[ThrowEvent], list[CatchEvent]]:
    throws: list[ThrowEvent] = []
    catches: list[CatchEvent] = []
    for arc in arcs:
        if arc.ay <= 0:
            continue
        if arc.apex_y() > hand_line - min_apex_above:
            continue  # never rose meaningfully above the hands: bounce or noise

        disc = arc.by**2 - 4.0 * arc.ay * (arc.cy - hand_line)
        if disc >= 0:
            root = math.sqrt(disc)
            dt_throw = (-arc.by - root) / (2.0 * arc.ay)
            dt_catch = (-arc.by + root) / (2.0 * arc.ay)
        else:  # fitted arc sits entirely above the hand line: fall back to span
            dt_throw, dt_catch = 0.0, arc.t_end - arc.t_start

        t_throw = arc.t_start + dt_throw
        throws.append(ThrowEvent(t=t_throw, x=arc.x_at(t_throw), arc_id=arc.id))

        if arc.y_at(arc.t_end) <= hand_line + floor_margin:
            t_catch = arc.t_start + dt_catch
            catches.append(CatchEvent(t=t_catch, x=arc.x_at(t_catch), arc_id=arc.id))

    throws.sort(key=lambda e: e.t)
    catches.sort(key=lambda e: e.t)
    return throws, catches
