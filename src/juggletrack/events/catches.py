"""Sub-frame throw/catch events from arc coefficients (spec §4).

throw = rising crossing of the hand line (smaller quadratic root);
catch = falling crossing (larger root), only if the arc actually terminates
near the hand line — an arc that keeps descending toward the floor is a drop
candidate and yields no catch. A catch must also be witnessed: if the
falling crossing is only reached by extrapolating the fitted parabola well
past the arc's last real detection (truncated video, occlusion), it doesn't
count as a catch either.
"""
from __future__ import annotations

import math

from juggletrack.events import CATCH_EXTRAPOLATION_MARGIN, FLOOR_MARGIN
from juggletrack.types import Arc, CatchEvent, ThrowEvent


def hand_line_crossings(arc: Arc, hand_line: float) -> tuple[float, float] | None:
    """Absolute times where the arc crosses the hand line (rising, falling).

    Falls back to (t_start, t_end) when the fitted arc never reaches the line.
    """
    if arc.ay <= 0:
        return None
    disc = arc.by**2 - 4.0 * arc.ay * (arc.cy - hand_line)
    if disc < 0:
        return arc.t_start, arc.t_end
    root = math.sqrt(disc)
    dt_rise = (-arc.by - root) / (2.0 * arc.ay)
    dt_fall = (-arc.by + root) / (2.0 * arc.ay)
    return arc.t_start + dt_rise, arc.t_start + dt_fall


def derive_events(
    arcs: list[Arc],
    hand_line: float,
    *,
    min_apex_above: float = 0.05,
    floor_margin: float = FLOOR_MARGIN,
) -> tuple[list[ThrowEvent], list[CatchEvent]]:
    throws: list[ThrowEvent] = []
    catches: list[CatchEvent] = []
    for arc in arcs:
        crossings = hand_line_crossings(arc, hand_line)
        if crossings is None:
            continue
        # never rose meaningfully above the hands: bounce or noise
        if arc.apex_y() > hand_line - min_apex_above:
            continue
        t_throw, t_catch = crossings
        throws.append(ThrowEvent(t=t_throw, x=arc.x_at(t_throw), arc_id=arc.id))
        witnessed = t_catch <= arc.t_end + CATCH_EXTRAPOLATION_MARGIN
        if arc.y_at(arc.t_end) <= hand_line + floor_margin and witnessed:
            catches.append(CatchEvent(t=t_catch, x=arc.x_at(t_catch), arc_id=arc.id))

    throws.sort(key=lambda e: e.t)
    catches.sort(key=lambda e: e.t)
    return throws, catches
