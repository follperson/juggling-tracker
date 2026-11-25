"""Separate a witnessed descending hand carry from an otherwise ballistic fit."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
import math
from statistics import median

from juggletrack.arcs.extract import _GRAVITY_BAND, _prune
from juggletrack.arcs.fit import fit_arc, points_array, x_residuals
from juggletrack.events import FLOOR_MARGIN, catch_witness_margin
from juggletrack.events.catches import hand_line_crossings
from juggletrack.events.drops import is_floor_bound
from juggletrack.types import Arc, Detection


def _distance(a: Detection, b: Detection) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _continuous_witness(
    points: list[Detection],
    dets: list[Detection],
    times: list[float],
    arcs: list[Arc],
    candidate: Arc,
    tol: float,
    max_dt: float,
) -> bool:
    """Reject gaps, rising bounces and distinguishable competing ball paths.

    Multiple detector boxes within one compact residual-sized group can
    describe one ball. A second separated group near the turn is ambiguous.
    An overlapping fitted flight also vetoes a turn at its crossing.
    """
    for a, b in zip(points, points[1:]):
        if b.frame_idx != a.frame_idx + 1 or not 0 < b.t - a.t <= max_dt:
            return False
        if _distance(a, b) >= 2 * tol:
            return False
        if b.y < a.y - tol / 2:
            return False
    for point in points:
        frame = dets[bisect_left(times, point.t) : bisect_right(times, point.t)]
        neighbors = [d for d in frame if _distance(d, point) < 4 * tol]
        if any(_distance(a, b) > 2 * tol for a in neighbors for b in neighbors):
            return False
        for other in arcs:
            if other.id == candidate.id or not other.t_start <= point.t <= other.t_end:
                continue
            if math.hypot(other.x_at(point.t) - point.x, other.y_at(point.t) - point.y) < 2 * tol:
                return False
    return True


def _trim_candidate(
    arc: Arc,
    arcs: list[Arc],
    dets: list[Detection],
    times: list[float],
    hand: float,
    tol: float,
    link_max_dt: float,
    g_range: tuple[float, float],
    min_points: int,
    min_duration: float,
) -> Arc:
    by_frame: dict[int, list[Detection]] = defaultdict(list)
    for d in dets[bisect_left(times, arc.t_start) : bisect_right(times, arc.t_end)]:
        residual = max(abs(d.x - arc.x_at(d.t)), abs(d.y - arc.y_at(d.t)))
        if residual < 2 * tol:
            by_frame[d.frame_idx].append(d)
    samples = [max(boxes, key=lambda d: d.confidence) for boxes in by_frame.values()]
    observed = len(samples)
    if observed < min_points:
        return arc

    # Extra consecutive observations corroborate the turn, without extending
    # the reported flight. Stop at the linker's existing time/distance budget.
    after: dict[int, list[Detection]] = defaultdict(list)
    for d in dets[bisect_right(times, arc.t_end) : bisect_right(times, arc.t_end + link_max_dt)]:
        after[d.frame_idx].append(d)
    previous = samples[-1]
    while previous.frame_idx + 1 in after:
        nearby = min(after[previous.frame_idx + 1], key=lambda d: _distance(d, previous))
        if _distance(nearby, previous) >= 2 * tol:
            break
        samples.append(nearby)
        previous = nearby

    direction = 1 if arc.bx > 0 else -1
    lateral = [
        direction * median(d.x for d in samples[max(0, i - 1) : i + 2]) for i in range(len(samples))
    ]
    possible = [
        i for i, d in enumerate(samples[:observed]) if d.t >= arc.apex_t() and d.y >= hand - 2 * tol
    ]
    if not possible:
        return arc
    pivot = max(possible, key=lateral.__getitem__)
    if pivot < 4 or pivot >= observed - 1 or pivot + 2 >= len(samples):
        return arc
    if samples[pivot].y > hand + FLOOR_MARGIN:
        return arc
    if lateral[pivot] - lateral[0] < 2 * tol:
        return arc
    # Two later samples must move back sideways while continuing down.
    returned = [
        j
        for j in range(pivot + 1, len(samples))
        if lateral[pivot] - lateral[j] >= tol and samples[j].y - samples[pivot].y >= tol
    ]
    if len(returned) < 2:
        return arc
    keep = [d for boxes in by_frame.values() for d in boxes if d.t <= samples[pivot].t]
    retained = points_array(keep)
    if retained[-1, 0] <= retained[0, 0]:
        return arc
    refit = fit_arc(retained, arc.id)
    if x_residuals(refit, retained).max() > 2 * tol:
        return arc
    if not _prune([refit], g_range, tol, min_points, min_duration, max_abs_bx=0.6):
        return arc
    if abs(refit.ay - arc.ay) > arc.ay * _GRAVITY_BAND or is_floor_bound(refit, hand):
        return arc
    # A turn well after the predicted landing can be an ordinary movement
    # of a ball already held. Require the same time agreement we use when
    # relying on the last observed point to witness a catch.
    crossings = hand_line_crossings(refit, hand)
    if crossings is None:
        return arc
    release, landing = crossings
    if abs(samples[pivot].t - landing) > catch_witness_margin(landing - release):
        return arc
    # A sideways deflection alone is also consistent with a missed ball.
    # Catching must slow the descent relative to the incoming ballistic fit,
    # witnessed in two consecutive observations on the same continuous path.
    departed = [
        j for j in range(pivot + 1, len(samples)) if refit.y_at(samples[j].t) - samples[j].y >= tol
    ]
    pair_end = next((b for a, b in zip(departed, departed[1:]) if b == a + 1), None)
    if pair_end is None:
        return arc
    if not _continuous_witness(
        samples[pivot - 1 : max(returned[1], pair_end) + 1],
        dets,
        times,
        arcs,
        arc,
        tol,
        link_max_dt,
    ):
        return arc
    # The ball stayed in the hand through the original endpoint. Other
    # flights still need that height when distinguishing a catch from a miss.
    return refit.model_copy(update={"carry_fit_endpoints_y": arc.peer_hand_heights()})


def trim_hand_carry(
    dets: list[Detection],
    arcs: list[Arc],
    hand: float,
    tol: float,
    *,
    link_max_dt: float = 0.18,
    g_range: tuple[float, float] = (0.1, 8.0),
    min_points: int = 6,
    min_duration: float = 0.15,
) -> list[Arc]:
    """Trim only unambiguous floor-bound flight tails carrying with a hand.

    Runs once after extraction. Caller preserves its pre-trim shared hand
    line so changing a candidate cannot shift unrelated events.
    """
    candidates = [a for a in arcs if is_floor_bound(a, hand) and a.t_start < a.apex_t() < a.t_end]
    if not candidates:
        return arcs
    dets = sorted(dets, key=lambda d: (d.t, -d.confidence, d.x, d.y))
    times = [d.t for d in dets]
    replacements = {
        a.id: _trim_candidate(
            a, arcs, dets, times, hand, tol, link_max_dt, g_range, min_points, min_duration
        )
        for a in candidates
    }
    return [replacements.get(a.id, a) for a in arcs]
