"""Three-signal drop detection (spec §4 — no prior art; first-principles design)."""
from __future__ import annotations

from statistics import median

from juggletrack.events import FLOOR_MARGIN
from juggletrack.events.catches import hand_line_crossings
from juggletrack.types import Arc, DropEvent, Run


def is_floor_bound(arc: Arc, hand_line: float, *, floor_margin: float = FLOOR_MARGIN) -> bool:
    """True if `arc` ends past the floor margin below the hand line.

    y is normalized image coordinates: larger y = lower in frame. An arc
    that ends at or above `hand_line + floor_margin` terminated near the
    hands (caught, or at least not evidence of a floor-ward miss). One that
    ends below that threshold kept descending toward the floor -- the shape
    a real drop leaves, as opposed to an arc that's merely unwitnessed past
    that point (extraction miss, occlusion) while still airborne.

    Single source of truth for this predicate: shared by `detect_drops`
    (drop candidacy) and `segment_runs` (which uncaught arcs may truncate a
    run's end_t).
    """
    return arc.y_at(arc.t_end) > hand_line + floor_margin


def detect_drops(
    arcs: list[Arc],
    runs: list[Run],
    hand_line: float,
    *,
    floor_margin: float = FLOOR_MARGIN,
    bounce_window: float = 0.6,
    bounce_x_tol: float = 0.10,
    collapse_factor: float = 1.5,
) -> tuple[list[DropEvent], list[Run]]:
    drops: list[DropEvent] = []
    by_id = {a.id: a for a in arcs}

    for run in runs:
        period = run.period_s or 0.5
        run_arcs = [by_id[i] for i in run.arc_ids if i in by_id]
        candidates = [a for a in run_arcs if is_floor_bound(a, hand_line, floor_margin=floor_margin)]
        if not candidates:
            continue
        # Bounce candidates only need to be considered within this run's own
        # span (padded generously on both sides for slack in boundary
        # estimates): a bounce, by definition, follows shortly after a
        # candidate that itself belongs to this run. Scanning the full
        # global arc list here instead is unnecessary and, for long
        # sessions with many runs, wastefully quadratic.
        bounce_candidates = [
            b for b in arcs
            if run.start_t - 1.0 <= b.t_start <= run.end_t + bounce_window + 1.0
        ]
        peer_heights = {a.id: a.peer_hand_heights() for a in run_arcs}
        fall_ends = [peer_heights[a.id][1] for a in run_arcs
                     if a.carry_fit_endpoints_y is not None or a.vy_at(a.t_end) > 0]
        lowest_end = max(fall_ends, default=hand_line)
        hand_heights = [y for y in fall_ends if y <= hand_line + floor_margin
                        or y < lowest_end - floor_margin]
        # A common low endpoint may be the floor. Bound hand depth using
        # ordinary landings and rising starts. When no landing distinguishes
        # the hand, starts alone are evidence only for a group already in
        # flight together. A longer sequence may repeatedly start tracking
        # each ball late in the air and end at an ordinary catch.
        if hand_heights or max(a.t_start for a in run_arcs) < min(a.t_end for a in run_arcs):
            hand_heights.extend(peer_heights[a.id][0] for a in run_arcs
                                if a.carry_fit_endpoints_y is not None or a.vy_at(a.t_start) < 0)
        deepest_hand = max(hand_heights, default=float("inf"))
        for cand in candidates:
            # A flight that is only partly tracked can have one endpoint up in
            # the air. Use its lower endpoint as the hand observation. Exclude
            # the candidate itself, whose last observation may be on the floor.
            heights = [max(ends) for arc_id, ends in peer_heights.items() if arc_id != cand.id]
            miss_line = min(median(heights), deepest_hand) if len(heights) >= 3 else hand_line
            # A crossing at the last observation, within the fit error, also
            # describes an ordinary catch. Confirm the descent past this line
            # in the observed arc before putting a missed catch on the clock.
            if cand.y_at(cand.t_end) - miss_line <= cand.rmse:
                continue
            signals = []
            if cand.vy_at(cand.t_end) > 0:
                signals.append("floor_descent")

            for b in bounce_candidates:
                if b.id == cand.id:
                    continue
                starts_after = 0.0 < b.t_start - cand.t_end < bounce_window
                near_x = abs(b.x_at(b.t_start) - cand.x_at(cand.t_end)) < bounce_x_tol
                stays_low = b.apex_y() > hand_line
                if starts_after and near_x and stays_low:
                    signals.append("bounce")
                    break

            crossings = hand_line_crossings(cand, miss_line)
            miss_t = crossings[1] if crossings else cand.t_end
            shared = hand_line_crossings(cand, hand_line)
            # Keep the shared event clock when the two estimated crossings
            # differ by at most half an observed sampling interval.
            half_sample = cand.duration() / (2 * (cand.n_points - 1)) if cand.n_points > 1 else 0.0
            if shared and abs(shared[1] - miss_t) <= half_sample:
                miss_t = shared[1]
            # "Continued" here means a *new* throw was made after the miss —
            # i.e. the juggler kept the pattern going. Arcs already in flight
            # from before the miss (thrown pre-drop, landing shortly after
            # due to reaction latency) don't count: they're artifacts of the
            # drop itself, not evidence the pattern survived it.
            # A new start within the crossing's sampling uncertainty gives
            # insufficient evidence that the pattern stopped at this instant.
            pattern_continued = any(
                a.id != cand.id and miss_t - half_sample < a.t_start <= miss_t + collapse_factor * period
                for a in run_arcs
            )
            if not pattern_continued:
                signals.append("periodicity_collapse")

            if len(signals) >= 2:
                drops.append(DropEvent(
                    t=miss_t, x=cand.x_at(miss_t), arc_id=cand.id, signals=signals,
                ))

    drops.sort(key=lambda d: d.t)
    updated: list[Run] = []
    for run in runs:
        period = run.period_s or 0.5
        ends_in_drop = any(abs(d.t - run.end_t) <= period for d in drops)
        updated.append(run.model_copy(update={"end_reason": "drop"}) if ends_in_drop else run)
    return drops, updated
