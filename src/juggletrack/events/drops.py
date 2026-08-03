"""Three-signal drop detection (spec §4 — no prior art; first-principles design)."""
from __future__ import annotations

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
        for cand in run_arcs:
            if not is_floor_bound(cand, hand_line, floor_margin=floor_margin):
                continue  # ended near the hands: caught, not dropped

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

            crossings = hand_line_crossings(cand, hand_line)
            miss_t = crossings[1] if crossings else cand.t_end
            # "Continued" here means a *new* throw was made after the miss —
            # i.e. the juggler kept the pattern going. Arcs already in flight
            # from before the miss (thrown pre-drop, landing shortly after
            # due to reaction latency) don't count: they're artifacts of the
            # drop itself, not evidence the pattern survived it.
            pattern_continued = any(
                a.id != cand.id and miss_t < a.t_start <= miss_t + collapse_factor * period
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
