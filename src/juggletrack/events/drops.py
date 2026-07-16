"""Three-signal drop detection (spec §4 — no prior art; first-principles design)."""
from __future__ import annotations

from juggletrack.events import FLOOR_MARGIN
from juggletrack.events.catches import hand_line_crossings
from juggletrack.types import Arc, DropEvent, Run


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
        for cand in run_arcs:
            end_y = cand.y_at(cand.t_end)
            if end_y <= hand_line + floor_margin:
                continue  # ended near the hands: caught, not dropped

            signals = []
            if cand.vy_at(cand.t_end) > 0:
                signals.append("floor_descent")

            for b in arcs:
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
            # "Airborne" here means a *new* throw was made after the miss —
            # i.e. the juggler kept the pattern going. Arcs already in flight
            # from before the miss (thrown pre-drop, landing shortly after
            # due to reaction latency) don't count: they're artifacts of the
            # drop itself, not evidence the pattern survived it.
            others_airborne = any(
                a.id != cand.id and miss_t < a.t_start <= miss_t + collapse_factor * period
                for a in run_arcs
            )
            if not others_airborne:
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
