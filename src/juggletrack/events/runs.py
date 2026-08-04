"""Run segmentation: chain event-bearing arcs, score with periodicity (spec §4)."""
from __future__ import annotations

import numpy as np

from juggletrack.events import FLOOR_MARGIN
from juggletrack.events.catches import hand_line_crossings
from juggletrack.events.drops import is_floor_bound
from juggletrack.events.periodicity import periodicity_score
from juggletrack.types import Arc, CatchEvent, Run, ThrowEvent


def estimate_period(throws: list[ThrowEvent], default: float = 0.5) -> float:
    if len(throws) < 3:
        return default
    ts = sorted(e.t for e in throws)
    return float(np.median(np.diff(ts)))


def segment_runs(
    arcs: list[Arc],
    throws: list[ThrowEvent],
    catches: list[CatchEvent],
    hand_line: float,
    *,
    gap_factor: float = 1.3,
    min_arcs: int = 3,
    default_period: float = 0.5,
    # detect_drops already exposes this as a parameter; segment_runs used to
    # hardcode is_floor_bound's own default instead, so the two truncation-
    # relevant predicates (drop candidacy here, run-span truncation there)
    # could silently diverge if floor_margin were ever tuned in one place
    # and not the other. Threading it through keeps them in lockstep.
    floor_margin: float = FLOOR_MARGIN,
) -> list[Run]:
    throw_arc_ids = {e.arc_id for e in throws}
    juggling_arcs = sorted((a for a in arcs if a.id in throw_arc_ids), key=lambda a: a.t_start)
    if not juggling_arcs:
        return []
    period = estimate_period(throws, default_period)

    def arc_end(a: Arc) -> float:
        crossings = hand_line_crossings(a, hand_line)
        return crossings[1] if crossings else a.t_end

    groups: list[list[Arc]] = [[juggling_arcs[0]]]
    cur_end = arc_end(juggling_arcs[0])
    for a in juggling_arcs[1:]:
        if a.t_start > cur_end + gap_factor * period:
            groups.append([a])
            cur_end = arc_end(a)
        else:
            groups[-1].append(a)
            cur_end = max(cur_end, arc_end(a))

    catch_arc_ids = {e.arc_id for e in catches}

    runs: list[Run] = []
    for group in groups:
        if len(group) < min_arcs:
            continue
        ids = {a.id for a in group}
        run_throws = sorted(e.t for e in throws if e.arc_id in ids)
        run_catches = [e for e in catches if e.arc_id in ids]
        start_t = run_throws[0]
        # A thrown arc with no matching catch AND that is floor-bound (kept
        # descending toward the floor rather than ending near the hands,
        # per `is_floor_bound`) is a real miss: everything after it in this
        # group is either the ball still descending past the hand line
        # (uncaught) or arcs from throws the juggler made before noticing
        # the drop. The run itself ended at that first floor-bound miss's
        # scheduled (falling-crossing) time, not at whatever later arc
        # happens to have the largest arc_end.
        #
        # An uncaught arc that is NOT floor-bound is merely unwitnessed --
        # an extraction miss or occlusion cut its catch out of the
        # detection stream while it was still airborne, not evidence the
        # run actually ended there -- and must not truncate the span
        # (measured at scale, ss42_id_011: one unwitnessed catch mid-run
        # truncated a reported end_t to 38.7s while the run's 95 arcs, and
        # its 93 counted catches, span 0..200s). Only fall back to the max
        # over all arcs when no floor-bound miss exists in the group (every
        # throw was caught, or every uncaught arc was merely unwitnessed).
        ordered = sorted(group, key=lambda a: a.t_start)
        first_miss = next(
            (
                a for a in ordered
                if a.id not in catch_arc_ids
                and is_floor_bound(a, hand_line, floor_margin=floor_margin)
            ),
            None,
        )
        end_t = arc_end(first_miss) if first_miss is not None else max(arc_end(a) for a in group)
        p = float(np.median(np.diff(run_throws))) if len(run_throws) >= 3 else None
        # Score periodicity over the arcs' OWN span (min t_start .. max
        # arc-end), not [start_t, end_t]: end_t truncates at the first
        # missed catch, which can be an extraction miss on an otherwise
        # continuous run, not a real drop -- scoring that truncated sliver
        # makes quality garbage (measured: a real run scored 0.004 there vs
        # 0.351 over its own arc span). Lag band is adaptive to the run's
        # own period when known -- [max(0.1, 0.5p), 1.5p] -- because a fixed
        # band tuned to one framing can sit wholly outside another framing's
        # true period (Meschke ground truth: period 1.19-2.89s, outside the
        # fixed default (0.2, 1.2)); falls back to that fixed default only
        # when the run has too few throws for its own period estimate.
        # `arc_end(a)` (the hand-line crossing), not the raw `a.t_end`: a
        # real detection stream can keep tracking an arc past its own
        # analytic flight end (e.g. an uncaught ball still visible falling
        # toward the floor), and that extra tail is not part of "the arc's
        # own [flight] span" this comment means to bound the window by.
        arc_span = (min(a.t_start for a in group), max(arc_end(a) for a in group))
        lag_range = (max(0.1, 0.5 * p), 1.5 * p) if p else (0.2, 1.2)
        score, _ = periodicity_score(group, *arc_span, lag_range=lag_range)
        # periodicity_score returns None when the window can't be judged at
        # all (too short to search any lag in the band) -- distinct from a
        # judged score of 0.0. Run.quality is a plain float, so "can't
        # judge" collapses to 0.0 here; callers cannot distinguish the two
        # from quality alone (0.0 means "unjudgeable OR judged-and-flat").
        quality = score if score is not None else 0.0
        runs.append(Run(
            start_t=start_t, end_t=end_t,
            catches=len(run_catches), throws=len(run_throws),
            arc_ids=sorted(ids), end_reason="stop", period_s=p, quality=quality,
        ))
    return runs
