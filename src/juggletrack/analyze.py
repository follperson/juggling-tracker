"""Detections -> SessionResult: the full event core in one call (spec §3 data flow)."""
from __future__ import annotations

from pydantic import BaseModel

from juggletrack.arcs.extract import extract_arcs
from juggletrack.events.catches import derive_events
from juggletrack.events.drops import detect_drops
from juggletrack.events.handline import estimate_hand_line
from juggletrack.events.runs import segment_runs
from juggletrack.types import Detection, SessionResult


class AnalyzeConfig(BaseModel):
    g_range: tuple[float, float] = (0.5, 8.0)
    resid_tol: float = 0.02
    min_points: int = 6
    min_duration: float = 0.15
    gap_factor: float = 1.3
    min_arcs: int = 3
    video_end_margin: float = 1.0  # in throw periods


def analyze_detections(
    dets: list[Detection], config: AnalyzeConfig | None = None
) -> SessionResult:
    cfg = config or AnalyzeConfig()
    if not dets:
        return SessionResult()

    arcs = extract_arcs(
        dets, g_range=cfg.g_range, resid_tol=cfg.resid_tol,
        min_points=cfg.min_points, min_duration=cfg.min_duration,
    )
    hand_line = estimate_hand_line(arcs)
    throws, catches = derive_events(arcs, hand_line)
    runs = segment_runs(
        arcs, throws, catches, hand_line,
        gap_factor=cfg.gap_factor, min_arcs=cfg.min_arcs,
    )
    drops, runs = detect_drops(arcs, runs, hand_line)

    t_last = max(d.t for d in dets)
    final: list = []
    for run in runs:
        period = run.period_s or 0.5
        # A "stop"-tagged run's end_t is only trustworthy when it lands at or
        # before the last thing actually observed: a fully-witnessed catch's
        # analytic hand-line-crossing time sits within a frame of the last
        # detection. When the video/detection stream cuts out before the arc
        # reaches the hand line, end_t is extrapolated from the fitted
        # parabola well past t_last — that overshoot (not a small gap toward
        # t_last) is the signal that the ending is unconfirmed.
        if run.end_reason == "stop" and run.end_t - t_last > cfg.video_end_margin * period:
            run = run.model_copy(update={"end_reason": "video_end"})
        final.append(run)

    return SessionResult(
        runs=final, drops=drops, arcs=arcs, hand_line_y=hand_line,
        meta={"n_detections": len(dets), "t_last": t_last},
    )
