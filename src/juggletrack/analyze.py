"""Detections -> SessionResult: the full event core in one call (spec §3 data flow)."""
from __future__ import annotations

from pydantic import BaseModel

from juggletrack.arcs.extract import extract_arcs
from juggletrack.events import CATCH_EXTRAPOLATION_MARGIN
from juggletrack.events.catches import derive_events
from juggletrack.events.drops import detect_drops
from juggletrack.events.handline import estimate_hand_line
from juggletrack.events.runs import segment_runs
from juggletrack.types import Arc, Detection, Run, SessionResult


class AnalyzeConfig(BaseModel):
    g_range: tuple[float, float] = (0.1, 8.0)
    resid_tol: float = 0.02
    min_points: int = 6
    min_duration: float = 0.15
    # Greedy linker's per-step gates (extract_arcs defaults): how far ahead in
    # time / normalized distance a fragment may look for its next point.
    # Fixed too tight and closer-framed footage with wider per-sample
    # displacement links nothing (see docs/superpowers/plans/
    # 2026-07-16-plan3-flywheel-turn2-findings.md).
    # 0.12 -> 0.18: 0.12s tolerated zero consecutive missed-detection frames
    # at 24fps stride-2 (and only ~2 at 30fps stride-1); turn-4's outdoor
    # holdout showed 3-4 consecutive-frame detection gaps that exceeded the
    # old budget and cost 60% of that video's real misses (see
    # docs/superpowers/sdd/turn4-diagnosis.md). 0.18s tolerates 3 missed
    # frames at 24fps. Mirrors extract_arcs's default -- keep them in sync.
    link_max_dt: float = 0.18
    link_max_dist: float = 0.08
    em_iters: int = 5
    gap_factor: float = 1.3
    min_arcs: int = 3
    # Static-detection pre-filter (extract_arcs's filter_static_detections):
    # drops detections stuck in one cell-sized spatial bin across more than
    # static_max_span_s of video -- background false positives that would
    # otherwise poison the greedy linker (see docs/superpowers/plans/
    # 2026-07-16-plan3-flywheel-turn2-findings.md).
    static_cell: float = 0.03
    static_max_span_s: float = 1.5
    # Seconds of overshoot tolerance before an unconfirmed "stop" is
    # reclassified "video_end". Matches events.catches's
    # CATCH_EXTRAPOLATION_MARGIN so the two checks agree on what counts as
    # "confirmed by real data": a run's end_t comes from the same falling
    # hand-line crossing a catch would need to be witnessed at.
    video_end_margin: float = CATCH_EXTRAPOLATION_MARGIN


def _events_from_arcs(
    arcs: list[Arc], dets_t_last: float, cfg: AnalyzeConfig
) -> SessionResult:
    """Post-extraction tail shared by every arc-set consumer (spec §3 data flow).

    Everything after arc extraction is identical whether the arcs came from
    ``analyze_detections``'s single global extraction or from
    ``oracle_events``'s per-ball extraction + union (see
    ``juggletrack.data.meschke_import``): hand-line estimate, throw/catch
    derivation, run segmentation, drop detection, and the video-end
    reclassification, all keyed only off the arc list and the timestamp of
    the last real observation (``dets_t_last``) — never off how the arcs
    were produced. Extracted so both callers share one code path instead of
    two copies that could drift out of sync.
    """
    hand_line = estimate_hand_line(arcs)
    throws, catches = derive_events(arcs, hand_line)
    runs = segment_runs(
        arcs, throws, catches, hand_line,
        gap_factor=cfg.gap_factor, min_arcs=cfg.min_arcs,
    )
    drops, runs = detect_drops(arcs, runs, hand_line)

    final: list[Run] = []
    for run in runs:
        # A "stop"-tagged run's end_t is only trustworthy when it lands at or
        # before the last thing actually observed: a fully-witnessed catch's
        # analytic hand-line-crossing time sits within a frame of the last
        # detection. When the video/detection stream cuts out before the arc
        # reaches the hand line, end_t is extrapolated from the fitted
        # parabola well past t_last — that overshoot (not a small gap toward
        # t_last) is the signal that the ending is unconfirmed.
        if run.end_reason == "stop" and run.end_t - dets_t_last > cfg.video_end_margin:
            run = run.model_copy(update={"end_reason": "video_end"})
        final.append(run)

    return SessionResult(
        runs=final, drops=drops, arcs=arcs, hand_line_y=hand_line,
        meta={"t_last": dets_t_last},
    )


def analyze_detections(
    dets: list[Detection], config: AnalyzeConfig | None = None
) -> SessionResult:
    cfg = config or AnalyzeConfig()
    if not dets:
        return SessionResult()

    arcs = extract_arcs(
        dets, g_range=cfg.g_range, resid_tol=cfg.resid_tol,
        min_points=cfg.min_points, min_duration=cfg.min_duration,
        link_max_dt=cfg.link_max_dt, link_max_dist=cfg.link_max_dist,
        em_iters=cfg.em_iters,
        static_cell=cfg.static_cell, static_max_span_s=cfg.static_max_span_s,
    )
    t_last = max(d.t for d in dets)
    result = _events_from_arcs(arcs, t_last, cfg)
    result.meta["n_detections"] = len(dets)
    return result
