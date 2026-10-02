"""Detections -> SessionResult: the full event core in one call (spec §3 data flow)."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from juggletrack.arcs.extract import (
    ARC_DEDUP_OVERLAP_FRAC,
    ARC_DEDUP_TRAJ_TOL,
    dedup_parallel_arcs,
    extract_arcs,
)
from juggletrack.detect.cluster import cluster_detections
from juggletrack.events import CATCH_EXTRAPOLATION_MARGIN
from juggletrack.events.catches import derive_events
from juggletrack.events.drops import detect_drops
from juggletrack.events.handline import estimate_hand_line
from juggletrack.events.runs import segment_runs
from juggletrack.events.validate import is_drift_cohort
from juggletrack.types import Arc, Detection, Run, SessionResult


class AnalyzeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

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
    # Merge near-duplicate detector boxes before extraction. Zero disables it.
    # Tuned on 22 development videos, not a held-out set; stronger merging
    # can erase real balls at crossings. See docs/parameter-tuning.md.
    cluster_merge_dist: float = 0.012
    # Suppress arcs that follow the same trajectory for most of their overlap.
    # These gates complement box clustering; they do not solve every duplicate
    # storm. In particular ss3_id_110 remains a known failure.
    arc_dedup_overlap_frac: float = ARC_DEDUP_OVERLAP_FRAC
    arc_dedup_traj_tol: float = ARC_DEDUP_TRAJ_TOL

    @model_validator(mode="after")
    def _validate_cluster_merge_dist(self) -> "AnalyzeConfig":
        if self.cluster_merge_dist < 0:
            raise ValueError(
                f"cluster_merge_dist must be >= 0 (got {self.cluster_merge_dist})"
            )
        if not 0.0 <= self.arc_dedup_overlap_frac <= 1.0:
            raise ValueError(
                f"arc_dedup_overlap_frac must be in [0, 1] "
                f"(got {self.arc_dedup_overlap_frac})"
            )
        if self.arc_dedup_traj_tol < 0.0:
            raise ValueError(
                f"arc_dedup_traj_tol must be >= 0 (got {self.arc_dedup_traj_tol})"
            )
        return self


def _events_from_arcs(
    arcs: list[Arc], dets_t_last: float, cfg: AnalyzeConfig, *,
    hand_line_override: float | None = None,
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

    ``hand_line_override``: when given, skip re-estimating the hand line
    from ``arcs`` and use this value for every downstream derivation
    instead (throw/catch derivation, run segmentation, drop detection).
    RealtimeAnalyzer uses this so its EMA-smoothed hand line drives EVERY
    derivation in a cycle, not just ``derive_events`` — see
    ``pipeline/realtime.py``'s module docstring (E2 fix) for the bug this
    closes: without a single shared line, the same arc could be counted as
    a catch per one line and a drop per the other.
    """
    hand_line = hand_line_override if hand_line_override is not None else estimate_hand_line(arcs)
    throws, catches = derive_events(arcs, hand_line)
    runs = segment_runs(
        arcs, throws, catches, hand_line,
        gap_factor=cfg.gap_factor, min_arcs=cfg.min_arcs,
    )

    # Drift-cohort run gate (spec §4 validator): a self-consistent slow-drift
    # junk cohort can pass every arc-level check (each arc individually looks
    # like a plausible ballistic flight) yet be one object drifting across the
    # frame rather than juggling. Reject such a run outright; the conditions
    # that define the cohort are in is_drift_cohort's docstring. The run's
    # arcs stay in `arcs` (and in any other surviving run's arc_ids) untouched.
    #
    # This must run BEFORE detect_drops: a rejected run's arcs can still look
    # like a floor-descending "drop" in isolation (e.g. a drift path that
    # never returns to hand height), and detect_drops has no way to know the
    # run it belongs to was just thrown out. Gating first means a junk run's
    # arcs never reach detect_drops at all, so they can't mint a DropEvent
    # that would otherwise leak into SessionResult.drops even though the
    # corresponding run is absent from SessionResult.runs.
    by_id = {a.id: a for a in arcs}
    runs = [
        run for run in runs
        if not is_drift_cohort([by_id[i] for i in run.arc_ids if i in by_id])
    ]

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

    n_raw = len(dets)
    dets = cluster_detections(dets, merge_dist=cfg.cluster_merge_dist)

    arcs = extract_arcs(
        dets, g_range=cfg.g_range, resid_tol=cfg.resid_tol,
        min_points=cfg.min_points, min_duration=cfg.min_duration,
        link_max_dt=cfg.link_max_dt, link_max_dist=cfg.link_max_dist,
        em_iters=cfg.em_iters,
        static_cell=cfg.static_cell, static_max_span_s=cfg.static_max_span_s,
    )
    arcs = dedup_parallel_arcs(
        arcs, overlap_frac=cfg.arc_dedup_overlap_frac, traj_tol=cfg.arc_dedup_traj_tol,
    )
    t_last = max(d.t for d in dets)
    result = _events_from_arcs(arcs, t_last, cfg)
    # n_detections is the RAW input count (pre-clustering), matching
    # oracle_events' meaning of the same key (meschke_import.py) so the two
    # are comparable; n_detections_clustered is the post-clustering count,
    # the merge-rate diagnostic n_detections alone used to conflate with the
    # raw count when this stage briefly rebound `dets` before recording it.
    result.meta["n_detections"] = n_raw
    result.meta["n_detections_clustered"] = len(dets)
    return result
