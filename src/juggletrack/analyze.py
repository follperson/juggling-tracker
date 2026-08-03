"""Detections -> SessionResult: the full event core in one call (spec §3 data flow)."""
from __future__ import annotations

from pydantic import BaseModel, model_validator

from juggletrack.arcs.extract import dedup_parallel_arcs, extract_arcs
from juggletrack.detect.cluster import cluster_detections
from juggletrack.events import CATCH_EXTRAPOLATION_MARGIN
from juggletrack.events.catches import derive_events
from juggletrack.events.drops import detect_drops
from juggletrack.events.handline import estimate_hand_line
from juggletrack.events.runs import segment_runs
from juggletrack.events.validate import is_drift_cohort
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
    # Per-frame duplicate-box clustering radius (see detect/cluster.py's
    # module docstring for the full measured justification): a detector
    # firing 2-3 overlapping boxes for one physical ball mints parallel
    # "ghost" arcs downstream that inflate catch counts. Applied BEFORE
    # filter_static_detections/extract_arcs so every downstream stage --
    # offline and the realtime window path alike -- sees clustered
    # detections. 0.0 disables clustering entirely (identity passthrough,
    # see cluster_detections).
    #
    # 0.0, not the earlier 0.023 (Plan 5 task 2): retuned DOWN in task 2b
    # once arc_dedup_traj_tol below took over most of the duplicate-storm-
    # collapsing job at the ARC level. Box-level clustering at 0.023 could
    # not tell a duplicate echo from a genuine crossing on REAL footage
    # (real crossing balls almost always differ in confidence, so the
    # strict-lower-confidence guard doesn't protect them the way it
    # protects the sim's exactly-tied detections) -- and turning merge_dist
    # up far enough to collapse duplicate storms regressed real crossings
    # badly: ss531_id_005 40->16 catches (oracle 63), ss531_id_989 18->15
    # (oracle 19), ss50505_id_012 155->151 (oracle 157), see
    # docs/superpowers/plans/2026-08-03-meschke-validation-findings.md.
    # Swept {0.010, 0.012, 0.015, 0.018, 0.023} WITH arc dedup active per
    # task 2b's brief, then finer (0.0005 steps, 0.000-0.023) once no value
    # in that grid passed every field gate: results were flat/identical
    # across 0.000-0.006 on every required video (the sim-crossing tie-
    # break makes clustering a no-op on synthetic data regardless of this
    # value, same as before), and 0.0 additionally scored BETTER than 0.023
    # on ss423_id_088 (15, exact oracle match, vs 0.023's 16) and
    # ss531_id_988 (22 vs 0.023's 17, oracle 21) in a broader spot-check.
    #
    # KNOWN COST (measured, out of this task's required-gate scope):
    # ss50505_id_093 (high-pattern family, not gated by this task) relied
    # on box-level clustering's PRE-extraction cleanup -- its raw detection
    # stream is messy enough that arc-level dedup alone cannot recover the
    # same result (135 raw arcs post-extraction; collapsing them after the
    # fact only gets to ~9 catches vs merge_dist=0.023's 38, oracle 42, a
    # real regression). No merge_dist satisfies both this video AND
    # ss531_id_989/ss50505_id_012 (opposite requirements on the SAME knob);
    # since 093 isn't a required gate here, this task accepts that cost
    # rather than reopen the regression-repair gates it IS scored on. See
    # .superpowers/sdd/task-2b-report.md for the full sweep and this
    # trade-off's measurement.
    cluster_merge_dist: float = 0.0
    # Arc-level parallel-arc dedup (Plan 5 task 2b, arcs/extract.py's
    # dedup_parallel_arcs): applied AFTER extract_arcs, BEFORE the
    # event-derivation tail, offline and realtime alike (same integration
    # point cluster_merge_dist used). Field motivation: box-level
    # clustering can't tell a duplicate echo from a genuine crossing on
    # real footage; full arc trajectories can (see dedup_parallel_arcs's
    # own docstring) -- a duplicate storm's arcs trace nearly the same
    # parabola over their WHOLE shared window, while a genuine crossing's
    # arcs only agree briefly.
    #
    # traj_tol=0.15, overlap_frac=0.75 (both widened from an initial
    # 0.02/0.5 guess): the field-video evidence alone (ss3_id_086/
    # ss441_id_089's duplicate pairs top out at mean-diff ~0.097;
    # ss531_id_989's genuine crossing floor is 0.164) suggested traj_tol
    # could go as high as ~0.10-0.16 with overlap_frac=0.5. That FAILED the
    # full test suite: two existing sim fixtures (test_low_gravity_framing_
    # recovers_events's low-g cascade, whose alternating-hand throws
    # overlap and trace mean-diff as low as 0.093 at overlap_frac=0.5; and
    # test_left_edge_guard_prevents_phantom_catches_from_truncated_refit's
    # noisy fixture, mean-diff 0.029 at overlap_frac=0.70) are genuinely
    # DIFFERENT arcs that a traj_tol wide enough for the field videos would
    # incorrectly collapse. Raising overlap_frac to 0.75 excludes both
    # conflicting sim pairs outright (their own overlap_frac, 0.59 and 0.70,
    # sits below the new threshold) regardless of traj_tol, which reopens
    # room to raise traj_tol to 0.15 -- verified safe up to 0.18 (0.15 keeps
    # a margin) against the full suite, and still collapses the bulk of
    # ss3_id_086/ss441_id_089's duplicate pairs (most of which sit above
    # 0.75 overlap_frac; see .superpowers/sdd/task-2b-report.md for the
    # measured pair distributions on both sides of this trade-off).
    #
    # ss3_id_110 is a documented, unresolved exception: its own duplicate-
    # storm arc pairs measure mean-diff >=0.23 (ABOVE ss531_id_989's
    # genuine-crossing floor of 0.164), so no traj_tol can collapse them
    # without also unsafely collapsing real crossings elsewhere -- and
    # exhaustive merge_dist sweeps (0.001 and 0.0005 steps, 0.000-0.023)
    # confirm no merge_dist recovers it without breaking ss531_id_989 or
    # ss50505_id_012 instead. BLOCKED on this one video; see
    # .superpowers/sdd/task-2b-report.md for the measured frontier.
    arc_dedup_overlap_frac: float = 0.75
    arc_dedup_traj_tol: float = 0.15

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

    # Drift-cohort run gate (spec §4 validator, turn-4 cascade-structure-gate
    # bake-off clause A): a self-consistent slow-drift junk cohort can pass
    # every arc-level check (each arc individually looks like a plausible
    # ballistic flight) yet be structurally not-juggling at the RUN level --
    # unidirectional, monotonically marching across the frame, never
    # alternating hands. Reject the run outright; its arcs stay in `arcs`
    # (and in any other surviving run's arc_ids) untouched.
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
    result.meta["n_detections"] = len(dets)
    return result
