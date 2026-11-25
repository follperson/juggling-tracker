"""Spec §6 metrics: run-boundary IoU, catch-count error, drop precision/recall."""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import linear_sum_assignment

from juggletrack.eval.labels import VideoLabels
from juggletrack.types import SessionResult

# Spec §6 pass criteria for a matched run, shared by per-video and aggregate scores.
RUN_IOU_PASS = 0.9
CATCH_ERROR_PASS = 1
# Matching gates: a run pair needs IoU above this, a drop pair a time error within this.
RUN_MATCH_MIN_IOU = 0.1
DROP_TOL_S = 1.0


def temporal_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


class RunMatch(BaseModel):
    pred_idx: int
    label_idx: int
    iou: float
    catch_error: int


class EvalReport(BaseModel):
    video: str
    n_labeled_runs: int
    n_pred_runs: int
    matches: list[RunMatch] = Field(default_factory=list)
    unmatched_labeled: list[int] = Field(default_factory=list)
    unmatched_pred: list[int] = Field(default_factory=list)
    frac_runs_iou90: float
    frac_catch_within_1: float
    drop_tp: int
    drop_fp: int
    drop_fn: int
    drop_precision: float
    drop_recall: float


def evaluate_session(
    session: SessionResult,
    labels: VideoLabels,
    *,
    drop_tol_s: float = DROP_TOL_S,
    min_match_iou: float = RUN_MATCH_MIN_IOU,
) -> EvalReport:
    pairs = sorted(
        (
            (temporal_iou(p.start_t, p.end_t, lab.start_t, lab.end_t), pi, li)
            for pi, p in enumerate(session.runs)
            for li, lab in enumerate(labels.runs)
        ),
        key=lambda x: (-x[0], session.runs[x[1]].start_t, labels.runs[x[2]].start_t),
    )
    matches: list[RunMatch] = []
    used_p: set[int] = set()
    used_l: set[int] = set()
    for iou, pi, li in pairs:
        if iou <= min_match_iou or pi in used_p or li in used_l:
            continue
        used_p.add(pi)
        used_l.add(li)
        matches.append(RunMatch(
            pred_idx=pi, label_idx=li, iou=iou,
            catch_error=abs(session.runs[pi].catches - labels.runs[li].catches),
        ))

    n_l = len(labels.runs)
    frac_iou = sum(1 for m in matches if m.iou >= RUN_IOU_PASS) / n_l if n_l else 1.0
    frac_catch = sum(1 for m in matches if m.catch_error <= CATCH_ERROR_PASS) / n_l if n_l else 1.0

    # Match drops one to one: most matches within drop_tol_s first, smallest time error second.
    pred_drops = np.array([d.t for d in session.drops])
    label_drops = np.array(labels.drops)
    dist = np.abs(pred_drops[:, None] - label_drops[None, :])
    # One invalid assignment must cost more than all valid distances combined.
    penalty = min(dist.shape) * drop_tol_s + 1.0
    rows, cols = linear_sum_assignment(np.where(dist <= drop_tol_s, dist, penalty))
    tp = int(np.sum(dist[rows, cols] <= drop_tol_s))
    fp = len(pred_drops) - tp
    fn = len(label_drops) - tp

    return EvalReport(
        video=labels.video,
        n_labeled_runs=n_l,
        n_pred_runs=len(session.runs),
        matches=matches,
        unmatched_labeled=[i for i in range(n_l) if i not in used_l],
        unmatched_pred=[i for i in range(len(session.runs)) if i not in used_p],
        frac_runs_iou90=frac_iou,
        frac_catch_within_1=frac_catch,
        drop_tp=tp, drop_fp=fp, drop_fn=fn,
        drop_precision=tp / (tp + fp) if (tp + fp) else 1.0,
        drop_recall=tp / (tp + fn) if (tp + fn) else 1.0,
    )


def aggregate_scores(reports: list[EvalReport]) -> dict:
    """Pool reports over all labeled runs and drops.

    Unlike EvalReport, which scores an empty set as 1.0, a score with no
    applicable observations is None rather than a claimed perfect result.
    """
    n = sum(r.n_labeled_runs for r in reports)
    catch_ok = sum(m.catch_error <= CATCH_ERROR_PASS for r in reports for m in r.matches)
    iou_ok = sum(m.iou >= RUN_IOU_PASS for r in reports for m in r.matches)
    tp = sum(r.drop_tp for r in reports)
    fp = sum(r.drop_fp for r in reports)
    fn = sum(r.drop_fn for r in reports)
    return {
        "frac_catch_within_1": catch_ok / n if n else None,
        "frac_runs_iou90": iou_ok / n if n else None,
        "drop_tp": tp, "drop_fp": fp, "drop_fn": fn,
        "drop_precision": tp / (tp + fp) if tp + fp else None,
        "drop_recall": tp / (tp + fn) if tp + fn else None,
    }
