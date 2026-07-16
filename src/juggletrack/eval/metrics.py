"""Spec §6 metrics: run-boundary IoU, catch-count error, drop precision/recall."""
from __future__ import annotations

from pydantic import BaseModel, Field

from juggletrack.eval.labels import VideoLabels
from juggletrack.types import SessionResult


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
    drop_tol_s: float = 1.0,
    min_match_iou: float = 0.1,
) -> EvalReport:
    pairs = sorted(
        (
            (temporal_iou(p.start_t, p.end_t, lab.start_t, lab.end_t), pi, li)
            for pi, p in enumerate(session.runs)
            for li, lab in enumerate(labels.runs)
        ),
        key=lambda x: -x[0],
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
    frac_iou = sum(1 for m in matches if m.iou >= 0.9) / n_l if n_l else 1.0
    frac_catch = sum(1 for m in matches if m.catch_error <= 1) / n_l if n_l else 1.0

    pred_drops = sorted(d.t for d in session.drops)
    label_drops = sorted(labels.drops)
    drop_pairs = sorted(
        (
            (abs(pt - lt), pi, li)
            for pi, pt in enumerate(pred_drops)
            for li, lt in enumerate(label_drops)
        ),
        key=lambda x: x[0],
    )
    dp_used: set[int] = set()
    dl_used: set[int] = set()
    tp = 0
    for dist, pi, li in drop_pairs:
        if dist > drop_tol_s or pi in dp_used or li in dl_used:
            continue
        dp_used.add(pi)
        dl_used.add(li)
        tp += 1
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
