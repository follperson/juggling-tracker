"""Per-frame duplicate-box clustering: collapse near-duplicate detections that a
detector emits for the same physical ball before they ever reach arc extraction.

Field motivation: a detector can fire 2-3 overlapping boxes for one ball in a
single frame (slightly jittered center, varying confidence). Left alone, each
duplicate becomes its own point in the detection cloud; ``extract_arcs``' EM
assigner can split them across sibling arcs that all pass through roughly the
same space, minting parallel "ghost" arcs that inflate downstream catch counts
(measured: ss3_id_086 went from an oracle-matched 24 catches to 57 with
duplicate storms in the raw detections -- see
docs/superpowers/sdd/task-2-report.md).

Default justification (merge_dist=0.03): measured duplicate offsets on
real footage are 0.01-0.03 in normalized units on w~=0.065 boxes
(ss3_id_086 frame 194). Distinct cascade balls approach closer than 0.03 only
in brief crossing instants; losing one detection point per ball during those
instants is absorbed by extract_arcs' EM assigner (verified by
test_duplicate_injection_does_not_inflate_catches's sim-based clone-injection
test), so the merge radius trades a rare, cheap point-loss for eliminating a
much more damaging parallel-arc over-count.
"""
from __future__ import annotations

import math

from juggletrack.types import Detection


def cluster_detections(dets: list[Detection], *, merge_dist: float) -> list[Detection]:
    """Collapse near-duplicate boxes within each frame into one detection.

    Pure and order-independent per frame: the output does not depend on the
    order detections were supplied in, and clustering in one frame never
    affects another (no cross-frame state). ``merge_dist`` of 0.0 disables
    clustering entirely (identity, input order preserved).

    Algorithm (greedy, confidence-first): within each frame, sort detections
    by confidence descending. For each detection (in that order), if its
    center lies within ``merge_dist`` (euclidean, normalized units) of an
    already-kept cluster's center, absorb it into that cluster; otherwise it
    starts a new cluster. A cluster's output detection takes the
    confidence-weighted mean of its members' x/y/w/h, ``confidence`` = the
    max confidence among members, and ``frame_idx``/``t`` preserved from the
    frame (all members share them).
    """
    if merge_dist <= 0.0:
        return list(dets)

    by_frame: dict[int, list[Detection]] = {}
    for d in dets:
        by_frame.setdefault(d.frame_idx, []).append(d)

    out: list[Detection] = []
    for frame_idx in sorted(by_frame):
        out.extend(_cluster_frame(by_frame[frame_idx], merge_dist))
    return out


def _cluster_frame(frame_dets: list[Detection], merge_dist: float) -> list[Detection]:
    # Membership is tested against each cluster's fixed ANCHOR -- the
    # position of the first (highest-confidence) detection that started it
    # -- never a running weighted mean. A moving center lets a chain of
    # small, individually-valid absorptions walk the cluster's effective
    # center past merge_dist of where it started; measured concretely on a
    # duplicate-injection frame (analyze.py's cluster_merge_dist docstring):
    # a real ball's own third clone (independently within merge_dist of the
    # ANCHOR) missed the drifted running mean by a hair and spawned a
    # spurious clone-only cluster with no real detection behind it -- worse
    # than the duplicate-inflation bug this module exists to fix. The
    # confidence-weighted mean is still computed, but only once, for the
    # final output position (_merge) -- never fed back into membership
    # testing.
    ordered = sorted(frame_dets, key=lambda d: d.confidence, reverse=True)
    clusters: list[list[Detection]] = []
    anchors: list[tuple[float, float]] = []

    for d in ordered:
        best_idx = None
        best_dist = merge_dist
        for i, (ax, ay) in enumerate(anchors):
            dist = math.hypot(d.x - ax, d.y - ay)
            if dist <= best_dist:
                best_idx = i
                best_dist = dist
        if best_idx is None:
            clusters.append([d])
            anchors.append((d.x, d.y))
        else:
            clusters[best_idx].append(d)

    return [_merge(members) for members in clusters]


def _merge(members: list[Detection]) -> Detection:
    if len(members) == 1:
        return members[0]
    total_w = sum(m.confidence for m in members)
    if total_w > 0:
        x = sum(m.x * m.confidence for m in members) / total_w
        y = sum(m.y * m.confidence for m in members) / total_w
        w = sum(m.w * m.confidence for m in members) / total_w
        h = sum(m.h * m.confidence for m in members) / total_w
    else:
        n = len(members)
        x = sum(m.x for m in members) / n
        y = sum(m.y for m in members) / n
        w = sum(m.w for m in members) / n
        h = sum(m.h for m in members) / n
    strongest = max(members, key=lambda m: m.confidence)
    return Detection(
        frame_idx=strongest.frame_idx, t=strongest.t,
        x=x, y=y, w=w, h=h, confidence=strongest.confidence,
    )
