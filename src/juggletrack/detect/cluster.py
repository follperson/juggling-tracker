"""Per-frame duplicate-box clustering: collapse near-duplicate detections that a
detector emits for the same physical ball before they ever reach arc extraction.

Field motivation: a detector can fire 2-3 overlapping boxes for one ball in a
single frame (slightly jittered center, varying confidence). Left alone, each
duplicate becomes its own point in the detection cloud; ``extract_arcs``' EM
assigner can split them across sibling arcs that all pass through roughly the
same space, minting parallel "ghost" arcs that inflate downstream catch counts
(measured: ss3_id_086 went from an oracle-matched 24 catches to 57 with
duplicate storms in the raw detections).

Plan 5 task 2b update: AnalyzeConfig.cluster_merge_dist's shipped default
moved from 0.023 (below) to 0.0 -- this per-frame box-level mechanism
stays available (and still fires whenever merge_dist > 0) but is no longer
relied on by default: real ball crossings almost always differ in
confidence, so no merge_dist here can collapse duplicate-box storms
without also risking real crossings, on some videos measurably regressing
them (ss531_id_005/989, ss50505_id_012 -- see AnalyzeConfig's own comment
and docs/superpowers/plans/2026-08-03-meschke-validation-findings.md).
arcs/extract.py's dedup_parallel_arcs now carries that job at the ARC/
trajectory level instead, where duplicate storms and genuine crossings are
measurably separable. Not a strict improvement on every video, though: at
least one out-of-scope high-pattern video (ss50505_id_093) depended on
THIS mechanism's pre-extraction cleanup and regresses when merge_dist is
0.0 (arc-level dedup cannot recover information this stage would have
kept raw detections from losing in the first place) -- callers with
similarly messy footage can still pass a nonzero cluster_merge_dist
explicitly; see AnalyzeConfig's own comment for the measured trade-off.
The rest of this docstring documents the mechanism and its ORIGINAL 0.023
tuning history, preserved for context.

Default justification (merge_dist, see AnalyzeConfig.cluster_merge_dist =
0.023): measured duplicate offsets on real footage are 0.01-0.03 in
normalized units on w~=0.065 boxes (ss3_id_086 frame 194). Distinct cascade
balls approach closer than that only in brief crossing instants; losing one
detection point per ball during those instants used to be "absorbed by
extract_arcs' EM assigner" -- see the strict-lower-confidence paragraph below
for why that framing was wrong and crossings must not merge at all, and
AnalyzeConfig.cluster_merge_dist's own comment for why 0.023 (not the
originally-measured 0.03) is the shipped value.

Strict-lower-confidence absorption: a detection may only be absorbed into a
cluster whose anchor confidence is STRICTLY GREATER than its own -- two
equal-confidence detections never merge, no matter how close. Real duplicate
boxes are always weaker echoes of the strongest box (measured, ss3_id_086
frame 194: anchor 0.303 vs echoes 0.220/0.177/0.152), while two REAL balls at
a crossing instant either tie exactly (the sim's detections are all
confidence=1.0 by construction) or are each independently strong -- never one
strictly weaker echo of the other. Requiring strict inequality is what keeps
real-ball crossings from ever being merged at all (rather than merged and
"absorbed by the EM assigner" as the paragraph above used to justify): with a
plain confidence-descending sort, real ties (crossing balls) and real echoes
(duplicate storms) were indistinguishable by proximity alone, and a
deterministic tie-break (needed for order-independence: same confidence, no
secondary key, means input order picks who is a cluster's anchor) made the
crossing-merge happen on EVERY tie, not just unluckily -- see
test_duplicate_injection_does_not_inflate_catches's measured pre-fix dirty
count for how badly that generalizes. Real detector confidences are
continuous floats and essentially never tie exactly (measured: 0/399
multi-detection frames on ss3_id_086), so this rule costs nothing on real
footage -- it only ever changes behavior on the sim's exactly-tied synthetic
detections.

Residual cross-contamination at genuine crossings (why the default is 0.023,
not 0.03): even with strict inequality, a duplicate clone near a crossing can
land closer to the OTHER real ball's anchor than to its own true source
(both anchors are eligible -- neither ties with a clone's lower confidence),
polluting BOTH balls' reconstructed positions instead of one blended
(previously "harmless") point. Measured directly on the sim integration
test: 0.03 costs -2 catches (a genuine under-count from this
cross-contamination, distinct from the crossing-merge bug strict inequality
already fixes); sweeping 0.005-0.03 found 0.023 as the value where the sim
integration test lands at its best measured margin (dirty exactly equals
clean) while every field spot-check target still holds (see
docs/superpowers/sdd/task-2-report.md's addendum for the full sweep table).
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

    Algorithm (greedy, confidence-first, strict-lower-confidence absorption):
    within each frame, sort detections by confidence descending. For each
    detection (in that order), if its center lies within ``merge_dist``
    (euclidean, normalized units) of an already-kept cluster's anchor AND
    its own confidence is STRICTLY LESS than that anchor's confidence, absorb
    it into that cluster (the nearest such eligible cluster, if more than
    one qualifies); otherwise it starts a new cluster. Two detections at
    equal confidence never merge, regardless of distance (see the module
    docstring's "Strict-lower-confidence absorption" section for why). A
    cluster's output detection takes the confidence-weighted mean of its
    members' x/y/w/h, ``confidence`` = the max confidence among members, and
    ``frame_idx``/``t`` preserved from the frame (all members share them).
    """
    if merge_dist < 0.0:
        raise ValueError("merge_dist must be >= 0")
    if merge_dist == 0.0:
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
    ordered = sorted(frame_dets, key=lambda d: (-d.confidence, d.x, d.y))
    clusters: list[list[Detection]] = []
    anchors: list[tuple[float, float, float]] = []  # (x, y, confidence)

    for d in ordered:
        best_idx = None
        best_dist = merge_dist
        for i, (ax, ay, aconf) in enumerate(anchors):
            # Strict-lower-confidence absorption: an anchor at or below this
            # detection's own confidence is never an eligible merge target,
            # however close -- see module docstring. Ties (equal confidence)
            # are excluded here too, not just lower anchors.
            if d.confidence >= aconf:
                continue
            dist = math.hypot(d.x - ax, d.y - ay)
            if dist <= best_dist:
                best_idx = i
                best_dist = dist
        if best_idx is None:
            clusters.append([d])
            anchors.append((d.x, d.y, d.confidence))
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
