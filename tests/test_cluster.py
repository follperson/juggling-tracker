import pytest

from juggletrack.detect.cluster import cluster_detections
from juggletrack.types import Detection


def _d(x, y, conf, frame=0, t=0.0):
    return Detection(frame_idx=frame, t=t, x=x, y=y, w=0.06, h=0.06, confidence=conf)


def test_near_duplicates_merge_to_strongest_center():
    dets = [_d(0.360, 0.770, 0.30), _d(0.363, 0.755, 0.12), _d(0.357, 0.780, 0.08)]
    out = cluster_detections(dets, merge_dist=0.03)
    assert len(out) == 1
    assert out[0].confidence == 0.30
    assert abs(out[0].x - 0.360) < 0.01  # confidence-weighted, dominated by strongest


def test_distinct_balls_survive():
    dets = [_d(0.36, 0.77, 0.3), _d(0.42, 0.44, 0.2), _d(0.73, 0.53, 0.3)]
    assert len(cluster_detections(dets, merge_dist=0.03)) == 3


def test_frames_are_independent():
    dets = [_d(0.36, 0.77, 0.3, frame=0), _d(0.36, 0.77, 0.3, frame=1, t=0.033)]
    assert len(cluster_detections(dets, merge_dist=0.03)) == 2


def test_zero_merge_dist_is_identity():
    dets = [_d(0.360, 0.770, 0.3), _d(0.361, 0.770, 0.2)]
    assert cluster_detections(dets, merge_dist=0.0) == dets


def test_negative_merge_dist_raises():
    dets = [_d(0.360, 0.770, 0.3)]
    import pytest
    with pytest.raises(ValueError, match="merge_dist must be >= 0"):
        cluster_detections(dets, merge_dist=-0.01)


def test_nan_confidence_raises_instead_of_silently_corrupting_merge():
    """A NaN confidence makes every ``>=`` comparison False, so the strict-
    lower-confidence guard (cluster.py's `d.confidence >= aconf: continue`)
    would vacuously pass and the NaN detection would merge into any nearby
    anchor despite never being strictly lower -- and the confidence-
    descending sort key would be undefined, breaking order-independence.
    Reproduced pre-fix: [conf=0.30, conf=NaN 0.002 away] merges to 1 cluster
    with confidence 0.3 in one input order and 1 cluster with confidence NaN
    in the reversed order -- same input set, different output. Real
    detectors never emit NaN; this guards malformed replayed/saved
    detection streams (e.g. `--detections` JSONL) so they fail loudly
    instead of silently corrupting the merge rule."""
    import math
    dets = [_d(0.360, 0.770, 0.30), _d(0.362, 0.771, float("nan"))]
    with pytest.raises(ValueError, match="non-finite confidence"):
        cluster_detections(dets, merge_dist=0.03)
    # inf is equally non-finite and must be rejected the same way.
    dets_inf = [_d(0.360, 0.770, 0.30), _d(0.362, 0.771, math.inf)]
    with pytest.raises(ValueError, match="non-finite confidence"):
        cluster_detections(dets_inf, merge_dist=0.03)


def test_absorption_is_deterministic_across_input_orderings():
    """Retuned (final review) from an all-tied-confidence fixture that the
    9f9d303 strict-lower-confidence rule made vestigial: all three
    detections shared confidence=0.30, so under the shipped
    `d.confidence >= aconf: continue` guard NOTHING could ever merge --
    both orderings produced 3 singleton clusters, and the test was
    comparing two identity outputs rather than exercising real absorption
    or the greedy nearest-eligible-anchor selection ("the nearest such
    eligible cluster, if more than one qualifies" in cluster_detections'
    own docstring).

    This fixture has genuine, unequal-confidence merge eligibility instead:
    A (0.30) and A2 (0.10) are two real echoes of the same ball (dist
    0.0054, well inside merge_dist) that must collapse into one cluster
    anchored at A; D (0.25) is a second, distant real ball (dist to A
    0.566, far outside merge_dist) that must survive alone, unmerged
    despite being an eligible-by-confidence anchor for A2 too (A2's own
    confidence 0.10 is strictly less than BOTH A's 0.30 and D's 0.25 --
    only proximity, not eligibility, decides A2's cluster). Asserting
    output equality across every permutation of the three inputs, plus the
    expected cluster count/membership/confidences, discriminates real
    absorption-and-selection determinism from no-op identity."""
    import itertools
    A = _d(0.30, 0.30, 0.30)
    A2 = _d(0.305, 0.302, 0.10)
    D = _d(0.70, 0.70, 0.25)
    merge_dist = 0.03

    import math
    assert math.hypot(A.x - A2.x, A.y - A2.y) < merge_dist  # A2 merges into A
    assert math.hypot(A.x - D.x, A.y - D.y) > merge_dist  # D stays separate

    results = [
        cluster_detections(list(perm), merge_dist=merge_dist)
        for perm in itertools.permutations([A, A2, D])
    ]
    first = results[0]
    assert len(first) == 2, f"expected A+A2 merged and D separate, got {len(first)} clusters"
    confidences = sorted(o.confidence for o in first)
    assert confidences == [0.25, 0.30], (
        "expected one merged cluster (confidence 0.30, the stronger echo) "
        f"and D untouched (confidence 0.25); got {confidences}"
    )
    for r in results[1:]:
        assert r == first, (
            "cluster output must not depend on input order: "
            f"got {[(o.x, o.y, o.confidence) for o in r]} vs "
            f"{[(o.x, o.y, o.confidence) for o in first]}"
        )


@pytest.mark.filterwarnings("ignore::numpy.exceptions.RankWarning")
def test_duplicate_injection_does_not_inflate_catches():
    """Duplicate-box storms mint parallel arcs and inflate catches (measured:
    ss3_id_086 oracle 24 -> 57 pre-fix). Injecting 2-3 jittered clones of every
    sim detection must leave the catch count at the clean baseline.

    History on this exact fixture (seed=3, clone rng seed=7): pre-clustering
    dirty=16 (inflated, the bug this module fixes). Deterministic
    confidence-descending sort alone (no strict-inequality rule): dirty=6
    (crossing balls -- tied at confidence=1.0 -- merged on every tie,
    collapsing arcs). Strict-lower-confidence absorption added, still at the
    original merge_dist=0.03: dirty=10 (crossing-merge fixed, but a distinct
    cross-contamination effect -- a clone landing closer to the OTHER real
    ball's anchor than its own true source -- costs 2 catches). Tuned to
    merge_dist=0.023: dirty=12, exactly matching clean -- passing with
    margin rather than at the ±1 band edge.

    Plan 5 task 2b: cluster_merge_dist's shipped default moved first to 0.0
    (box-level clustering retired to a no-op; arcs/extract.py's
    dedup_parallel_arcs alone doing this fixture's duplicate-collapsing
    work at the arc level -- measured clean=12, dirty=11 there), then to
    0.012 after a controller-directed spec-§6.1 adjudication restored a
    small amount of box clustering alongside arc dedup (see
    AnalyzeConfig's own comment for the full combo-sweep rationale).
    Verified directly on this exact fixture at the FINAL shipped default
    (merge_dist=0.012 + arc dedup): clean=12, dirty=12 -- exact match,
    passing with margin rather than at the ±1 band edge.

    Final review: the randomly-jittered clone cloud legitimately poorly-
    conditions np.polyfit on some draws (expected for this fixture's dense,
    near-degenerate point clusters, same rationale as test_extract.py's
    test_dense_same_timestamp_clusters_do_not_crash) -- RankWarning
    suppressed at the source via a marker, not globally, so one elsewhere
    in the suite still surfaces normally."""
    import numpy as np
    from juggletrack.analyze import AnalyzeConfig, analyze_detections
    from juggletrack.sim import simulate_cascade
    sim = simulate_cascade(n_throws=12, fps=30.0, seed=3)
    rng = np.random.default_rng(7)
    clones = []
    for d in sim.detections:
        for _ in range(int(rng.integers(2, 4))):
            clones.append(d.model_copy(update={
                "x": d.x + float(rng.uniform(-0.02, 0.02)),
                "y": d.y + float(rng.uniform(-0.02, 0.02)),
                "confidence": max(0.05, d.confidence * float(rng.uniform(0.3, 0.9))),
            }))
    clean = analyze_detections(sim.detections, config=AnalyzeConfig())
    dirty = analyze_detections(sim.detections + clones, config=AnalyzeConfig())
    clean_c = sum(r.catches for r in clean.runs)
    dirty_c = sum(r.catches for r in dirty.runs)
    assert clean_c == 12, "pinned clean baseline for this fixture; revisit if sim.py changes"
    assert abs(dirty_c - clean_c) <= 1
