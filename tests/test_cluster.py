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


def test_tied_confidence_sort_is_deterministic():
    """Three chained detections all at same confidence=0.30, spaced so
    A-B and B-C are within merge_dist but A-C is not. Clustering must
    yield the same clusters regardless of input order [A,B,C] vs [B,A,C]."""
    merge_dist = 0.025
    # A at (0.360, 0.770)
    # B at (0.368, 0.770) -- within 0.025 of A (dist=0.008)
    # C at (0.394, 0.770) -- within 0.025 of B (dist=0.026, just over, but let's use 0.024)
    # so A-C dist = 0.034, beyond merge_dist
    A = _d(0.360, 0.770, 0.30)
    B = _d(0.368, 0.770, 0.30)
    C = _d(0.391, 0.770, 0.30)  # dist to B = 0.023, well within 0.025

    # Verify distances manually
    import math
    assert math.hypot(A.x - B.x, A.y - B.y) < merge_dist  # A-B within
    assert math.hypot(B.x - C.x, B.y - C.y) < merge_dist  # B-C within
    assert math.hypot(A.x - C.x, A.y - C.y) > merge_dist  # A-C beyond

    result_abc = cluster_detections([A, B, C], merge_dist=merge_dist)
    result_bac = cluster_detections([B, A, C], merge_dist=merge_dist)

    assert result_abc == result_bac, (
        f"different cluster output for different input orders: "
        f"[A,B,C]→{len(result_abc)} clusters vs [B,A,C]→{len(result_bac)} clusters"
    )


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
    passing with margin rather than at the ±1 band edge."""
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
    assert abs(dirty_c - clean_c) <= 1
