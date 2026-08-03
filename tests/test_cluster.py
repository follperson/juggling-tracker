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


def test_duplicate_injection_does_not_inflate_catches():
    """Duplicate-box storms mint parallel arcs and inflate catches (measured:
    ss3_id_086 oracle 24 -> 57 pre-fix). Injecting 2-3 jittered clones of every
    sim detection must leave the catch count at the clean baseline."""
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
