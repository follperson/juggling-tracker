from juggletrack.arcs.extract import assign_detections, extract_arcs
from juggletrack.sim import simulate_cascade
from juggletrack.types import Detection


def test_flight_detections_get_assigned():
    r = simulate_cascade(n_throws=10, fps=30.0, noise=0.003, seed=1)
    arcs = extract_arcs(r.detections)
    labels = assign_detections(r.detections, arcs)
    assert len(labels) == len(r.detections)
    frac_assigned = sum(1 for a in labels if a != -1) / len(labels)
    assert frac_assigned >= 0.9
    arc_ids = {a.id for a in arcs}
    assert all(a in arc_ids for a in labels if a != -1)


def test_far_points_get_minus_one():
    r = simulate_cascade(n_throws=10, fps=30.0, seed=1)
    arcs = extract_arcs(r.detections)
    t_mid = (r.run_start + r.run_end) / 2
    junk = [Detection(frame_idx=999, t=t_mid, x=0.5, y=0.02),   # far above any apex
            Detection(frame_idx=999, t=t_mid, x=0.98, y=0.65)]  # far right of pattern
    labels = assign_detections(junk, arcs)
    assert labels == [-1, -1]


def test_input_order_preserved():
    r = simulate_cascade(n_throws=6, fps=30.0, seed=2)
    arcs = extract_arcs(r.detections)
    fwd = assign_detections(r.detections, arcs)
    rev = assign_detections(list(reversed(r.detections)), arcs)
    assert rev == list(reversed(fwd))


def test_empty_inputs():
    assert assign_detections([], []) == []
    r = simulate_cascade(n_throws=6, fps=30.0, seed=2)
    assert assign_detections(r.detections, []) == [-1] * len(r.detections)
