import numpy as np

from juggletrack.arcs.extract import extract_arcs
from juggletrack.events.catches import derive_events
from juggletrack.events.handline import estimate_hand_line
from juggletrack.sim import simulate_cascade


def pipeline(r):
    arcs = extract_arcs(r.detections)
    hl = estimate_hand_line(arcs)
    throws, catches = derive_events(arcs, hl)
    return arcs, hl, throws, catches


def test_clean_run_event_counts_and_subframe_accuracy():
    r = simulate_cascade(n_throws=12, fps=30.0, seed=1)
    _, _, throws, catches = pipeline(r)
    assert len(throws) == 12 and len(catches) == 12
    throw_err = np.abs(np.array([e.t for e in throws]) - np.array(r.throw_times))
    catch_err = np.abs(np.array([e.t for e in catches]) - np.array(r.catch_times))
    # sub-frame: better than half a frame at 30fps
    assert throw_err.max() < 0.017 and catch_err.max() < 0.017


def test_events_sorted_and_linked_to_arcs():
    r = simulate_cascade(n_throws=10, fps=30.0, seed=2)
    arcs, _, throws, catches = pipeline(r)
    assert [e.t for e in throws] == sorted(e.t for e in throws)
    arc_ids = {a.id for a in arcs}
    assert all(e.arc_id in arc_ids for e in throws + catches)


def test_dropped_ball_has_throw_but_no_catch():
    r = simulate_cascade(n_throws=20, fps=30.0, drop_at_throw=8, seed=1)
    _, _, throws, catches = pipeline(r)
    assert len(throws) == len(r.throw_times)
    assert len(catches) == len(r.catch_times)  # exactly the successful ones
    # and specifically: no catch event near the missed catch time
    assert all(abs(c.t - r.missed_catch_t) > 0.1 for c in catches)


def test_noisy_run_events_within_tolerance():
    r = simulate_cascade(n_throws=12, fps=30.0, noise=0.004, dropout=0.15, seed=2)
    _, _, throws, catches = pipeline(r)
    assert len(throws) == 12 and len(catches) == 12
    catch_err = np.abs(np.array([e.t for e in catches]) - np.array(r.catch_times))
    assert catch_err.max() < 0.05
