import numpy as np

from juggletrack.arcs.extract import extract_arcs
from juggletrack.events.catches import derive_events
from juggletrack.events.handline import estimate_hand_line
from juggletrack.sim import simulate_cascade
from juggletrack.types import Arc


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


def test_truncated_arc_yields_no_phantom_catch():
    """A catch must be witnessed, not just analytically extrapolated.

    This arc is truncated just past its apex: apex_t = t_start + 0.55 (from
    ay=1.0, by=-1.1), but t_end = t_start + 0.6 -- the fitted window ends
    right after the peak, long before the ball would actually reach the
    hand line again. The falling hand-line crossing is at t_start + 1.1,
    which is 0.5s past t_end: far beyond CATCH_EXTRAPOLATION_MARGIN (0.15s)
    of dropout tolerance, and well past the last real detection this arc
    could represent (truncated video or occlusion). A throw is still
    observed (the rising crossing is within the arc's own span), but no
    catch should be manufactured for a crossing that was never actually
    seen.
    """
    hand_line = 0.65
    arc = Arc(
        id=1, t_start=10.0, t_end=10.6,
        ay=1.0, by=-1.1, cy=0.65, bx=0.0, cx=0.5,
        n_points=10, rmse=0.01,
    )
    throws, catches = derive_events([arc], hand_line)
    assert len(throws) == 1
    assert catches == []
