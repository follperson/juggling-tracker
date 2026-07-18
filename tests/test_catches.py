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


def test_truncated_rising_arc_yields_throw():
    """The apex guard must only judge a WITNESSED apex, not an extrapolated one.

    This arc's observed span is truncated on the ascent, before ``apex_t``
    is ever reached: ``ay=1.0, by=-1.1`` puts the fitted apex at
    ``t_start + 0.55``, but ``t_end = t_start + 0.4`` -- the real detections
    stop 0.15s before the ball actually recrosses the hand line (occlusion/
    dropout cutting the ascent short, the turn-4-diagnosed mechanism).

    The fitted apex (0.3475) sits only 0.015 above the hand line (0.3625),
    under the default ``min_apex_above=0.02`` margin. The OLD guard
    evaluated ``arc.apex_y()`` unconditionally and rejected this as "never
    rose meaningfully above the hands" -- but that 0.015 figure comes from
    extrapolating the fit past its own observed window, not from anything
    actually seen. The NEW rule enforces the guard only when the apex is
    witnessed (``t_start <= apex_t <= t_end``); here it isn't, so the guard
    is skipped and the throw is correctly emitted.
    """
    hand_line = 0.3625
    arc = Arc(
        id=1, t_start=10.0, t_end=10.4,
        ay=1.0, by=-1.1, cy=0.65, bx=0.0, cx=0.5,
        n_points=8, rmse=0.01,
    )
    assert not (arc.t_start <= arc.apex_t() <= arc.t_end), "fixture must have an unwitnessed apex"
    throws, catches = derive_events([arc], hand_line)
    assert len(throws) == 1


def test_bounce_arc_apex_witnessed_still_rejected():
    """A bounce's apex is fully witnessed and sits below the hand line: the
    witnessed-apex-aware rule must not relax this case. Since the apex
    (0.82) is within the arc's own observed span [5.0, 5.4] (apex_t=5.2),
    the guard still applies, and a real bounce (never clearing the hands)
    keeps yielding no throw event, exactly as before this fix.
    """
    hand_line = 0.65
    arc = Arc(
        id=3, t_start=5.0, t_end=5.4,
        ay=2.0, by=-0.8, cy=0.9, bx=0.0, cx=0.5,
        n_points=8, rmse=0.01,
    )
    assert arc.t_start <= arc.apex_t() <= arc.t_end, "fixture must have a witnessed apex"
    assert arc.apex_y() > hand_line, "fixture apex must stay below (not clear) the hand line"
    throws, catches = derive_events([arc], hand_line)
    assert throws == []
    assert catches == []
