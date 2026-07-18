import pytest

from juggletrack.arcs.extract import extract_arcs
from juggletrack.events.periodicity import periodicity_score
from juggletrack.sim import simulate_cascade


def test_cascade_is_periodic_at_throw_period():
    r = simulate_cascade(n_throws=16, fps=30.0, seed=1)
    arcs = extract_arcs(r.detections)
    score, lag = periodicity_score(arcs, r.run_start, r.run_end)
    assert score > 0.5
    assert lag == pytest.approx(r.params.period_s, abs=0.07)


def test_no_arcs_scores_zero():
    """Zero variance (no arcs at all) is a real, judged score of 0 -- distinct
    from the "can't judge" (None) case below, which is about the window being
    too short to search any lag, not about the signal being flat."""
    score, lag = periodicity_score([], 0.0, 5.0)
    assert score == 0.0 and lag is None


def test_signal_dies_after_run_end():
    r = simulate_cascade(n_throws=16, fps=30.0, seed=1)
    arcs = extract_arcs(r.detections)
    score, _ = periodicity_score(arcs, r.run_end + 0.2, r.run_end + 3.0)
    assert score == 0.0  # nothing airborne after the run


def test_short_window_cannot_judge_instead_of_zero_padding():
    """Never zero-pad/extend a short window to fake a two-period view.

    Previously a window shorter than 2*lag_hi got silently extended past
    t_to, effectively zero-padding the signal (arcs don't exist past the
    real window) and biasing the ACF. Now the searchable lag band is
    [lag_lo, min(lag_hi, span/2)]; when that band is empty the window
    genuinely cannot hold two periods at any bandable lag, and the score
    must come back None ("can't judge"), not a fabricated number.
    """
    r = simulate_cascade(n_throws=16, fps=30.0, seed=1)
    arcs = extract_arcs(r.detections)
    # span=0.3s: max_lag = min(1.2, 0.15) = 0.15 < lag_lo=0.2 -> band empty.
    score, lag = periodicity_score(arcs, r.run_start, r.run_start + 0.3)
    assert score is None and lag is None


def test_short_window_with_narrow_band_can_still_be_judged():
    """A short window isn't automatically unjudgeable -- only when the band
    it could search is empty. Widening lag_range so span/2 still clears
    lag_lo must let a real (non-extended) score through."""
    r = simulate_cascade(n_throws=16, fps=30.0, seed=1)
    arcs = extract_arcs(r.detections)
    score, lag = periodicity_score(
        arcs, r.run_start, r.run_start + 0.3, lag_range=(0.05, 0.15)
    )
    assert score is not None
