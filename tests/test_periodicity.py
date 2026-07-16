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
    score, lag = periodicity_score([], 0.0, 5.0)
    assert score == 0.0 and lag is None


def test_signal_dies_after_run_end():
    r = simulate_cascade(n_throws=16, fps=30.0, seed=1)
    arcs = extract_arcs(r.detections)
    score, _ = periodicity_score(arcs, r.run_end + 0.2, r.run_end + 3.0)
    assert score == 0.0  # nothing airborne after the run
