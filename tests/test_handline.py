import pytest

from juggletrack.arcs.extract import extract_arcs
from juggletrack.events.handline import estimate_hand_line
from juggletrack.sim import simulate_cascade


def test_hand_line_clean():
    r = simulate_cascade(n_throws=12, fps=30.0, seed=1)
    hl = estimate_hand_line(extract_arcs(r.detections))
    assert hl == pytest.approx(r.params.hand_y, abs=0.02)


def test_hand_line_robust_to_drop():
    r = simulate_cascade(n_throws=20, fps=30.0, drop_at_throw=8, seed=1)
    hl = estimate_hand_line(extract_arcs(r.detections))
    assert hl == pytest.approx(r.params.hand_y, abs=0.03)


def test_hand_line_empty_arcs():
    assert estimate_hand_line([]) == 0.0
