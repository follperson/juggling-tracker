import json

import numpy as np
import pytest

from juggletrack.analyze import analyze_detections
from juggletrack.sim import simulate_cascade
from juggletrack.types import SessionResult


def test_end_to_end_clean_run():
    r = simulate_cascade(n_throws=12, fps=30.0, noise=0.003, dropout=0.1, seed=1)
    sr = analyze_detections(r.detections)
    assert len(sr.runs) == 1
    assert sr.runs[0].catches == 12
    assert sr.runs[0].end_reason == "stop"
    assert sr.hand_line_y == pytest.approx(r.params.hand_y, abs=0.03)
    back = SessionResult.model_validate(json.loads(sr.model_dump_json()))
    assert back == sr


def test_end_to_end_drop_run():
    r = simulate_cascade(n_throws=20, fps=30.0, drop_at_throw=8, seed=1)
    sr = analyze_detections(r.detections)
    assert len(sr.runs) == 1
    assert sr.runs[0].end_reason == "drop"
    assert len(sr.drops) == 1
    assert sr.drops[0].t == pytest.approx(r.missed_catch_t, abs=0.1)


def test_video_end_run():
    r = simulate_cascade(n_throws=12, fps=30.0, seed=1)
    cutoff = r.catch_times[8] + 0.05
    truncated = [d for d in r.detections if d.t <= cutoff]
    sr = analyze_detections(truncated)
    assert sr.runs and sr.runs[-1].end_reason == "video_end"


def test_shuffle_invariance_end_to_end():
    """Spec property test: catch counts invariant under detection reordering."""
    r = simulate_cascade(n_throws=12, fps=30.0, noise=0.002, seed=4)
    sr_a = analyze_detections(r.detections)
    shuffled = list(r.detections)
    np.random.default_rng(0).shuffle(shuffled)
    sr_b = analyze_detections(shuffled)
    assert sr_a == sr_b


def test_empty_input():
    sr = analyze_detections([])
    assert sr.runs == [] and sr.arcs == [] and sr.drops == []
