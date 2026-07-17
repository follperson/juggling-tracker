import json

import numpy as np
import pytest

from juggletrack.analyze import AnalyzeConfig, analyze_detections
from juggletrack.sim import CascadeParams, simulate_cascade
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


def test_config_plumbs_linker_knobs():
    """AnalyzeConfig's link_max_dist/link_max_dt must actually reach extract_arcs.

    Mirrors the field failure: on closer-framed footage, sparse per-sample
    displacement can exceed the default 0.08 link_max_dist and the greedy
    linker never forms fragments, so extract_arcs finds zero arcs. Simulate
    that by downsampling a clean cascade until per-sample gaps get wide
    enough to break the default linker, and confirm widening the knobs
    through AnalyzeConfig recovers the flights.

    Empirically (pinned by this test): every-3rd-frame (~10fps) sampling is
    still recovered fully by the default knobs (6/6 arcs both configs), so
    downsampling to every 4th frame (~7.5fps, per-sample dy > 0.08) is what's
    needed to make the default linker fail outright (0 arcs) while wider
    knobs still recover all 6 throws.
    """
    r = simulate_cascade(n_throws=6, fps=30.0, seed=1)
    sparse = [d for d in r.detections if d.frame_idx % 4 == 0]

    sr_default = analyze_detections(sparse, AnalyzeConfig())
    sr_wide = analyze_detections(
        sparse, AnalyzeConfig(link_max_dist=0.2, link_max_dt=0.35)
    )

    assert len(sr_wide.arcs) >= 4
    assert len(sr_default.arcs) < len(sr_wide.arcs)


def test_low_gravity_framing_recovers_events():
    """Normalized gravity is a framing artifact, not a physical constant.

    Meschke's tightly-cropped ground-truth footage has true-ball ay ~= 0.177
    -- below extract_arcs's pre-fix absolute floor of g_range[0]/2 == 0.25,
    so a perfectly clean, correctly-tracked cascade shot with that framing
    gets every arc pruned and the whole run vanishes (see
    .superpowers/sdd/meschke-import-report.md). Reproduce the failure mode
    directly with the simulator instead of real footage: CascadeParams(g=0.35)
    yields ay = g/2 = 0.175, in the same dead zone.

    Sim geometry check (kept here, not assumed): flight_s = n_balls*period_s
    - dwell_s = 3*0.45 - 0.25 = 1.1s (independent of g). v0 = g*flight_s/2
    = 0.35*1.1/2 ~= 0.1925. Apex rise = v0**2/(2*g) ~= 0.1925**2/0.7 ~= 0.053.
    hand_y defaults to 0.65, so the apex sits at y ~= 0.597 -- a shallow arc
    that stays safely inside [0, 1] (asserted below), unlike a naive low-g
    scenario that might send the ball out of frame.
    """
    r = simulate_cascade(n_throws=12, fps=30.0, seed=1, params=CascadeParams(g=0.35))
    assert all(0.0 <= d.y <= 1.0 and 0.0 <= d.x <= 1.0 for d in r.detections)

    sr = analyze_detections(r.detections)
    assert len(sr.runs) == 1
    assert abs(sr.runs[0].catches - 12) <= 1
