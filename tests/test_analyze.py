import json

import numpy as np
import pytest
from pydantic import ValidationError

from juggletrack.analyze import AnalyzeConfig, _events_from_arcs, analyze_detections
from juggletrack.arcs.extract import extract_arcs
from juggletrack.events.handline import estimate_hand_line
from juggletrack.sim import CascadeParams, simulate_cascade
from juggletrack.types import Detection, SessionResult


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


def test_drift_cohort_gate_runs_before_drop_detection():
    """A rejected drift-cohort run must not leak a DropEvent into sr.drops.

    Reuses the turn-4 drift-cohort fixture (tests/test_extract.py::
    test_slow_drift_junk_cohort_known_gap: 3 spatially-separated, briskly
    drifting paths that each fit as a plausible arc but are structurally
    not-juggling as a run -- unidirectional, monotonic, never alternating).
    That test already confirms ``sr.runs == []``. The bug this fixture adds
    on top: give the LAST path a longer tail so it keeps being "detected"
    well past its own hand-line crossing, ending far below hand height
    (``y`` far past ``hand_line + FLOOR_MARGIN``) while still descending
    (``vy_at(t_end) > 0``) and with no later arc to continue the pattern --
    exactly the two-signal combination (``floor_descent`` +
    ``periodicity_collapse``) ``detect_drops`` mints a DropEvent for. Before
    the fix, ``_events_from_arcs`` ran ``detect_drops`` on the ungated runs
    and only filtered ``runs`` afterward, so this junk run's drop survived
    into ``sr.drops`` even though the run itself was correctly rejected.
    Gating before drop detection means the junk run's arcs never reach
    ``detect_drops`` at all.

    The control (reusing ``test_end_to_end_drop_run``'s real cascade-drop
    scenario) confirms the fix doesn't just suppress drops outright: a
    genuine drop in a genuine (non-drift-cohort) run is still detected.
    """
    fps = 30.0
    dets = []
    for k in range(3):
        t0 = 0.5 + k * 0.7
        v0, a = 0.12, 0.1
        # Paths 0 and 1 keep the original 1.2s window (they end back near
        # the hand line, same as test_slow_drift_junk_cohort_known_gap).
        # Path 2 (the last -- no later path's detections to collide with in
        # time) gets a longer window so its tail keeps falling well past
        # its own hand-line crossing instead of stopping right at it.
        dur = 2.0 if k == 2 else 1.2
        for i in range(int(dur * fps)):
            dt = i / fps
            dets.append(Detection(
                frame_idx=int((t0 + dt) * fps), t=t0 + dt,
                x=0.15 + 0.22 * k + 0.15 * dt,
                y=0.6 - v0 * dt + a * dt * dt,
            ))
    sr_junk = analyze_detections(dets)
    assert sr_junk.runs == [], "drift-cohort run gate must still reject this cohort's run"
    assert sr_junk.drops == [], "a gated-out junk run must not leak a DropEvent"

    r = simulate_cascade(n_throws=20, fps=30.0, drop_at_throw=8, seed=1)
    sr_control = analyze_detections(r.detections)
    assert len(sr_control.runs) == 1
    assert len(sr_control.drops) == 1, "a real drop in a real run must still be detected"


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


def test_cluster_merge_dist_rejects_negative():
    """Plan 5 task 2: cluster_merge_dist feeds cluster_detections' merge_dist
    directly -- a negative value has no sane meaning there (euclidean
    distance is never negative) and would either silently disable
    clustering in a confusing way or blow up downstream; reject it at
    config construction, matching the existing AnalyzeConfig/RealtimeConfig
    validation pattern (see RealtimeConfig._validate_envelope)."""
    with pytest.raises(ValidationError):
        AnalyzeConfig(cluster_merge_dist=-0.01)
    AnalyzeConfig(cluster_merge_dist=0.0)  # 0.0 (disables clustering) stays valid


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

    Plan 5 task 2 note: this fixture has a genuine crossing (frame 52/92,
    measured min_dist=0.008) between two real balls, which briefly worried
    clustering into this test's territory. Moot as of the strict-lower-
    confidence absorption fix (detect/cluster.py): every raw sim detection
    is confidence=1.0, so two real balls always tie and NEVER merge under
    strict inequality -- clustering is a verified no-op on this (and every
    other) clean, non-duplicate-injected sim fixture. Reverted to plain
    AnalyzeConfig() (was briefly cluster_merge_dist=0.0-pinned; see
    task-2-report.md's strict-inequality addendum).
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


def test_events_from_arcs_hand_line_override():
    """E2 (realtime.py): `_events_from_arcs`'s `hand_line_override` lets a
    caller skip re-estimating the hand line from `arcs` and force every
    downstream derivation (throws/catches/runs/drops) onto one supplied
    value instead. RealtimeAnalyzer uses this so its EMA-smoothed hand
    line drives every derivation in a cycle, not just derive_events -- see
    pipeline/realtime.py's module docstring for the bug an unpatched split
    (EMA for catches, raw for drops/runs) caused. Three things to prove:
    (1) omitting the override is unaffected (offline parity untouched);
    (2) passing the SAME value the natural estimate would give reproduces
    the natural result exactly; (3) passing a genuinely different value
    changes the derivation (proves it's actually threaded through, not
    silently ignored)."""
    r = simulate_cascade(n_throws=12, fps=30.0, seed=1)
    arcs = extract_arcs(r.detections)
    t_last = max(d.t for d in r.detections)
    natural = estimate_hand_line(arcs)

    default = _events_from_arcs(arcs, t_last, AnalyzeConfig())
    assert default.hand_line_y == natural

    same = _events_from_arcs(arcs, t_last, AnalyzeConfig(), hand_line_override=natural)
    assert same.hand_line_y == natural
    assert [run.catches for run in same.runs] == [run.catches for run in default.runs]

    shifted = _events_from_arcs(arcs, t_last, AnalyzeConfig(), hand_line_override=natural + 0.2)
    assert shifted.hand_line_y == pytest.approx(natural + 0.2)
    assert [run.catches for run in shifted.runs] != [run.catches for run in default.runs], (
        "a meaningfully different override must change the derivation"
    )
