"""Pins for shipped defaults that no behavioral test distinguishes.

Each value below was found by a mutation pass to be changeable with the whole
suite still green, while moving counts on real footage. `juggletrack live`
runs `RealtimeConfig()` as is (cli.py), `juggletrack analyze` runs
`AnalyzeConfig()`, and `juggletrack label` calls `extract_arcs` with its own
keyword defaults. Retuning any of these is allowed, but it has to be a
deliberate edit to this file.
"""
import inspect

import pytest

from juggletrack.analyze import AnalyzeConfig
from juggletrack.arcs.extract import dedup_parallel_arcs, extract_arcs
from juggletrack.pipeline.realtime import RealtimeConfig

# Measured effects come from the audit's recorded-clock realtime replay of
# outputs/meschke-val/ss3_id_016, reported as live runs/catches/drops.
# The shipped defaults give 9/210/8.
REALTIME_PINS = [
    ("window_s", 8.0,
     "window_s 8.0 -> 6.0 collapsed ss3_id_016 from 9/210/8 to 1/198/8"),
    ("drop_freeze_s", 2.5,
     "drop_freeze_s 2.5 -> 1.5 raised ss3_id_016 drops from 8 to 24"),
    ("hand_line_ema_alpha", 0.2,
     "hand_line_ema_alpha 0.2 -> 1.0 (EMA off) gave ss3_id_016 9/244/33"),
]


@pytest.mark.parametrize(
    ("field", "shipped", "effect"), REALTIME_PINS, ids=[p[0] for p in REALTIME_PINS],
)
def test_realtime_default_is_pinned(field, shipped, effect):
    assert getattr(RealtimeConfig(), field) == pytest.approx(shipped), effect


def test_remaining_analyze_defaults_are_pinned():
    """cluster_merge_dist has its own adjudicated pin in test_analyze.py.
    The rest are pinned here. One measured example of why: static_max_span_s
    1.5 -> 5.0 removed 112 catches across 32 of the 113 saved sessions
    (webcam_20260804_174639-640 went from 6 catches to 2) with every test
    passing."""
    assert AnalyzeConfig().model_dump(exclude={"cluster_merge_dist"}) == {
        "g_range": (0.1, 8.0),
        "resid_tol": 0.02,
        "min_points": 6,
        "min_duration": 0.15,
        "link_max_dt": 0.18,
        "link_max_dist": 0.08,
        "em_iters": 5,
        "gap_factor": 1.3,
        "min_arcs": 3,
        "static_cell": 0.03,
        "static_max_span_s": 1.5,
        "video_end_margin": 0.15,
        "arc_dedup_overlap_frac": 0.75,
        "arc_dedup_traj_tol": 0.15,
    }


SHARED_KNOBS = [
    (extract_arcs, "g_range", "g_range"),
    (extract_arcs, "resid_tol", "resid_tol"),
    (extract_arcs, "min_points", "min_points"),
    (extract_arcs, "min_duration", "min_duration"),
    (extract_arcs, "link_max_dt", "link_max_dt"),
    (extract_arcs, "link_max_dist", "link_max_dist"),
    (extract_arcs, "em_iters", "em_iters"),
    (extract_arcs, "static_cell", "static_cell"),
    (extract_arcs, "static_max_span_s", "static_max_span_s"),
    (dedup_parallel_arcs, "overlap_frac", "arc_dedup_overlap_frac"),
    (dedup_parallel_arcs, "traj_tol", "arc_dedup_traj_tol"),
]


@pytest.mark.parametrize(
    ("func", "param", "field"), SHARED_KNOBS,
    ids=[f"{f.__name__}.{p}" for f, p, _ in SHARED_KNOBS],
)
def test_extraction_defaults_match_analyze_config(func, param, field):
    """`label` calls extract_arcs with its keyword defaults while `analyze`
    passes AnalyzeConfig's fields, so a default that drifts on one side makes
    the two commands disagree on the same footage."""
    default = inspect.signature(func).parameters[param].default
    assert default == getattr(AnalyzeConfig(), field)
