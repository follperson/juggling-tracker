"""Both sides of event-core gates that real footage depends on.

A mutation pass found each of these gates could be moved or deleted with the
whole suite green. Each test sits close to the shipped threshold, so loosening
or tightening it flips the result.
"""
import pytest

from juggletrack.analyze import AnalyzeConfig
from juggletrack.arcs.extract import dedup_parallel_arcs, extract_arcs
from juggletrack.detect.cluster import cluster_detections
from juggletrack.events.catches import derive_events, hand_line_crossings
from juggletrack.events.drops import detect_drops
from juggletrack.types import Arc, Detection, Run

MERGE_DIST = AnalyzeConfig().cluster_merge_dist


def _throw_dets(bx: float, fps: float = 30.0, n: int = 16) -> list[Detection]:
    """A clean, noise-free parabola moving sideways at `bx` per second."""
    return [
        Detection(frame_idx=i, t=i / fps, x=0.2 + bx * i / fps,
                  y=1.5 * (i / fps) ** 2 - 0.75 * (i / fps) + 0.6)
        for i in range(n)
    ]


@pytest.mark.parametrize("bx", [0.5, -0.5])
def test_fast_but_plausible_lateral_throw_is_kept(bx):
    arcs = extract_arcs(_throw_dets(bx))
    assert [round(a.bx, 3) for a in arcs] == [bx]


@pytest.mark.parametrize("bx", [0.8, -0.8])
def test_implausibly_fast_lateral_chain_is_pruned(bx):
    assert extract_arcs(_throw_dets(bx)) == []


def _det(x, conf):
    return Detection(frame_idx=0, t=0.0, x=x, y=0.5, w=0.06, h=0.06, confidence=conf)


@pytest.mark.parametrize(("gap", "n_out"), [(0.011, 1), (0.013, 2)])
def test_cluster_merge_radius_at_shipped_value(gap, n_out):
    dets = [_det(0.5, 0.9), _det(0.5 + gap, 0.5)]
    assert len(cluster_detections(dets, merge_dist=MERGE_DIST)) == n_out


def test_cluster_membership_uses_fixed_anchor_not_running_mean():
    """The third clone is 0.011 from the anchor but about 0.015 from the
    weighted mean of the first two, so it merges only if membership is
    tested against the anchor."""
    dets = [_det(0.500, 0.9), _det(0.511, 0.5), _det(0.489, 0.3)]
    assert len(cluster_detections(dets, merge_dist=MERGE_DIST)) == 1


def test_merged_position_is_confidence_weighted():
    """Weighted x is 0.501. An unweighted mean would give 0.505."""
    out = cluster_detections([_det(0.500, 0.9), _det(0.510, 0.1)], merge_dist=MERGE_DIST)
    assert len(out) == 1
    assert out[0].x == pytest.approx(0.501, abs=1e-9)


def test_dedup_collapses_parallel_arcs_inside_shipped_tolerance():
    """Mean separation 0.10 is inside the shipped 0.15 tolerance. The other
    side, 0.164 kept, is pinned in test_extract.py."""
    a = Arc(id=0, t_start=0.0, t_end=0.8, ay=1.0, by=-0.8, cy=0.6, bx=0.0, cx=0.40,
            n_points=20, rmse=0.005)
    b = a.model_copy(update={"id": 1, "cx": 0.50, "n_points": 15})
    assert [o.id for o in dedup_parallel_arcs([a, b])] == [0]


HAND_LINE = 0.65


def _arc_with_apex(apex_y: float) -> Arc:
    # Apex at t=5.2, inside the arc's own span, so it is witnessed.
    return Arc(id=1, t_start=5.0, t_end=5.4, ay=1.0, by=-0.4, cy=apex_y + 0.04,
               bx=0.0, cx=0.5, n_points=12, rmse=0.01)


@pytest.mark.parametrize("apex_y", [HAND_LINE - 0.01, HAND_LINE + 0.01],
                         ids=["just_above_hands", "just_below_hands"])
def test_apex_near_hand_line_yields_no_throw_and_no_catch(apex_y):
    arc = _arc_with_apex(apex_y)
    assert arc.t_start <= arc.apex_t() <= arc.t_end
    assert derive_events([arc], HAND_LINE) == ([], [])


def _bounce_scenario(bounce_cx: float):
    """A floor-bound miss while the juggler keeps going, so periodicity never
    collapses. Only a bounce near the miss can supply the second signal."""
    period = 0.45
    cand = Arc(id=1, t_start=10.0, t_end=11.31, ay=1.0, by=-1.1, cy=0.65, bx=0.0, cx=0.5,
               n_points=10, rmse=0.01)
    _, miss_t = hand_line_crossings(cand, HAND_LINE)
    continuing = Arc(id=2, t_start=miss_t + period, t_end=miss_t + period + 1.1,
                     ay=1.0, by=-1.1, cy=0.65, bx=0.0, cx=0.55, n_points=10, rmse=0.01)
    bounce = Arc(id=3, t_start=cand.t_end + 0.1, t_end=cand.t_end + 0.5, ay=2.0, by=-0.8,
                 cy=0.95, bx=0.0, cx=bounce_cx, n_points=12, rmse=0.01)
    run = Run(start_t=cand.t_start, end_t=miss_t, catches=1, throws=2,
              arc_ids=[cand.id, continuing.id], period_s=period, end_reason="stop")
    return detect_drops([cand, continuing, bounce], [run], HAND_LINE)


def test_bounce_after_run_end_decides_drop():
    drops, runs = _bounce_scenario(bounce_cx=0.5)
    assert [(d.arc_id, d.signals) for d in drops] == [(1, ["floor_descent", "bounce"])]
    assert runs[0].end_reason == "drop"


def test_bounce_far_from_miss_does_not_count():
    drops, runs = _bounce_scenario(bounce_cx=0.8)
    assert drops == []
    assert runs[0].end_reason == "stop"
