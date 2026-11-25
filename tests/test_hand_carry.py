"""Physical controls for separating a descending hand carry from a flight."""

import pytest

import juggletrack.analyze as analyzer
from juggletrack.arcs.fit import fit_arc, points_array
from juggletrack.events.carry import trim_hand_carry
from juggletrack.events.drops import is_floor_bound
from juggletrack.types import Detection


def fixture(turn=True, slowed=True):
    dets = []
    for i in range(47):
        t = i * 0.025
        x = 0.2 + 0.2 * t if not turn or t <= 0.9 else 0.38 - 0.3 * (t - 0.9)
        y = 0.4 - t + t * t if not slowed or t <= 0.9 else 0.31 + 0.6 * (t - 0.9)
        dets.append(Detection(frame_idx=i, t=t, x=x, y=y, confidence=0.9))
    return dets, fit_arc(points_array(dets), 7)


def test_turn_while_continuing_downward_is_trimmed():
    dets, arc = fixture()
    assert is_floor_bound(arc, 0.35)
    [trimmed] = trim_hand_carry(dets, [arc], 0.35, 0.02)
    assert trimmed.id == arc.id
    assert trimmed.t_end < arc.t_end
    assert not is_floor_bound(trimmed, 0.35)
    assert dets[-1].y > dets[-4].y  # Hand and ball keep descending.


def test_actual_drop_with_no_turn_is_untouched():
    dets, arc = fixture(False, False)
    assert trim_hand_carry(dets, [arc], 0.35, 0.02) == [arc]


def test_separated_crossing_hypotheses_cannot_witness_a_carry():
    dets, arc = fixture()
    for point in list(dets[35:]):
        dets.extend(
            [
                point.model_copy(update={"x": point.x - 0.035}),
                point.model_copy(update={"x": point.x + 0.035}),
            ]
        )
    dets.sort(key=lambda d: d.t)
    assert trim_hand_carry(dets, [arc], 0.35, 0.02) == [arc]


def test_gap_at_the_turn_cannot_jump_to_another_ball():
    dets, arc = fixture()
    dets = [d for d in dets if d.frame_idx != 38]
    assert trim_hand_carry(dets, [arc], 0.35, 0.02) == [arc]


def test_bounce_is_not_a_descending_hand_carry():
    dets, arc = fixture()
    dets = [d.model_copy(update={"y": d.y - 0.12}) if d.frame_idx == 40 else d for d in dets]
    assert trim_hand_carry(dets, [arc], 0.35, 0.02) == [arc]


def test_ordinary_nonfloor_arc_is_identical():
    dets, _ = fixture()
    arc = fit_arc(points_array(dets[:17]), 7)
    assert trim_hand_carry(dets, [arc], 0.35, 0.02)[0] is arc


def test_competing_flight_at_turn_vetoes_a_false_catch():
    dets, arc = fixture()
    other = arc.model_copy(update={"id": 8, "bx": -0.2, "cx": 0.56})
    result = trim_hand_carry(dets, [arc, other], 0.35, 0.02)
    assert result[0] == arc


def test_genuine_miss_near_another_ball_keeps_its_floor_descent():
    dets, dropped = fixture(False, False)
    # A neighboring trajectory turns back near the falling ball. Its boxes
    # cannot witness a catch on the genuine miss.
    neighbor, _ = fixture()
    for d in neighbor:
        if d.t >= 0.85:
            dets.append(d.model_copy(update={"confidence": 1.0}))
    dets.sort(key=lambda d: d.t)
    assert trim_hand_carry(dets, [dropped], 0.35, 0.02) == [dropped]


def test_deflected_miss_continuing_ballistic_descent_is_not_a_catch():
    dets, arc = fixture(turn=True, slowed=False)
    assert is_floor_bound(arc, 0.35)
    assert trim_hand_carry(dets, [arc], 0.35, 0.02) == [arc]


@pytest.mark.parametrize("slowed", [True, False], ids=["caught", "deflected-miss"])
def test_analyze_detections_uses_hand_motion_to_distinguish_a_catch(monkeypatch, slowed):
    # Inject the four camera tracks after extraction. Clustering, shared hand
    # height, catch/drop decisions and run segmentation use the public pipeline.
    detections, arcs = [], []
    for flight in range(4):
        last = flight == 3
        samples, _ = fixture(turn=last, slowed=last and slowed)
        start_frame = 20 * flight
        samples = [
            d.model_copy(
                update={
                    "frame_idx": start_frame + d.frame_idx,
                    "t": (start_frame + d.frame_idx) / 40,
                    "y": d.y if last else d.y - 0.05,
                }
            )
            for d in samples
            if last or d.frame_idx <= 40
        ]
        detections.extend(samples)
        arcs.append(fit_arc(points_array(samples), flight))
    monkeypatch.setattr(analyzer, "extract_arcs", lambda *args, **kwargs: list(arcs))
    result = analyzer.analyze_detections(detections)
    assert result.hand_line_y == pytest.approx(0.35)
    assert result.arcs[:3] == arcs[:3]
    assert len(result.runs) == 1
    if slowed:
        assert result.runs[0].catches == 4
        assert 2.35 <= result.arcs[-1].t_end <= 2.45
        assert result.drops == []
    else:
        assert result.runs[0].catches == 3
        assert result.arcs[-1] == arcs[-1]
        assert [drop.arc_id for drop in result.drops] == [3]
