"""A trimmed neighbor must not invent a miss by moving the peer hand band."""

from juggletrack.analyze import AnalyzeConfig, _events_from_arcs
from juggletrack.events.drops import detect_drops
from juggletrack.types import Arc, Run, SessionResult


def scene():
    peers = [
        Arc(
            id=i,
            t_start=i * 0.5,
            t_end=i * 0.5 + 1,
            ay=0.8,
            by=-0.71,
            cy=0.63,
            bx=0.2 if i % 2 else -0.2,
            cx=0.3,
            n_points=30,
            rmse=0.005,
            carry_fit_endpoints_y=(0.63, 0.76),
        )
        for i in range(4)
    ]
    candidate = Arc(
        id=99,
        t_start=2,
        t_end=3,
        ay=0.8,
        by=-0.682,
        cy=0.63,
        bx=0.2,
        cx=0.3,
        n_points=30,
        rmse=0.008,
    )
    arcs = peers + [candidate]
    run = Run(start_t=0, end_t=3, catches=4, throws=5, arc_ids=[a.id for a in arcs], period_s=0.5)
    return arcs, run


def test_peer_evidence_prevents_inventing_drop_on_an_unchanged_ball():
    arcs, run = scene()
    drops, _ = detect_drops(arcs, [run], 0.63)
    assert not any(d.arc_id == 99 for d in drops)
    lost_evidence = [a.model_copy(update={"carry_fit_endpoints_y": None}) for a in arcs]
    drops, _ = detect_drops(lost_evidence, [run], 0.63)
    assert any(d.arc_id == 99 for d in drops)


def test_observed_hand_evidence_survives_roundtrip_and_ema_rederivation():
    arcs, _ = scene()
    session = SessionResult(arcs=arcs, hand_line_y=0.63, meta={"t_last": 3.0})
    restored = SessionResult.model_validate_json(session.model_dump_json())
    assert restored.arcs[0].carry_fit_endpoints_y == (0.63, 0.76)
    assert "carry_fit_endpoints_y" not in restored.arcs[-1].model_dump()
    cfg = AnalyzeConfig()
    for hand_line in [0.63, 0.64]:
        expected = _events_from_arcs(arcs, 3.0, cfg, hand_line_override=hand_line)
        actual = _events_from_arcs(restored.arcs, 3.0, cfg, hand_line_override=hand_line)
        assert actual == expected
        assert not any(d.arc_id == 99 for d in actual.drops)
