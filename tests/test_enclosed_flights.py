"""Regressions for whole flights hidden inside low-curvature enclosing pieces."""

import numpy as np
import pytest

import juggletrack.arcs.extract as extraction

from juggletrack.arcs.extract import (
    _EnclosedFlight,
    _restore_enclosed_flights,
    _seed_pieces,
)
from juggletrack.arcs.fit import fit_arc
from juggletrack.types import Arc, Detection


def _cloud():
    # A held ball moves down at constant speed, then follows a whole flight.
    # Confidence is one constant per physical phase; every sample is 1/30 s.
    t = np.arange(28) / 30
    x = 0.4 + 0.1 * t
    y = np.where(t < 0.4, 0.32 + 0.1 * t, 0.3 + (t - 0.65) ** 2)
    confidence = np.where(t < 0.4, 0.5, 1.0)
    return np.column_stack((t, x, y, confidence))


def _cohort():
    return [
        Arc(
            id=i,
            t_start=5 + i,
            t_end=5.6 + i,
            ay=1.0,
            by=-0.6,
            cy=0.4,
            bx=0.1,
            cx=0.3,
            n_points=20,
            rmse=0.001,
        )
        for i in range(4)
    ]


def _restore(arr, arcs, cuts, **overrides):
    config = dict(
        g_range=(0.1, 8.0),
        resid_tol=0.02,
        min_points=6,
        min_duration=0.15,
        max_abs_bx=0.6,
    )
    config.update(overrides)
    return _restore_enclosed_flights(arr, arcs, cuts, **config)


def test_enclosing_carry_preserves_typed_apex_backup_and_recovers_after_em():
    arr = _cloud()
    cut = _seed_pieces(arr, list(range(len(arr))), 0.02)
    assert not cut.flights
    assert len(cut.pieces) == 1
    assert fit_arc(arr[cut.pieces[0]]).ay < 0.2
    assert len(cut.enclosed) == 1
    enclosed = cut.enclosed[0]
    assert isinstance(enclosed, _EnclosedFlight)
    assert enclosed.flight == list(range(12, 28))
    arcs = _cohort()
    restored = _restore(arr, arcs, [cut])
    assert restored[:4] == arcs
    assert len(arcs) == 4  # restoration does not mutate the established cohort
    assert len(restored) == 5
    assert restored[-1].ay == pytest.approx(1.0)
    assert restored[-1].rmse < 1e-12


def test_one_independently_claimed_crossing_point_vetoes_recovery():
    arr = _cloud()
    cut = _seed_pieces(arr, list(range(len(arr))), 0.02)
    t, x, y, _ = arr[12]
    crossing = Arc(
        id=4,
        t_start=t - 0.3,
        t_end=t + 0.3,
        ay=1.0,
        by=-0.6,
        cy=y + 0.09,
        bx=-0.4,
        cx=x + 0.12,
        n_points=20,
        rmse=0.001,
    )
    # At the apex this otherwise distinct flight crosses a backup point.
    assert crossing.x_at(t) == pytest.approx(x)
    assert crossing.y_at(t) == pytest.approx(y)
    arcs = [*_cohort(), crossing]
    assert _restore(arr, arcs, [cut]) == arcs


def test_accepted_backup_claims_points_before_later_backup():
    arr = _cloud()
    cut = _seed_pieces(arr, list(range(len(arr))), 0.02)
    duplicate = cut._replace(enclosed=cut.enclosed * 2)
    assert len(_restore(arr, _cohort(), [duplicate])) == 5


@pytest.mark.parametrize(
    "config",
    [
        {"min_points": 17},
        {"min_duration": 0.6},
        {"max_abs_bx": 0.01},
        {"g_range": (0.1, 1.5)},
    ],
)
def test_recovery_obeys_existing_arc_quality_gates(config):
    arr = _cloud()
    cut = _seed_pieces(arr, list(range(len(arr))), 0.02)
    arcs = _cohort()
    assert _restore(arr, arcs, [cut], **config) == arcs


def test_recovery_requires_supported_gravity_and_existing_quorum():
    arr = _cloud()
    cut = _seed_pieces(arr, list(range(len(arr))), 0.02)
    arcs = _cohort()
    unsupported = cut._replace(enclosed=(cut.enclosed[0]._replace(ay=3.0),))
    assert _restore(arr, arcs, [unsupported]) is arcs
    assert _restore(arr, arcs[:3], [cut]) == arcs[:3]


def test_enclosing_supported_model_is_never_replaced():
    arr = _cloud()
    cut = _seed_pieces(arr, list(range(len(arr))), 0.02)
    flight = cut.enclosed[0].flight
    supported = cut._replace(enclosed=(_EnclosedFlight(flight, flight, 1.0),))
    arcs = _cohort()
    assert _restore(arr, arcs, [supported]) == arcs


def test_ordinary_whole_flight_keeps_original_seed_path():
    t = np.linspace(0, 0.6, 20)
    arr = np.column_stack((t, 0.4 + 0.1 * t, 0.3 + 1.39 * (t - 0.3) ** 2, np.ones(len(t))))
    cut = _seed_pieces(arr, list(range(len(t))), 0.02)
    assert cut.enclosed == ()
    arcs = _cohort()
    assert _restore(arr, arcs, [cut]) is arcs


def test_fitted_recovery_gravity_is_checked_against_cohort():
    arr = _cloud()
    cut = _seed_pieces(arr, list(range(len(arr))), 0.02)
    flight = cut.enclosed[0].flight
    arr[flight, 2] = 0.273 + 2.5 * (arr[flight, 0] - 0.665) ** 2
    arcs = _cohort()
    assert _restore(arr, arcs, [cut]) == arcs


def test_extract_arcs_recovers_planted_carry_without_changing_four_ordinary_flights(monkeypatch):
    arr = _cloud()
    dets = [
        Detection(frame_idx=i, t=t, x=x, y=y, confidence=confidence)
        for i, (t, x, y, confidence) in enumerate(arr)
    ]
    for start in (2.0, 3.0, 4.0, 5.0):
        for i in range(19):
            elapsed = i / 30
            t = start + elapsed
            dets.append(
                Detection(
                    frame_idx=round(t * 30),
                    t=t,
                    x=0.1 + 0.1 * elapsed,
                    y=0.35 + (elapsed - 0.3) ** 2,
                )
            )
    restored = extraction.extract_arcs(dets)
    monkeypatch.setattr(extraction, "_restore_enclosed_flights", lambda arr, arcs, cuts, **kw: arcs)
    original = extraction.extract_arcs(dets)
    assert len(original) == 4
    assert len(restored) == 5
    assert [a.model_copy(update={"id": -1}) for a in restored[1:]] == [
        a.model_copy(update={"id": -1}) for a in original
    ]
    assert restored[0].t_start == pytest.approx(0.4)
    assert restored[0].t_end == pytest.approx(0.9)
    assert restored[0].ay == pytest.approx(1.0)


def test_low_recovery_endpoints_cannot_erase_an_existing_catch():
    from juggletrack.arcs.extract import _preserves_catch_witnesses
    from juggletrack.events.catches import derive_events
    from juggletrack.events.handline import estimate_hand_line

    def height_arc(i, start_y, end_y):
        return Arc(
            id=i,
            t_start=5 + i,
            t_end=5.6 + i,
            ay=1.0,
            by=(end_y - start_y - 0.36) / 0.6,
            cy=start_y,
            bx=0.1,
            cx=0.3,
            n_points=20,
            rmse=0.001,
        )

    existing = [
        height_arc(0, 0.5, 0.64),
        height_arc(1, 0.5, 0.5),
        height_arc(2, 0.55, 0.55),
        height_arc(3, 0.6, 0.6),
    ]
    candidate = height_arc(4, 0.44, 0.44)
    before = estimate_hand_line(existing)
    after = estimate_hand_line([*existing, candidate])
    assert derive_events([candidate], before)[1]  # recovery itself is catch-compatible
    assert derive_events([existing[0]], before)[1]
    assert not derive_events([existing[0]], after)[1]
    assert not _preserves_catch_witnesses(existing, candidate)


def test_late_floor_bound_backup_needs_more_than_a_clean_parabola():
    from juggletrack.events.catches import derive_events

    arr = _cloud()
    cut = _seed_pieces(arr, list(range(len(arr))), 0.02)
    flight = cut.enclosed[0].flight
    arr[flight, 2] = 0.3 + (arr[flight, 0] - 0.5) ** 2
    arcs = [a.model_copy(update={"cy": 0.34}) for a in _cohort()]
    trial = fit_arc(arr[flight])
    assert trial.rmse < 1e-12
    assert trial.ay == pytest.approx(1.0)
    assert trial.y_at(trial.t_end) > 0.34 + 0.10
    assert not derive_events([trial], 0.34)[1]
    assert _restore(arr, arcs, [cut]) == arcs


def test_full_apex_without_observed_landing_stays_unrecovered():
    arr = _cloud()
    cut = _seed_pieces(arr, list(range(len(arr))), 0.02)
    arcs = [arc.model_copy(update={"cy": 0.5}) for arc in _cohort()]
    assert _restore(arr, arcs, [cut]) == arcs
