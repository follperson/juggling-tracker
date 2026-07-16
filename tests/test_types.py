import json

import pytest

from juggletrack import SCHEMA_VERSION
from juggletrack.types import Arc, Detection, Run, SessionResult


def test_package_imports():
    assert SCHEMA_VERSION == "1.0"


def make_arc() -> Arc:
    # y(dt) = 1.0*dt^2 - 1.1*dt + 0.65  (down-positive: starts at hand, rises, returns)
    # vy(dt) = 2*dt - 1.1  -> apex at dt = 0.55, flight ends at dt = 1.1
    return Arc(
        id=1, t_start=10.0, t_end=11.1,
        ay=1.0, by=-1.1, cy=0.65, bx=0.16, cx=0.41,
        n_points=30, rmse=0.004,
    )


def test_arc_y_endpoints():
    arc = make_arc()
    assert arc.y_at(10.0) == pytest.approx(0.65)
    assert arc.y_at(11.1) == pytest.approx(0.65)  # symmetric flight returns to launch height


def test_arc_apex():
    arc = make_arc()
    assert arc.apex_t() == pytest.approx(10.55)
    # apex height above hand: v0^2/(2g) with v0=1.1, g=2*ay=2.0 -> 0.3025
    assert arc.apex_y() == pytest.approx(0.65 - 0.3025)
    assert arc.apex_y() < arc.y_at(10.0)  # apex is above (smaller y) than endpoints


def test_arc_velocity_sign():
    arc = make_arc()
    assert arc.vy_at(10.0) < 0  # rising (y shrinking) at throw
    assert arc.vy_at(11.1) > 0  # falling at catch


def test_arc_x_linear():
    arc = make_arc()
    assert arc.x_at(10.0) == pytest.approx(0.41)
    assert arc.x_at(11.1) == pytest.approx(0.41 + 0.16 * 1.1)


def test_session_result_json_roundtrip():
    arc = make_arc()
    run = Run(start_t=10.0, end_t=15.0, catches=9, throws=10, arc_ids=[1],
              end_reason="drop", period_s=0.45, quality=0.8)
    sr = SessionResult(runs=[run], drops=[], arcs=[arc], hand_line_y=0.65)
    blob = sr.model_dump_json()
    back = SessionResult.model_validate(json.loads(blob))
    assert back == sr
    assert back.schema_version == "1.0"


def test_detection_defaults():
    d = Detection(frame_idx=3, t=0.1, x=0.5, y=0.6)
    assert d.confidence == 1.0 and d.w == 0.0
