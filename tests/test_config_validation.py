import math

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from juggletrack.analyze import AnalyzeConfig
from juggletrack.pipeline.realtime import RealtimeConfig

REALTIME_FLOATS = ["window_s", "freeze_s", "drop_freeze_s", "event_match_tol", "edge_pad"]


def test_realtime_config_rejects_unknown_field():
    with pytest.raises(ValidationError, match="windw_s"):
        RealtimeConfig(windw_s=6.0)


def test_realtime_config_rejects_unknown_json_key():
    with pytest.raises(ValidationError, match="bogus"):
        RealtimeConfig.model_validate_json('{"bogus": 1}')


@pytest.mark.parametrize("value", [math.nan, math.inf])
@pytest.mark.parametrize("field", REALTIME_FLOATS)
def test_realtime_config_rejects_non_finite(field, value):
    with pytest.raises(ValidationError, match=field):
        RealtimeConfig(**{field: value})


@pytest.mark.parametrize(("field", "value"), [
    ("link_max_dist", 0.0),
    ("link_max_dist", -0.08),
    ("link_max_dt", 0.0),
    ("resid_tol", 0.0),
    ("resid_tol", -0.02),
    ("g_range", (8.0, 0.1)),
    ("g_range", (2.0, 2.0)),
    ("em_iters", 0),
    ("em_iters", -1),
    ("min_points", 0),
    ("static_cell", 0.0),
])
def test_analyze_config_rejects_impossible_value(field, value):
    """A zero or negative gate or a reversed g_range returned zero arcs,
    em_iters < 1 skipped every plausibility check, and static_cell=0 raised
    ZeroDivisionError. All of them are reachable through benchmark --config."""
    with pytest.raises(ValidationError, match=field):
        AnalyzeConfig(**{field: value})


@pytest.mark.parametrize(("field", "value"), [
    ("min_points", 1),
    ("em_iters", 1),
    ("g_range", (-1.0, 8.0)),
])
def test_analyze_config_keeps_values_that_work(field, value):
    assert getattr(AnalyzeConfig(**{field: value}), field) == value


def test_analyze_cli_rejects_zero_link_max_dist(tmp_path):
    from juggletrack.cli import app

    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    dets = tmp_path / "detections.jsonl"
    dets.write_text("")
    out = tmp_path / "out"
    result = CliRunner().invoke(app, [
        "analyze", str(video), "--out", str(out), "--detections", str(dets),
        "--link-max-dist", "0",
    ])
    assert result.exit_code == 2, result.output
    assert "greater than 0" in result.output
    assert not out.exists()
