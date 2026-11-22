import math

import pytest
from pydantic import ValidationError

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
