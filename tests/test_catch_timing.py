"""Reviewed contact coverage is an optional, independent timing denominator."""
import math

import pytest

from juggletrack.eval.catch_timing import evaluate_catch_timing, score_catch_windows
from juggletrack.eval.labels import ReviewedCatchWindow, VideoLabels
from juggletrack.types import Arc, CatchEvent, Run, SessionResult


def window(catches=(), start=0, end=4):
    return ReviewedCatchWindow(start_t=start, end_t=end, catches=[
        {"earliest_t": a, "latest_t": b} for a, b in catches
    ])


def prediction(t, arc_id=0):
    return CatchEvent(t=t, x=.5, arc_id=arc_id)


def test_overlapping_brackets_use_maximum_cardinality_not_nearest_time():
    # Taking the broad first bracket for t=1 would incorrectly lose t=3.
    result = score_catch_windows([prediction(1), prediction(3, 1)],
                                [window([(0, 3), (1, 1)])])
    assert (result["tp"], result["fp"], result["fn"]) == (2, 0, 0)
    assert [m["reference_index"] for m in result["windows"][0]["matches"]] == [1, 0]


def test_simultaneous_contacts_are_distinct_and_bounds_are_inclusive():
    result = score_catch_windows([prediction(1, 3), prediction(1, 7), prediction(2, 9)],
                                [window([(1, 1), (1, 1), (1.5, 2)])])
    assert result["tp"] == 3
    assert [m["arc_id"] for m in result["windows"][0]["matches"]] == [3, 7, 9]


def test_only_reviewed_half_open_coverage_counts_without_tolerance():
    result = score_catch_windows(
        [prediction(.9), prediction(1), prediction(1.99), prediction(2), prediction(3)],
        [window([(1.01, 1.02)], start=1, end=2)],
    )
    assert (result["tp"], result["fp"], result["fn"]) == (0, 2, 1)
    assert result["windows"][0]["misses"] == [
        {"reference_index": 0, "earliest_t": 1.01, "latest_t": 1.02}]
    assert result["precision"] == 0 and result["recall"] == 0


def test_empty_reviewed_negative_is_not_missing_timing():
    negative = score_catch_windows([prediction(1)], [window()])
    assert (negative["tp"], negative["fp"], negative["fn"]) == (0, 1, 0)
    assert negative["precision"] == 0 and negative["recall"] is None
    empty = score_catch_windows([], [window()])
    assert empty["precision"] is None and empty["recall"] is None
    assert empty["reviewed_seconds"] == 4


@pytest.mark.parametrize("bad", [
    {"start_t": -1, "end_t": 2, "catches": []},
    {"start_t": 2, "end_t": 2, "catches": []},
    {"start_t": 3, "end_t": 2, "catches": []},
    {"start_t": 0, "end_t": math.inf, "catches": []},
    {"start_t": math.nan, "end_t": 2, "catches": []},
    {"start_t": 0, "end_t": 2},  # exhaustive list must be explicit
    {"start_t": 0, "end_t": 2, "catches": [{"earliest_t": 1, "latest_t": .9}]},
    {"start_t": 0, "end_t": 2, "catches": [{"earliest_t": 1, "latest_t": 2}]},
    {"start_t": 1, "end_t": 2, "catches": [{"earliest_t": .9, "latest_t": 1}]},
    {"start_t": 0, "end_t": 2, "catches": [{"earliest_t": 1, "latest_t": math.nan}]},
])
def test_rejects_invalid_coverage_and_contact_bounds(bad):
    with pytest.raises(ValueError):
        ReviewedCatchWindow.model_validate(bad)


def test_overlapping_coverage_rejected_but_adjacent_windows_allowed():
    with pytest.raises(ValueError, match="overlap"):
        VideoLabels(video="v", catch_windows=[window(start=1, end=3), window(start=0, end=2)])
    VideoLabels(video="v", catch_windows=[window(start=1, end=2), window(start=0, end=1)])
    with pytest.raises(ValueError, match="overlap"):
        score_catch_windows([prediction(1)], [window([(1, 1)]), window([(1, 1)])])


def test_only_surviving_run_arcs_supply_predictions():
    def arc(i):
        return Arc(id=i, t_start=0, t_end=2, ay=1, by=-2, cy=.65,
                   bx=0, cx=.5, n_points=20, rmse=0)
    session = SessionResult(
        arcs=[arc(1), arc(2)], hand_line_y=.65,
        runs=[Run(start_t=0, end_t=2, catches=1, throws=1, arc_ids=[2])],
    )
    result = evaluate_catch_timing(session, [window([(2, 2), (2, 2)])])
    assert (result["tp"], result["fp"], result["fn"]) == (1, 0, 1)
    assert result["windows"][0]["matches"][0]["arc_id"] == 2
    session.runs[0].catches = 2
    with pytest.raises(ValueError, match="disagree"):
        evaluate_catch_timing(session, [window([(2, 2)])])
    session.runs = []
    result = evaluate_catch_timing(session, [window([(2, 2)])])
    assert (result["tp"], result["fp"], result["fn"]) == (0, 0, 1)
