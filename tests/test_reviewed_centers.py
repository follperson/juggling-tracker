import copy
import json

import pytest
from pydantic import ValidationError

from juggletrack.eval.reviewed_centers import ReviewedCenters, evaluate_reviewed_centers
from juggletrack.types import Detection


def center(x, state="airborne", y=.5):
    return {"x": x, "y": y, "state": state}


def labels(*frames):
    return ReviewedCenters.model_validate({
        "schema_version": 1, "review_method": "synthetic", "source_sha256": "a" * 64,
        "width": 100, "height": 100,
        "frames": [{"frame_idx": i, "t": i / 30, "centers": centers}
                   for i, centers in frames],
    })


def pred(x, frame=0, **updates):
    return Detection(frame_idx=frame, t=frame / 30, x=x, y=.5).model_copy(update=updates)


def score(predictions, reviewed, **kwargs):
    return evaluate_reviewed_centers(predictions, reviewed, tolerance_px=6.01, **kwargs)


def test_definite_assignment_maximizes_cardinality_before_distance():
    result = score([pred(.54), pred(.45)], labels((0, [center(.5), center(.6, "held")])))
    assert (result["tp"], result["fp"], result["fn"]) == (2, 0, 0)
    assert result["states"]["airborne"] == {"n": 1, "matched": 1, "recall": 1}
    assert result["states"]["held"] == {"n": 1, "matched": 1, "recall": 1}
    assert result["mean_center_error_px"] == pytest.approx(5.5)


def test_assignment_minimizes_distance_after_cardinality():
    result = score([pred(.51), pred(.57)], labels((0, [center(.5), center(.6)])))
    assert result["mean_center_error_px"] == pytest.approx(2)


def test_definite_targets_have_priority_and_ambiguous_absorption_is_one_to_one():
    reviewed = labels((0, [center(.5, "held"), center(.54, "ambiguous")]))
    result = score([pred(.54), pred(.54), pred(.54)], reviewed)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 0)
    assert result["n_ignored_predictions"] == 1
    assert result["states"]["ambiguous"] == {"n": 1, "matched": 1, "recall": None}
    assert result["precision"] == .5
    assert result == score([pred(.54), pred(.54), pred(.54)], reviewed)


def test_ambiguous_does_not_rescue_missing_definite_target():
    result = score([pred(.8)], labels((0, [center(.2), center(.8, "ambiguous")])))
    assert (result["tp"], result["fp"], result["fn"]) == (0, 0, 1)
    assert result["precision"] is None
    assert result["recall"] == 0


def test_sparse_unreviewed_frames_are_excluded_but_explicit_empty_frames_count():
    result = score([pred(.5), pred(.5, 1), pred(.5, 2)], labels((0, [center(.5)]), (2, [])))
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 0)
    assert result["n_reviewed_frames"] == 2
    assert [frame["frame_idx"] for frame in result["frames"]] == [0, 2]
    assert result["frames"][1]["fp"] == 1


def test_confidence_threshold_preserves_label_counts():
    result = score([pred(.5, confidence=.2)], labels((0, [center(.5)])), min_conf=.5)
    assert (result["tp"], result["fp"], result["fn"]) == (0, 0, 1)


@pytest.mark.parametrize("confidence", [0, 1])
def test_pts_validation_runs_before_confidence_filter(confidence):
    with pytest.raises(ValueError, match="PTS"):
        score([pred(.5, t=.00101, confidence=confidence)], labels((0, [])), min_conf=.5)
    score([pred(.5, t=.001, confidence=confidence)], labels((0, [])), min_conf=.5)


@pytest.mark.parametrize("updates", [
    {"x": float("nan")}, {"x": -.01}, {"y": 1.01}, {"confidence": float("inf")},
    {"confidence": -.1}, {"t": -1}, {"t": float("nan")}, {"frame_idx": -1},
    {"frame_idx": 1.5},
])
def test_all_predictions_are_validated_even_unreviewed_below_threshold(updates):
    with pytest.raises(ValueError):
        score([pred(.5, frame=10, confidence=.1).model_copy(update=updates)],
              labels((0, [])), min_conf=.5)


@pytest.mark.parametrize("kwargs", [{"tolerance_px": 0}, {"tolerance_px": float("inf")},
                                    {"min_conf": float("nan")}, {"min_conf": 1.1}])
def test_bad_settings(kwargs):
    with pytest.raises(ValueError):
        evaluate_reviewed_centers([], labels((0, [])), **{"tolerance_px": 5, **kwargs})


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(schema_version=True), lambda d: d.update(schema_version="1"),
    lambda d: d.update(schema_version=2), lambda d: d.update(width=0),
    lambda d: d.update(width="100"), lambda d: d.update(source_sha256="xyz"),
    lambda d: d.update(frames=[]), lambda d: d.update(unknown=1),
    lambda d: d.pop("review_method"), lambda d: d.update(review_method="unreviewed"),
    lambda d: d["frames"][0].pop("centers"),
    lambda d: d["frames"][0].update(frame_idx=True),
    lambda d: d["frames"][0].update(t=float("nan")),
    lambda d: d["frames"][0]["centers"][0].update(x=1.1),
    lambda d: d["frames"][0]["centers"][0].update(state="predicted"),
    lambda d: d["frames"].append(copy.deepcopy(d["frames"][0])),
    lambda d: d["frames"].append({"frame_idx": 1, "t": 0, "centers": []}),
])
def test_strict_json_contract(mutation):
    document = labels((0, [center(.5)])).model_dump()
    mutation(document)
    with pytest.raises(ValidationError):
        ReviewedCenters.model_validate_json(json.dumps(document))


def test_round_trip_and_empty_reviewed_frame_are_valid():
    reviewed = labels((0, []), (3, []))
    assert ReviewedCenters.model_validate_json(reviewed.model_dump_json()) == reviewed
    result = score([], reviewed)
    assert result["precision"] is result["recall"] is None


def test_pixel_distances_use_both_image_dimensions():
    reviewed = labels((0, [center(.5)] )).model_copy(update={"height": 1000})
    result = score([pred(.5, y=.52)], reviewed)
    assert (result["tp"], result["fp"], result["fn"]) == (0, 1, 1)
