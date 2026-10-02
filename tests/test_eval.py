import pytest
from pydantic import ValidationError

from juggletrack.eval.labels import LabeledRun, VideoLabels
from juggletrack.eval.metrics import aggregate_scores, evaluate_session, temporal_iou
from juggletrack.types import DropEvent, Run, SessionResult


def run(start, end, catches, reason="stop"):
    return Run(start_t=start, end_t=end, catches=catches, throws=catches,
               end_reason=reason, period_s=0.45, quality=0.8)


def session(runs=(), drops=()):
    return SessionResult(
        runs=list(runs),
        drops=[DropEvent(t=t, x=0.5, arc_id=None) for t in drops],
    )


def test_temporal_iou():
    assert temporal_iou(0, 10, 0, 10) == 1.0
    assert temporal_iou(0, 10, 5, 15) == pytest.approx(5 / 15)
    assert temporal_iou(0, 1, 2, 3) == 0.0


def test_perfect_match():
    sr = session(runs=[run(1.0, 6.0, 12)], drops=[6.0])
    labels = VideoLabels(video="v.mp4", runs=[LabeledRun(start_t=1.0, end_t=6.0, catches=12)],
                         drops=[6.2])
    rep = evaluate_session(sr, labels)
    assert rep.frac_runs_iou90 == 1.0
    assert rep.frac_catch_within_1 == 1.0
    assert rep.matches[0].catch_error == 0
    assert (rep.drop_tp, rep.drop_fp, rep.drop_fn) == (1, 0, 0)
    assert rep.drop_precision == 1.0 and rep.drop_recall == 1.0


def test_catch_error_and_boundary_miss():
    sr = session(runs=[run(1.0, 5.0, 9)])  # boundary short, catches off by 3
    labels = VideoLabels(video="v.mp4",
                         runs=[LabeledRun(start_t=1.0, end_t=6.0, catches=12)])
    rep = evaluate_session(sr, labels)
    assert rep.matches[0].catch_error == 3
    assert rep.matches[0].iou == pytest.approx(4 / 5)
    assert rep.frac_runs_iou90 == 0.0
    assert rep.frac_catch_within_1 == 0.0


def test_unmatched_runs_count_as_failures():
    sr = session(runs=[run(1.0, 6.0, 12), run(20.0, 22.0, 4)])  # 2nd is spurious
    labels = VideoLabels(video="v.mp4", runs=[
        LabeledRun(start_t=1.0, end_t=6.0, catches=12),
        LabeledRun(start_t=10.0, end_t=15.0, catches=10),  # missed entirely
    ])
    rep = evaluate_session(sr, labels)
    assert len(rep.matches) == 1
    assert rep.unmatched_labeled == [1]
    assert rep.unmatched_pred == [1]
    assert rep.frac_runs_iou90 == 0.5
    assert rep.frac_catch_within_1 == 0.5


def test_drop_precision_recall():
    sr = session(runs=[run(1, 6, 12)], drops=[6.0, 30.0])  # one true, one spurious
    labels = VideoLabels(video="v.mp4",
                         runs=[LabeledRun(start_t=1, end_t=6, catches=12)],
                         drops=[6.3, 12.0])  # 12.0 missed
    rep = evaluate_session(sr, labels)
    assert (rep.drop_tp, rep.drop_fp, rep.drop_fn) == (1, 1, 1)
    assert rep.drop_precision == 0.5 and rep.drop_recall == 0.5


def test_empty_everything():
    rep = evaluate_session(session(), VideoLabels(video="v.mp4"))
    assert rep.frac_runs_iou90 == 1.0 and rep.frac_catch_within_1 == 1.0
    assert rep.drop_precision == 1.0 and rep.drop_recall == 1.0


def test_labels_json_roundtrip(tmp_path):
    labels = VideoLabels(video="v.mp4",
                         runs=[LabeledRun(start_t=1, end_t=6, catches=12)], drops=[6.0])
    p = tmp_path / "labels.json"
    p.write_text(labels.model_dump_json(indent=2))
    assert VideoLabels.model_validate_json(p.read_text()) == labels


@pytest.mark.parametrize("document", [
    '{"video": "v.mp4", "runs": [], "drop": [6.0]}',
    '{"video": "v.mp4", "runz": []}',
    '{"video": "v.mp4", "runs": [{"start_t": 1, "end_t": 6, "catches": 12, "end_reasn": "drop"}]}',
])
def test_misspelled_label_keys_are_rejected(document):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        VideoLabels.model_validate_json(document)


@pytest.mark.parametrize(("predicted", "passes"), [(13, True), (14, False)])
def test_catch_error_of_one_passes_and_two_fails(predicted, passes):
    labels = VideoLabels(video="v.mp4", runs=[LabeledRun(start_t=0, end_t=10, catches=12)])
    rep = evaluate_session(session(runs=[run(0, 10, predicted)]), labels)
    assert rep.frac_catch_within_1 == float(passes)
    assert aggregate_scores([rep])["frac_catch_within_1"] == float(passes)


@pytest.mark.parametrize(("pred_end", "passes"), [(9.0, True), (8.99, False)])
def test_run_iou_passes_at_exactly_the_threshold(pred_end, passes):
    labels = VideoLabels(video="v.mp4", runs=[LabeledRun(start_t=0, end_t=10, catches=12)])
    rep = evaluate_session(session(runs=[run(0, pred_end, 12)]), labels)
    assert rep.matches[0].iou == pytest.approx(pred_end / 10)
    assert rep.frac_runs_iou90 == float(passes)
    assert aggregate_scores([rep])["frac_runs_iou90"] == float(passes)


@pytest.mark.parametrize(("predicted", "labeled", "counts", "precision", "recall"), [
    ([6.0, 30.0], [6.3, 12.0], (1, 1, 1), 0.5, 0.5),
    ([6.0, 30.0, 40.0], [6.3], (1, 2, 0), 1 / 3, 1.0),
], ids=["one_tp_fp_fn", "asymmetric"])
def test_aggregate_drop_scores(predicted, labeled, counts, precision, recall):
    sr = session(runs=[run(1, 6, 12)], drops=predicted)
    labels = VideoLabels(video="v.mp4", runs=[LabeledRun(start_t=1, end_t=6, catches=12)],
                         drops=labeled)
    scores = aggregate_scores([evaluate_session(sr, labels)])
    assert (scores["drop_tp"], scores["drop_fp"], scores["drop_fn"]) == counts
    assert scores["drop_precision"] == pytest.approx(precision)
    assert scores["drop_recall"] == pytest.approx(recall)


def test_aggregate_scores_do_not_claim_perfect_empty_sets():
    scores = aggregate_scores([evaluate_session(session(), VideoLabels(video="v.mp4"))])
    assert scores["frac_runs_iou90"] is None and scores["frac_catch_within_1"] is None
    assert scores["drop_precision"] is None and scores["drop_recall"] is None
