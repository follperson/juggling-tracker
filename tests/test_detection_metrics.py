import pytest

from juggletrack.types import Detection


def point(frame, x, y=0.5):
    return Detection(frame_idx=frame, t=frame / 30, x=x, y=y)


def test_counts_misses_duplicates_and_background_on_empty_frames():
    from juggletrack.eval.detection_metrics import evaluate_ball_centers

    result = evaluate_ball_centers(
        [point(0, .5), point(0, .51), point(0, .9), point(2, .2)],
        [point(0, .5), point(1, .3)],
        width=100, height=100, frame_count=3, tolerance_px=5,
    )
    assert (result["tp"], result["fp"], result["fn"]) == (1, 3, 1)
    assert result["duplicate_fp"] == 1
    assert result["background_fp"] == 2
    assert result["precision"] == .25
    assert result["recall"] == .5
    assert result["mean_center_error_px"] == 0


def test_one_to_one_matching_maximizes_matches_at_crossings():
    from juggletrack.eval.detection_metrics import evaluate_ball_centers

    result = evaluate_ball_centers(
        [point(0, .54), point(0, .45)], [point(0, .5), point(0, .6)],
        width=100, height=100, frame_count=1, tolerance_px=6.01,
    )
    # Greedy nearest-first would spend the .50 target on .54 and miss .45.
    assert result["tp"] == 2
    assert result["fp"] == result["fn"] == 0


def test_distances_use_pixels_on_non_square_video():
    from juggletrack.eval.detection_metrics import evaluate_ball_centers

    result = evaluate_ball_centers(
        [point(0, .5, .52)], [point(0, .5, .5)],
        width=100, height=1000, frame_count=1, tolerance_px=10,
    )
    assert (result["tp"], result["fp"], result["fn"]) == (0, 1, 1)


def test_stride_scores_only_sampled_frames():
    from juggletrack.eval.detection_metrics import evaluate_ball_centers

    result = evaluate_ball_centers(
        [point(0, .5), point(2, .5)], [point(i, .5) for i in range(4)],
        width=100, height=100, frame_count=4, stride=2,
    )
    assert result["n_sampled_frames"] == 2
    assert result["tp"] == 2 and result["fn"] == 0


def test_empty_sample_does_not_claim_perfect_accuracy():
    from juggletrack.eval.detection_metrics import evaluate_ball_centers

    result = evaluate_ball_centers([], [], width=100, height=100, frame_count=1)
    assert result["precision"] is None and result["recall"] is None


def test_confidence_filter_preserves_ground_truth_denominator():
    from juggletrack.eval.detection_metrics import evaluate_ball_centers

    predictions = [point(0, .5), point(1, .3).model_copy(update={"confidence": .2})]
    result = evaluate_ball_centers(
        predictions, [point(0, .5), point(1, .3)], width=100, height=100,
        frame_count=2, min_conf=.5,
    )
    assert result["tp"] == 1 and result["fn"] == 1
    assert result["precision"] == 1 and result["recall"] == .5


@pytest.mark.parametrize("kwargs", [{"tolerance_px": float("nan")}, {"stride": 0}])
def test_invalid_metric_settings_rejected(kwargs):
    from juggletrack.eval.detection_metrics import evaluate_ball_centers

    with pytest.raises(ValueError):
        evaluate_ball_centers([], [], width=100, height=100, frame_count=1, **kwargs)


def test_detection_eval_cli_uses_centers_without_event_oracle(tmp_path):
    import json

    from typer.testing import CliRunner

    from juggletrack.cli import app
    from juggletrack.pipeline.offline import save_detections_jsonl
    from tests.helpers import write_test_video

    video = tmp_path / "v.mp4"
    write_test_video(video, n_frames=3, fps=30, size=(100, 100))
    csv = tmp_path / "centers.csv"
    csv.write_text("x,y\n50,50\n30,50\n0,0\n")
    dets = tmp_path / "detections.jsonl"
    save_detections_jsonl([point(0, .5), point(2, .8)], dets)
    out = tmp_path / "report.json"
    result = CliRunner().invoke(app, [
        "detection-eval", str(video), str(dets), str(csv), "--out", str(out),
    ])
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text())
    assert (report["tp"], report["fp"], report["fn"]) == (1, 1, 1)
    assert "Meschke" in report["citation"]
    assert "50.0%" in result.output


@pytest.mark.parametrize("case", ["out_of_range", "overwrite_csv", "overwrite_video",
                                  "overwrite_detections", "permitted_tail_trim"])
def test_detection_eval_validates_inputs_before_scoring_or_writing(tmp_path, case):
    from typer.testing import CliRunner

    from juggletrack.cli import app
    from juggletrack.pipeline.offline import save_detections_jsonl
    from tests.helpers import write_test_video

    video = tmp_path / "v.mp4"
    write_test_video(video, n_frames=4 if case == "permitted_tail_trim" else 3,
                     fps=30, size=(100, 100))
    csv = tmp_path / "centers.csv"
    csv.write_text("x,y\n50,50\n0,0\n0,0\n")
    dets = tmp_path / "detections.jsonl"
    predictions = [point(0, .5)]
    if case in {"out_of_range", "permitted_tail_trim"}:
        predictions.append(point(100 if case == "out_of_range" else 3, .8))
    save_detections_jsonl(predictions, dets)
    out = {"overwrite_csv": csv, "overwrite_video": video,
           "overwrite_detections": dets}.get(case, tmp_path / "report.json")
    before = {p: p.read_bytes() for p in [video, csv, dets]}
    result = CliRunner().invoke(app, [
        "detection-eval", str(video), str(dets), str(csv), "--out", str(out),
    ])
    assert result.exit_code == (0 if case == "permitted_tail_trim" else 2), result.output
    assert {p: p.read_bytes() for p in before} == before
