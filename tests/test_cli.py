import json

import pytest
from typer.testing import CliRunner

from juggletrack.pipeline.offline import save_detections_jsonl
from juggletrack.sim import simulate_cascade
from juggletrack.types import SessionResult
from tests.helpers import write_test_video

runner = CliRunner()


@pytest.fixture()
def workspace(tmp_path):
    sim = simulate_cascade(n_throws=12, fps=30.0, seed=1)
    n_frames = max(d.frame_idx for d in sim.detections) + 1
    video = tmp_path / "cascade.mp4"
    write_test_video(video, n_frames=n_frames, fps=30.0)
    dets = tmp_path / "dets.jsonl"
    save_detections_jsonl(sim.detections, dets)
    return sim, video, dets, tmp_path


def test_analyze_with_saved_detections(workspace):
    from juggletrack.cli import app

    sim, video, dets, tmp = workspace
    out = tmp / "out"
    result = runner.invoke(app, [
        "analyze", str(video), "--out", str(out), "--detections", str(dets),
    ])
    assert result.exit_code == 0, result.output
    sr = SessionResult.model_validate(json.loads((out / "analysis.json").read_text()))
    assert sr.runs and sr.runs[0].catches == 12
    assert (out / "overlay.mp4").exists()
    assert "catches" in result.output


def test_analyze_no_overlay(workspace):
    from juggletrack.cli import app

    _, video, dets, tmp = workspace
    out = tmp / "out2"
    result = runner.invoke(app, [
        "analyze", str(video), "--out", str(out),
        "--detections", str(dets), "--no-overlay",
    ])
    assert result.exit_code == 0, result.output
    assert (out / "analysis.json").exists()
    assert not (out / "overlay.mp4").exists()


def test_eval_command(workspace, tmp_path):
    from juggletrack.cli import app

    sim, video, dets, tmp = workspace
    out = tmp / "out3"
    runner.invoke(app, [
        "analyze", str(video), "--out", str(out),
        "--detections", str(dets), "--no-overlay",
    ])
    labels = {
        "video": str(video),
        "runs": [{"start_t": sim.run_start, "end_t": sim.run_end, "catches": 12}],
        "drops": [],
    }
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels))
    report_path = tmp_path / "eval.json"
    result = runner.invoke(app, [
        "eval", str(out / "analysis.json"), str(labels_path),
        "--out", str(report_path),
    ])
    assert result.exit_code == 0, result.output
    report = json.loads(report_path.read_text())
    assert report["frac_catch_within_1"] == 1.0
    assert "catch" in result.output.lower()


def test_label_command_with_saved_detections(workspace):
    from juggletrack.cli import app

    sim, video, dets, tmp = workspace
    out = tmp / "labels_out"
    result = runner.invoke(app, [
        "label", str(video), "--out", str(out), "--detections", str(dets),
    ])
    assert result.exit_code == 0, result.output
    assert (out / "annotations.json").exists()
    assert (out / "review_manifest.json").exists()
    assert any((out / "images").iterdir())
    assert "boxes" in result.output and "review" in result.output


def test_coverage_command(workspace):
    from juggletrack.cli import app

    sim, video, dets, tmp = workspace
    result = runner.invoke(app, [
        "coverage", str(video), "--detections", str(dets),
    ])
    assert result.exit_code == 0, result.output
    assert f"detections {len(sim.detections)}" in result.output
    assert ">=2:" in result.output and ">=3:" in result.output
