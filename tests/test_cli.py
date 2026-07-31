import json

import numpy as np
import pytest
from typer.testing import CliRunner

from juggletrack.pipeline.offline import save_detections_jsonl
from juggletrack.sim import simulate_cascade
from juggletrack.types import Detection, SessionResult
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


@pytest.fixture()
def workspace_with_junk(tmp_path):
    """Same as `workspace`, plus a persistent background false positive (see
    tests/test_autolabel.py's `_persistent_junk_cascade`) parked at (0.91,
    0.71) for the whole video, AND a junk-only idle tail well past the run
    (activity windows exclude near-run frames from negative mining, so the
    lead-in junk alone no longer yields candidates -- the idle tail is what
    `--negatives` mines).
    """
    sim = simulate_cascade(n_throws=12, fps=30.0, seed=1)
    real = list(sim.detections)
    fps = 30.0
    run_frames = max(d.frame_idx for d in real) + 1
    # idle tail: junk continues 2s after the run ends, for 30 frames
    tail_start = run_frames + int(2.0 * fps)
    n_frames = tail_start + 30
    rng = np.random.default_rng(11)
    junk = []
    for i in list(range(run_frames)) + list(range(tail_start, n_frames)):
        t = i / fps
        for _ in range(2):
            x = 0.91 + float(rng.uniform(-0.005, 0.005))
            y = 0.71 + float(rng.uniform(-0.005, 0.005))
            junk.append(Detection(frame_idx=i, t=t, x=x, y=y))
    video = tmp_path / "cascade.mp4"
    write_test_video(video, n_frames=n_frames, fps=fps)
    dets = tmp_path / "dets.jsonl"
    save_detections_jsonl(real + junk, dets)
    return sim, video, dets, tmp_path


def test_analyze_with_saved_detections(workspace):
    from juggletrack.cli import app

    sim, video, dets, tmp = workspace
    out = tmp / "out"
    result = runner.invoke(app, [
        "analyze", str(video), "--out", str(out), "--detections", str(dets),
        "--link-max-dist", "0.2",
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


def test_analyze_dots_option_passthrough(workspace):
    from juggletrack.cli import app

    sim, video, dets, tmp = workspace
    out = tmp / "out_dots"
    result = runner.invoke(app, [
        "analyze", str(video), "--out", str(out), "--detections", str(dets),
        "--link-max-dist", "0.2", "--dots", "all",
    ])
    assert result.exit_code == 0, result.output
    assert (out / "overlay.mp4").exists()


def test_analyze_dots_invalid_choice_rejected(workspace):
    from juggletrack.cli import app

    _, video, dets, tmp = workspace
    out = tmp / "out_dots_bad"
    result = runner.invoke(app, [
        "analyze", str(video), "--out", str(out), "--detections", str(dets),
        "--dots", "bogus",
    ])
    assert result.exit_code != 0


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


def test_label_command_negatives_option(workspace_with_junk):
    from juggletrack.cli import app

    sim, video, dets, tmp = workspace_with_junk
    out = tmp / "labels_out"
    negatives = tmp / "negatives_out"
    result = runner.invoke(app, [
        "label", str(video), "--out", str(out), "--detections", str(dets),
        "--negatives", str(negatives),
        # 'none' keeps the test hermetic (no stock-model download); the CLI
        # default is yolo11n.pt so field mining gets the person veto for free
        "--neg-person-model", "none",
    ])
    assert result.exit_code == 0, result.output
    # the normal label export still happens
    assert (out / "annotations.json").exists()
    # plus the hard-negative export
    assert (negatives / "annotations.json").exists()
    assert (negatives / "negatives_manifest.json").exists()
    coco = json.loads((negatives / "annotations.json").read_text())
    assert coco["annotations"] == []
    assert coco["images"]
    assert coco["categories"] == [{"id": 1, "name": "ball"}]
    for im in coco["images"]:
        assert (negatives / "images" / im["file_name"]).exists()
    assert "negatives" in result.output.lower()
    # the CLI echo must surface n_skipped_active too, not just
    # n_skipped_ambiguous -- both are useful diagnostics for why a
    # candidate frame count came out lower than expected.
    assert "skipped active" in result.output.lower()
    # ...and the contamination-gate counters (transient/floor/person), so a
    # mining run that rejected real-ball frames says so instead of silently
    # yielding fewer images.
    assert "transient" in result.output.lower()
    assert "person" in result.output.lower()


def test_label_command_without_negatives_skips_export(workspace):
    from juggletrack.cli import app

    _, video, dets, tmp = workspace
    out = tmp / "labels_out2"
    result = runner.invoke(app, [
        "label", str(video), "--out", str(out), "--detections", str(dets),
    ])
    assert result.exit_code == 0, result.output
    assert "negatives" not in result.output.lower()


def test_coverage_command(workspace):
    from juggletrack.cli import app

    sim, video, dets, tmp = workspace
    result = runner.invoke(app, [
        "coverage", str(video), "--detections", str(dets),
    ])
    assert result.exit_code == 0, result.output
    assert f"detections {len(sim.detections)}" in result.output
    assert ">=2:" in result.output and ">=3:" in result.output
