import json
import re

import numpy as np
import pytest
from typer.testing import CliRunner

from juggletrack.pipeline.offline import save_detections_jsonl
from juggletrack.sim import simulate_cascade
from juggletrack.types import Detection, SessionResult
from tests.helpers import write_test_video

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def plain(output: str) -> str:
    """Rich splits a styled option token, so "--model" is absent from coloured bytes."""
    return _ANSI.sub("", output)


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


def test_live_command_file_replay(workspace):
    from juggletrack.cli import app

    sim, video, dets, tmp = workspace
    out = tmp / "live_out"
    result = runner.invoke(app, [
        "live", str(video), "--detections", str(dets),
        "--no-display", "--out", str(out),
    ])
    assert result.exit_code == 0, result.output
    assert "catches" in result.output and "fps" in result.output
    saved = json.loads((out / "live_session.json").read_text())
    assert saved["catches_total"] == 12


def test_live_command_detections_with_webcam_index_rejected(workspace):
    """S7: --detections replays a saved detections.jsonl keyed by frame_idx
    against FILE-source frame timestamps (see FakeDetector); a webcam
    index has no frames to replay against."""
    from juggletrack.cli import app

    _, _, dets, _ = workspace
    result = runner.invoke(app, ["live", "0", "--detections", str(dets), "--no-display"])
    assert result.exit_code != 0
    assert "file source" in result.output.lower()


def test_live_command_non_ascii_digit_source_stays_a_string(workspace, monkeypatch):
    """S4: '²' (superscript 2) is str.isdigit()==True but not ASCII;
    requiring isascii() too keeps it as a file-path-shaped string instead
    of either crashing int() or (for other non-ASCII digits int() accepts,
    like full-width '３') silently misinterpreting it as a webcam
    index. run_live is monkeypatched here purely to avoid actually trying
    to open a bogus source -- this test is about argument parsing, not
    video I/O."""
    import juggletrack.pipeline.live as live_module
    from juggletrack.cli import app
    from juggletrack.pipeline.realtime import RealtimeState

    _, _, dets, _ = workspace
    captured = {}

    def fake_run_live(src, detector, **kwargs):
        captured["src"] = src
        return RealtimeState(t=0.0), 0.0

    monkeypatch.setattr(live_module, "run_live", fake_run_live)
    result = runner.invoke(app, [
        "live", "²", "--detections", str(dets), "--no-display",
    ])
    assert result.exit_code == 0, result.output
    assert captured["src"] == "²"


@pytest.mark.parametrize("command", ["analyze", "live", "coverage", "label"])
def test_inference_commands_explain_missing_default_model(workspace, monkeypatch, command):
    from juggletrack.cli import app

    def unexpected_model_load(*args, **kwargs):
        raise AssertionError("missing defaults must fail before loading/downloading weights")

    monkeypatch.setattr("juggletrack.detect.yolo.YOLODetector", unexpected_model_load)
    _, video, _, tmp = workspace
    monkeypatch.chdir(tmp)
    monkeypatch.delenv("JUGGLETRACK_MODEL", raising=False)
    result = runner.invoke(app, [command, str(video)])
    output = " ".join(plain(result.output).replace("│", " ").split())
    assert result.exit_code == 2, output
    # No --model was passed, so the error must not blame that option.
    assert not re.search(r"Invalid value for '?--model", output)
    assert "relative to the current directory" in output
    assert "--model" in output
    assert "JUGGLETRACK_MODEL" in output


@pytest.mark.parametrize(("args", "environment", "expected"), [
    (["--model", "x.pt"], None, "x.pt"),
    ([], "y.pt", "y.pt"),
    (["--model", "x.pt"], "y.pt", "x.pt"),
])
def test_analyze_forwards_selected_model_to_detector(
    workspace, monkeypatch, args, environment, expected,
):
    from juggletrack.cli import app
    from juggletrack.detect.fake import FakeDetector

    sim, video, _, tmp = workspace
    captured = {}

    def capturing_detector(model_path, **kwargs):
        captured["model_path"] = model_path
        return FakeDetector(sim.detections)

    monkeypatch.setattr("juggletrack.detect.yolo.YOLODetector", capturing_detector)
    monkeypatch.chdir(tmp)
    monkeypatch.delenv("JUGGLETRACK_MODEL", raising=False)
    if environment is not None:
        monkeypatch.setenv("JUGGLETRACK_MODEL", environment)
    result = runner.invoke(app, [
        "analyze", str(video), "--out", str(tmp / "out"), "--no-overlay", *args,
    ])
    assert result.exit_code == 0, result.output
    assert captured["model_path"] == expected
