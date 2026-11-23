import json
import re

import numpy as np
import pytest
from typer.testing import CliRunner

from juggletrack.detect.fake import FakeDetector
from juggletrack.pipeline.offline import save_detections_jsonl
from juggletrack.sim import simulate_cascade
from juggletrack.types import Detection, SessionResult
from juggletrack.video.reader import VideoReader
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


def _replay_as_yolo(monkeypatch, dets):
    monkeypatch.setattr(
        "juggletrack.detect.yolo.YOLODetector",
        lambda model_path, **kwargs: FakeDetector(dets),
    )


def _overlay_frames(out_dir):
    with VideoReader(out_dir / "overlay.mp4") as reader:
        return [frame for _, _, frame in reader.frames()]


def test_analyze_overlay_draws_the_detections_it_analysed(workspace, monkeypatch):
    from juggletrack.cli import app

    sim, video, _, tmp = workspace
    _replay_as_yolo(monkeypatch, sim.detections)
    for dots in ("all", "none"):
        result = runner.invoke(app, [
            "analyze", str(video), "--model", "x.pt", "--out", str(tmp / dots), "--dots", dots,
        ])
        assert result.exit_code == 0, result.output
    with_dots, without_dots = _overlay_frames(tmp / "all"), _overlay_frames(tmp / "none")
    assert any(not np.array_equal(a, b) for a, b in zip(with_dots, without_dots))


def test_analyze_overlay_ignores_stale_detections_in_out_dir(workspace, monkeypatch):
    from juggletrack.cli import app

    sim, video, _, tmp = workspace
    _replay_as_yolo(monkeypatch, sim.detections)
    stale = tmp / "stale"
    stale.mkdir()
    save_detections_jsonl(
        [Detection(frame_idx=i, t=i / 30.0, x=0.02, y=0.02) for i in range(5)],
        stale / "detections.jsonl",
    )
    for out in (stale, tmp / "clean"):
        result = runner.invoke(app, [
            "analyze", str(video), "--model", "x.pt", "--out", str(out), "--dots", "all",
        ])
        assert result.exit_code == 0, result.output
    stale_frames, clean_frames = _overlay_frames(stale), _overlay_frames(tmp / "clean")
    assert all(np.array_equal(a, b) for a, b in zip(stale_frames, clean_frames))


def test_analyze_replay_without_overlay_decodes_no_frames(workspace, monkeypatch):
    from juggletrack.cli import app

    _, video, dets, tmp = workspace

    def no_decoding(self):
        raise AssertionError("a frame was decoded")

    monkeypatch.setattr(VideoReader, "frames", no_decoding)
    out = tmp / "replay"
    result = runner.invoke(app, [
        "analyze", str(video), "--out", str(out), "--detections", str(dets), "--no-overlay",
    ])
    assert result.exit_code == 0, result.output
    sr = SessionResult.model_validate_json((out / "analysis.json").read_text())
    assert [r.catches for r in sr.runs] == [12]


def flat(output: str) -> str:
    return " ".join(plain(output).replace("│", " ").split())


@pytest.mark.parametrize("command", ["analyze", "live", "label", "coverage"])
def test_detections_past_the_end_of_the_video_are_rejected(workspace, command):
    from juggletrack.cli import app

    sim, video, _, tmp = workspace
    longer = tmp / "longer.jsonl"
    with VideoReader(video) as reader:
        frame_count = reader.info.frame_count
    beyond = Detection(frame_idx=frame_count, t=frame_count / 30.0, x=0.5, y=0.5)
    save_detections_jsonl([*sim.detections, beyond], longer)
    args = [command, str(video), "--detections", str(longer)]
    if command != "coverage":
        args += ["--out", str(tmp / "out")]
    if command == "live":
        args.append("--no-display")
    result = runner.invoke(app, args)
    output = flat(result.output)
    assert result.exit_code == 2, output
    assert re.search(r"Invalid value for '?--detections'?", output), output
    assert f"frame {frame_count}" in output and f"{frame_count} frames" in output


class _NeverDetects:
    def detect(self, frame, frame_idx, t):
        raise AssertionError("detection started before --out was checked")


@pytest.mark.parametrize("command", ["analyze", "label", "live"])
def test_unusable_out_fails_before_detection(workspace, monkeypatch, command):
    from juggletrack.cli import app

    _, video, _, tmp = workspace
    monkeypatch.setattr(
        "juggletrack.detect.yolo.YOLODetector", lambda model_path, **kwargs: _NeverDetects(),
    )
    taken = tmp / "taken"
    taken.write_text("")
    args = [command, str(video), "--model", "x.pt", "--out", str(taken)]
    if command == "live":
        args.append("--no-display")
    result = runner.invoke(app, args)
    output = flat(result.output)
    assert result.exit_code == 2, output
    assert re.search(r"Invalid value for '?--out'?", output), output


@pytest.mark.parametrize(("command", "help_text"), [
    ("analyze", "Output directory (default outputs/<stem>)"),
    ("label", "Output directory (default outputs/labels/<stem>)"),
    ("live", "Directory to write live_session.json into"),
])
def test_out_help_says_it_is_a_directory(command, help_text):
    from juggletrack.cli import app

    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0
    assert help_text in flat(result.output)


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


@pytest.mark.parametrize("value", ["inf", "nan"])
def test_analyze_rejects_non_finite_link_distance_as_usage_error(workspace, value):
    from juggletrack.cli import app

    _, video, dets, tmp = workspace
    out = tmp / "out_nonfinite"
    result = runner.invoke(app, [
        "analyze", str(video), "--out", str(out), "--detections", str(dets),
        "--link-max-dist", value,
    ])
    output = plain(result.output)
    assert result.exit_code == 2, output
    assert re.search(r"Invalid value for '?--link-max-dist'?: Input should be a finite number",
                     output), output
    assert not out.exists()


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


def expected_coverage(dets, n_sampled):
    from collections import Counter

    per_frame = Counter(d.frame_idx for d in dets)
    return [
        f"{sum(1 for v in per_frame.values() if v >= k) / n_sampled:.1%}" for k in (1, 2, 3)
    ]


def printed_coverage(output):
    match = re.search(r"coverage >=1: (\S+)  >=2: (\S+)  >=3: (\S+)", output)
    assert match, output
    return list(match.groups())


def test_coverage_command(workspace):
    from juggletrack.cli import app

    sim, video, dets, tmp = workspace
    result = runner.invoke(app, [
        "coverage", str(video), "--detections", str(dets),
    ])
    assert result.exit_code == 0, result.output
    assert f"detections {len(sim.detections)} over 197 sampled frames" in result.output
    assert printed_coverage(result.output) == expected_coverage(sim.detections, 197)


def test_coverage_with_stride_scores_the_sampled_frames(workspace):
    from juggletrack.cli import app

    sim, video, _, tmp = workspace
    even = [d for d in sim.detections if d.frame_idx % 2 == 0]
    strided = tmp / "stride2.jsonl"
    save_detections_jsonl(even, strided)
    result = runner.invoke(app, [
        "coverage", str(video), "--detections", str(strided), "--stride", "2",
    ])
    assert result.exit_code == 0, result.output
    assert "over 99 sampled frames" in result.output
    assert printed_coverage(result.output) == expected_coverage(even, 99)


def test_coverage_rejects_detections_off_the_stride_grid(workspace):
    from juggletrack.cli import app

    sim, video, _, tmp = workspace
    strided = tmp / "stride2.jsonl"
    save_detections_jsonl([d for d in sim.detections if d.frame_idx % 2 == 0], strided)
    result = runner.invoke(app, [
        "coverage", str(video), "--detections", str(strided), "--stride", "3",
    ])
    output = flat(result.output)
    assert result.exit_code == 2, output
    assert re.search(r"Invalid value for '?--stride'?", output), output
    assert "not on the --stride 3 grid" in output


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
    import shutil

    import juggletrack.pipeline.live as live_module
    from juggletrack.cli import app
    from juggletrack.pipeline.realtime import RealtimeState

    _, video, dets, tmp = workspace
    shutil.copy(video, tmp / "²")
    monkeypatch.chdir(tmp)
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


@pytest.mark.parametrize(("args", "environment", "named"), [
    (["--model", "models/juggletrack-v9/best.pt"], None, "--model"),
    ([], "/nonexistent/best.pt", "JUGGLETRACK_MODEL"),
])
def test_missing_weights_path_is_a_usage_error(workspace, monkeypatch, args, environment, named):
    from juggletrack.cli import app

    def unexpected_model_load(*args, **kwargs):
        raise AssertionError("a missing weights path must fail before the detector is built")

    monkeypatch.setattr("juggletrack.detect.yolo.YOLODetector", unexpected_model_load)
    _, video, _, tmp = workspace
    monkeypatch.chdir(tmp)
    monkeypatch.delenv("JUGGLETRACK_MODEL", raising=False)
    if environment is not None:
        monkeypatch.setenv("JUGGLETRACK_MODEL", environment)
    result = runner.invoke(app, ["analyze", str(video), "--no-overlay", *args])
    output = flat(result.output)
    assert result.exit_code == 2, output
    assert f"{named} weights not found" in output


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


@pytest.mark.parametrize(("command", "option", "value"), [
    (command, option, value)
    for command in ("analyze", "label", "coverage", "live")
    for option, value in (("--conf", "5"), ("--conf", "-0.1"), ("--imgsz", "0"))
] + [("live", "--max-frames", "-5"), ("live", "--max-frames", "0")])
def test_out_of_range_detector_options_are_usage_errors(
    workspace, monkeypatch, command, option, value,
):
    from juggletrack.cli import app

    def unexpected_model_load(*args, **kwargs):
        raise AssertionError("an out-of-range option must fail before the detector is built")

    monkeypatch.setattr("juggletrack.detect.yolo.YOLODetector", unexpected_model_load)
    _, video, _, tmp = workspace
    args = [command, str(video), "--model", "x.pt", option, value]
    if command != "coverage":
        args += ["--out", str(tmp / "out")]
    if command == "live":
        args.append("--no-display")
    result = runner.invoke(app, args)
    output = flat(result.output)
    assert result.exit_code == 2, output
    assert re.search(rf"Invalid value for '?{option}'?", output), output


def test_analyze_warns_when_the_video_ends_early(workspace, monkeypatch):
    from juggletrack.cli import app
    from tests.test_pipeline_offline import decode_only_first

    sim, video, _, tmp = workspace
    _replay_as_yolo(monkeypatch, sim.detections)
    args = ["analyze", str(video), "--model", "x.pt", "--no-overlay"]
    full = runner.invoke(app, [*args, "--out", str(tmp / "full")])
    assert full.exit_code == 0, full.output
    assert "warning" not in full.stderr

    decode_only_first(monkeypatch, 100)
    cut = runner.invoke(app, [*args, "--out", str(tmp / "cut")])
    assert cut.exit_code == 0, cut.output
    assert "warning: only decoded 100/197 frames" in cut.stderr
    assert cut.stdout.splitlines()[-1].startswith("1 run(s)")


@pytest.mark.parametrize("replay", [False, True])
def test_live_missing_file_source_is_a_usage_error(workspace, monkeypatch, replay):
    from juggletrack.cli import app

    sim, _, dets, tmp = workspace
    _replay_as_yolo(monkeypatch, sim.detections)
    missing = tmp / "nope.mp4"
    source = ["--detections", str(dets)] if replay else ["--model", "x.pt"]
    result = runner.invoke(app, ["live", str(missing), *source, "--no-display"],
                           env={"COLUMNS": "1000"})
    output = flat(result.output)
    assert result.exit_code == 2, output
    assert re.search(r"Invalid value for '?source'?", output, re.IGNORECASE), output
    assert str(missing) in output and "does not exist" in output


def _fake_capture(monkeypatch, *, opens=True, frames=3):
    """Replace cv2.VideoCapture in the live loop; return the sources it was asked to open."""
    import cv2

    import juggletrack.pipeline.live as live_module

    sources = []

    class Capture:
        def __init__(self, source):
            sources.append(source)
            self.left = frames

        def isOpened(self):
            return opens

        def get(self, prop):
            return 30.0 if prop == cv2.CAP_PROP_FPS else 0.0

        def read(self):
            if not self.left:
                return False, None
            self.left -= 1
            return True, np.zeros((48, 64, 3), np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(live_module.cv2, "VideoCapture", Capture)
    return sources


@pytest.mark.parametrize("source", ["http://127.0.0.1:8765/clip.mp4", "seq/f_%03d.png"])
def test_live_hands_a_stream_or_image_sequence_to_the_capture(monkeypatch, source):
    from juggletrack.cli import app

    _replay_as_yolo(monkeypatch, [])
    opened = _fake_capture(monkeypatch)
    result = runner.invoke(app, ["live", source, "--model", "x.pt", "--no-display"])
    assert result.exit_code == 0, result.output
    assert opened == [source]
    assert "0 runs, 0 catches, 0 drops" in result.output


@pytest.mark.parametrize("source", ["http://127.0.0.1:8765/clip.mp4", "seq/f_%03d.png"])
def test_live_stream_that_will_not_open_is_a_usage_error(monkeypatch, source):
    from juggletrack.cli import app

    _replay_as_yolo(monkeypatch, [])
    _fake_capture(monkeypatch, opens=False)
    result = runner.invoke(app, ["live", source, "--model", "x.pt", "--no-display"],
                           env={"COLUMNS": "1000"})
    output = flat(result.output)
    assert result.exit_code == 2, output
    assert re.search(r"Invalid value for '?source'?", output, re.IGNORECASE), output
    assert f"could not open source {source}" in output
    assert "does not exist" not in output and "camera" not in output


def test_live_detector_error_on_an_open_camera_is_not_blamed_on_the_camera(monkeypatch):
    from juggletrack.cli import app

    class Boom:
        def detect(self, frame, frame_idx, t):
            raise ValueError("expected 3 channels but got 1")

    monkeypatch.setattr("juggletrack.detect.yolo.YOLODetector", lambda model_path, **kw: Boom())
    _fake_capture(monkeypatch)
    result = runner.invoke(app, ["live", "0", "--model", "x.pt", "--no-display"])
    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, ValueError)
    assert str(result.exception) == "expected 3 channels but got 1"
    assert "camera" not in result.output


def test_live_camera_that_will_not_open_is_a_usage_error(workspace, monkeypatch):
    import juggletrack.pipeline.live as live_module
    from juggletrack.cli import app

    class ClosedCapture:
        def __init__(self, source):
            pass

        def isOpened(self):
            return False

    sim, _, _, _ = workspace
    _replay_as_yolo(monkeypatch, sim.detections)
    monkeypatch.setattr(live_module.cv2, "VideoCapture", ClosedCapture)
    result = runner.invoke(app, ["live", "7", "--model", "x.pt", "--no-display"])
    output = flat(result.output)
    assert result.exit_code == 2, output
    assert re.search(r"Invalid value for '?source'?", output, re.IGNORECASE), output
    assert "could not open source: 7" in output and "camera" in output


def test_detections_with_a_non_finite_value_are_a_usage_error(workspace):
    from juggletrack.cli import app

    sim, video, _, tmp = workspace
    bad = tmp / "nan.jsonl"
    save_detections_jsonl(sim.detections[:3], bad)
    with bad.open("a") as f:
        f.write('{"frame_idx": 20, "t": 0.67, "x": NaN, "y": 0.5}\n')
    result = runner.invoke(app, [
        "analyze", str(video), "--detections", str(bad), "--out", str(tmp / "out"),
    ], env={"COLUMNS": "1000"})
    output = flat(result.output)
    assert result.exit_code == 2, output
    assert re.search(r"Invalid value for '?--detections'?", output), output
    assert f"{bad}:4: x is not finite (nan)" in output


def test_version_flag_prints_the_installed_version():
    from importlib.metadata import version

    from juggletrack.cli import app

    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == f"juggletrack {version('juggletrack')}"


def test_package_runs_as_a_module():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "juggletrack", "--help"], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "analyze" in result.stdout


def test_analyze_has_a_help_summary():
    from juggletrack.cli import app

    root = runner.invoke(app, ["--help"], env={"COLUMNS": "200"})
    listed = next(line for line in plain(root.output).splitlines() if " analyze " in line)
    assert listed.replace("│", "").split()[1:], listed
    own = runner.invoke(app, ["analyze", "--help"])
    assert listed.replace("│", "").split()[1] in plain(own.output)


def _analyze_replay(video, dets, out, *extra):
    from juggletrack.cli import app

    result = runner.invoke(app, [
        "analyze", str(video), "--detections", str(dets), "--out", str(out), "--no-overlay",
        *extra,
    ])
    assert result.exit_code == 0, result.output
    session = SessionResult.model_validate_json((out / "analysis.json").read_text())
    return result.output.splitlines(), session


def test_analyze_summary_reports_each_run_and_the_totals(tmp_path):
    sim = simulate_cascade(n_throws=20, fps=30.0, drop_at_throw=8, seed=1)
    video = tmp_path / "drop.mp4"
    write_test_video(video, n_frames=max(d.frame_idx for d in sim.detections) + 1, fps=30.0)
    dets = tmp_path / "drop.jsonl"
    save_detections_jsonl(sim.detections, dets)
    out = tmp_path / "out"
    lines, session = _analyze_replay(video, dets, out)

    assert any(r.end_reason == "drop" and r.catches != r.throws for r in session.runs)
    run_lines = [line for line in lines if line.startswith("run ")]
    assert [int(re.search(r"catches (\d+)", line).group(1)) for line in run_lines] == [
        r.catches for r in session.runs
    ]
    assert [re.search(r"end=(\w+)", line).group(1) for line in run_lines] == [
        r.end_reason for r in session.runs
    ]
    catches, throws = (sum(r.catches for r in session.runs), sum(r.throws for r in session.runs))
    assert f"total: {catches} catches, {throws} throws" in lines
    assert lines[-1] == f"{len(session.runs)} run(s), {len(session.drops)} drop(s); results in {out}"


def test_analyze_summary_marks_a_run_that_began_before_the_clip(tmp_path):
    sim = simulate_cascade(n_throws=12, fps=30.0, seed=1)
    cut = [
        d.model_copy(update={"frame_idx": d.frame_idx - 30, "t": d.t - 1.0})
        for d in sim.detections if d.frame_idx >= 30
    ]
    video = tmp_path / "cut.mp4"
    write_test_video(video, n_frames=max(d.frame_idx for d in cut) + 1, fps=30.0)
    dets = tmp_path / "cut.jsonl"
    save_detections_jsonl(cut, dets)
    lines, session = _analyze_replay(video, dets, tmp_path / "out")

    assert session.runs[0].start_t < 0
    assert lines[0] == (
        f"run 1: 0.00s to {session.runs[0].end_t:.2f}s (began before the clip)  "
        f"catches {session.runs[0].catches}  end={session.runs[0].end_reason}"
    )


@pytest.fixture()
def sparse_workspace(tmp_path):
    """Every 4th frame of a 6-throw cascade: the default linker distance finds
    no arcs here and a wider one finds them, so the knob visibly matters."""
    sim = simulate_cascade(n_throws=6, fps=30.0, seed=1)
    video = tmp_path / "six.mp4"
    write_test_video(video, n_frames=max(d.frame_idx for d in sim.detections) + 1, fps=30.0)
    dets = tmp_path / "sparse.jsonl"
    save_detections_jsonl([d for d in sim.detections if d.frame_idx % 4 == 0], dets)
    return video, dets, tmp_path


def test_analyze_link_max_dist_reaches_the_linker(sparse_workspace):
    video, dets, tmp = sparse_workspace
    arcs = {
        dist: len(_analyze_replay(video, dets, tmp / dist, "--link-max-dist", dist)[1].arcs)
        for dist in ("0.08", "0.2")
    }
    assert arcs["0.08"] < arcs["0.2"], arcs


def test_label_link_max_dist_reaches_the_linker(sparse_workspace):
    from juggletrack.cli import app

    video, dets, tmp = sparse_workspace
    boxes = {}
    for dist in ("0.08", "0.2"):
        result = runner.invoke(app, [
            "label", str(video), "--detections", str(dets), "--out", str(tmp / dist),
            "--link-max-dist", dist,
        ])
        assert result.exit_code == 0, result.output
        boxes[dist] = int(re.search(r"(\d+) boxes", result.output).group(1))
    assert boxes["0.08"] < boxes["0.2"], boxes


def test_live_max_frames_stops_the_replay_early(workspace):
    from juggletrack.cli import app

    _, video, dets, tmp = workspace
    out = tmp / "live_short"
    result = runner.invoke(app, [
        "live", str(video), "--detections", str(dets), "--no-display",
        "--max-frames", "60", "--out", str(out),
    ])
    assert result.exit_code == 0, result.output
    saved = json.loads((out / "live_session.json").read_text())
    assert saved["t"] < 2.0
    assert saved["catches_total"] < 12
