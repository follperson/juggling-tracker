"""Export a profile and proposals through the public commands, then replay them."""
import os

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from juggletrack.cli import app
from juggletrack.detect.registered import BallProfile
from juggletrack.pipeline.offline import load_detections_jsonl


def invoke(*args):
    return CliRunner().invoke(app, list(map(str, args)))


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "practice.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30., (320, 180))
    assert writer.isOpened()
    try:
        for i in range(12):
            frame = np.zeros((180, 320, 3), dtype=np.uint8)
            cv2.circle(frame, (50 + i*7, 90), 10, (10, 190, 50), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


def register(clip, path, *selection, frame=0):
    return invoke("balls", "register", clip, "--frame", frame, *selection, "--out", path)


def test_export_and_replay(clip, tmp_path):
    profile = tmp_path / "profiles" / "green.json"
    result = register(clip, profile, "--circle", 50, 90, 10)
    assert result.exit_code == 0, result.output
    assert BallProfile.model_validate_json(profile.read_bytes()).radius_fraction == 10/180
    out = tmp_path / "proposals" / "candidates.jsonl"
    result = invoke("balls", "propose", clip, "--profile", profile, "--profile", profile, "--out", out)
    assert result.exit_code == 0, result.output
    proposals = load_detections_jsonl(out)
    assert [p.frame_idx for p in proposals] == list(range(12))
    result = invoke("analyze", clip, "--detections", out, "--dots", "all", "--out", tmp_path/"review")
    assert result.exit_code == 0, result.output
    assert (tmp_path/"review"/"overlay.mp4").stat().st_size > 0


def test_picker_scales_upright_display_coordinates(monkeypatch, tmp_path, clip):
    from juggletrack.cli import _pick_circle

    seen = {}
    def pick(title, image, **kwargs):
        seen["shape"] = image.shape
        return 50, 100, 20, 20

    monkeypatch.setenv("DISPLAY", ":1")
    monkeypatch.setattr(cv2, "selectROI", pick)
    monkeypatch.setattr(cv2, "destroyWindow", lambda title: None)
    assert _pick_circle(np.zeros((1500, 1800, 3), np.uint8)) == (120, 220, 20)
    assert seen["shape"] == (750, 900, 3)
    monkeypatch.setattr(cv2, "selectROI", lambda *args, **kwargs: (40, 80, 20, 20))
    profile = tmp_path/"ball.json"
    assert register(clip, profile, "--pick").exit_code == 0
    assert BallProfile.model_validate_json(profile.read_bytes()).radius_fraction == 10/180


def test_registration_errors_preserve_outputs(clip, tmp_path):
    path = tmp_path / "previous.json"
    path.write_text("previous")
    for args, frame in [
        ([], 0), (["--pick", "--circle", 50, 90, 10], 0),
        (["--circle", 50, 90, 10], 80), (["--circle", 200, 90, 10], 0),
    ]:
        assert register(clip, path, *args, frame=frame).exit_code == 2
        assert path.read_text() == "previous"


def test_bad_profile_and_overlapping_inputs(clip, tmp_path):
    profile = tmp_path / "ball.json"
    assert register(clip, profile, "--circle", 50, 90, 10).exit_code == 0
    alias = tmp_path / "alias"
    os.link(profile, alias)
    assert invoke("balls", "propose", clip, "--profile", profile, "--out", alias).exit_code == 2
    assert register(clip, clip, "--circle", 50, 90, 10).exit_code == 2
    BallProfile.model_validate_json(profile.read_bytes())
    alias.unlink()
    alias.write_text("malformed")
    out = tmp_path / "out"
    out.write_text("previous")
    result = invoke("balls", "propose", clip, "--profile", alias, "--out", out)
    assert result.exit_code == 2 and "invalid ball profile" in result.output
    assert out.read_text() == "previous"


def test_proposal_failure_after_first_write_preserves_previous_output(clip, tmp_path, monkeypatch):
    profile = tmp_path / "ball.json"
    assert register(clip, profile, "--circle", 50, 90, 10).exit_code == 0
    import juggletrack.detect.registered as registered
    original = registered.registered_proposals

    def fails_after_one(*args):
        yield next(original(*args))
        raise ValueError("bad timestamp")

    monkeypatch.setattr(registered, "registered_proposals", fails_after_one)
    out = tmp_path / "out.jsonl"
    out.write_text("previous")
    result = invoke("balls", "propose", clip, "--profile", profile, "--out", out)
    assert result.exit_code == 2
    assert out.read_text() == "previous"



def test_no_matching_color_exports_empty_replay(clip, tmp_path):
    path = tmp_path / "red.json"
    path.write_text(BallProfile(hue=0, saturation_min=100, value_min=60,
                                radius_fraction=10/180).model_dump_json())
    out = tmp_path / "empty.jsonl"
    result = invoke("balls", "propose", clip, "--profile", path, "--out", out)
    assert result.exit_code == 0 and "No moving candidates" in result.output
    assert load_detections_jsonl(out) == []
