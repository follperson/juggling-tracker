import hashlib
import json
import os

import pytest
from typer.testing import CliRunner

from juggletrack.cli import app
from juggletrack.pipeline.offline import save_detections_jsonl
from juggletrack.types import Detection
from juggletrack.video.reader import VideoReader
from tests.helpers import write_test_video


@pytest.fixture
def review_inputs(tmp_path):
    video = tmp_path / "practice.mp4"
    centers = write_test_video(video, n_frames=4, size=(160, 96))
    with VideoReader(video) as reader:
        frames = [{"frame_idx": i, "t": t,
                   "centers": [{"x": centers[i][0], "y": centers[i][1], "state": "airborne"}]}
                  for i, t, _ in reader.frames()]
    review = tmp_path / "review.json"
    review.write_text(json.dumps({
        "schema_version": 1, "review_method": "synthetic",
        "source_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "width": 160, "height": 96, "frames": [frames[0], frames[3]],
    }))
    dets = tmp_path / "detections.jsonl"
    save_detections_jsonl([Detection(frame_idx=0, t=0, x=centers[0][0], y=centers[0][1]),
                           Detection(frame_idx=1, t=frames[1]["t"], x=.1, y=.9)], dets)
    return video, dets, review


def run(inputs, out):
    return CliRunner().invoke(app, ["reviewed-detection-eval", *map(str, inputs),
                                    "--tolerance-px", "5", "--out", str(out)])


def test_scored_recording_can_be_reproduced_without_detector(review_inputs, tmp_path):
    out = tmp_path / "reports" / "scored.json"
    result = run(review_inputs, out)
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text())
    assert report["review_method"] == "synthetic"
    assert (report["tp"], report["fp"], report["fn"]) == (1, 0, 1)
    assert report["states"]["airborne"]["recall"] == .5
    for path, key in zip(review_inputs, ("source_sha256", "detections_sha256", "labels_sha256")):
        assert report[key] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(report["implementation_sha256"]) == 64


@pytest.mark.parametrize("invalid", ["source", "dimensions", "pts_on_missed_frame",
                                     "missing_frame", "detection_beyond_end", "output_alias"])
def test_source_and_clock_checked_before_output(review_inputs, tmp_path, invalid):
    _, dets, labels = review_inputs
    out = tmp_path / "report.json"
    doc = json.loads(labels.read_text())
    if invalid == "source":
        doc["source_sha256"] = "b" * 64
    elif invalid == "dimensions":
        doc["height"], doc["width"] = doc["width"], doc["height"]
    elif invalid == "pts_on_missed_frame":
        doc["frames"][-1]["t"] += .02
    elif invalid == "missing_frame":
        doc["frames"][-1]["frame_idx"] = 9
    elif invalid == "detection_beyond_end":
        save_detections_jsonl([Detection(frame_idx=4, t=.2, x=.1, y=.2)], dets)
    elif invalid == "output_alias":
        os.link(labels, out)
    labels.write_text(json.dumps(doc))
    before = [p.read_bytes() for p in review_inputs]
    result = run(review_inputs, out)
    assert result.exit_code == 2, result.output
    assert [p.read_bytes() for p in review_inputs] == before
    assert out.exists() == (invalid == "output_alias")


def test_reports_oversized_box_at_correct_center_through_cli(review_inputs, tmp_path):
    _, detections, labels = review_inputs
    doc = json.loads(labels.read_bytes())
    ball = doc["frames"][0]["centers"][0]
    x, y = ball["x"], ball["y"]
    ball.update(state="visible", visible_box={
        "left": x - .025, "top": y - .025, "right": x + .025, "bottom": y + .025,
    })
    labels.write_text(json.dumps(doc))
    save_detections_jsonl([Detection(frame_idx=0, t=0, x=x, y=y, w=.20, h=.20)], detections)
    out = tmp_path/"report.json"
    result = run(review_inputs, out)
    assert result.exit_code == 0, result.output
    assert "Definite centers matched 1/2" in result.output
    assert "labeled area 1/1" in result.output
    report = json.loads(out.read_bytes())
    assert report["box_localization"]["targets"][0]["area_ratio"] == pytest.approx(16)
