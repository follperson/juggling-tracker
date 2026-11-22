import json
from collections import Counter

import numpy as np
from typer.testing import CliRunner

from juggletrack.pipeline.offline import save_detections_jsonl
from juggletrack.sim import simulate_cascade
from tests.helpers import write_test_video

runner = CliRunner()


def _frame_of(file_name: str) -> int:
    return int(file_name.removeprefix("frame_").removesuffix(".jpg"))


def test_label_writes_one_box_per_ball_when_detector_echoes(tmp_path):
    # Every real ball also gets a weaker echo box 0.02 to its right. On this
    # input the raw extraction mints parallel ghost arcs, so keeping one label
    # per arc is not enough on its own: the ghosts must be collapsed too.
    from juggletrack.cli import app

    sim = simulate_cascade(n_throws=12, fps=30.0, seed=1)
    dets = []
    for d in sim.detections:
        dets.append(d.model_copy(update={"confidence": 0.9}))
        dets.append(d.model_copy(update={"x": d.x + 0.02, "confidence": 0.4}))
    n_frames = max(d.frame_idx for d in dets) + 1
    video = tmp_path / "cascade.mp4"
    write_test_video(video, n_frames=n_frames, fps=30.0)
    det_path = tmp_path / "dets.jsonl"
    save_detections_jsonl(dets, det_path)
    out = tmp_path / "labels_out"

    result = runner.invoke(app, [
        "label", str(video), "--out", str(out), "--detections", str(det_path),
    ])

    assert result.exit_code == 0, result.output
    coco = json.loads((out / "annotations.json").read_text())
    frame_of_image = {im["id"]: _frame_of(im["file_name"]) for im in coco["images"]}
    boxes_per_frame = Counter(frame_of_image[a["image_id"]] for a in coco["annotations"])
    balls_per_frame = Counter(d.frame_idx for d in sim.detections)
    over = {f: n for f, n in boxes_per_frame.items() if n > balls_per_frame[f]}
    assert over == {}
    assert sum(boxes_per_frame.values()) >= 0.9 * len(sim.detections)


def test_label_motion_exports_one_calibrated_box_size(tmp_path):
    # Motion blobs vary in size with ball speed; the motion branch replaces
    # every box with one physics-calibrated size before export.
    from juggletrack.cli import app

    sim = simulate_cascade(n_throws=12, fps=30.0, seed=1)
    rng = np.random.default_rng(0)
    dets = [d.model_copy(update={"w": float(s), "h": float(s)})
            for d, s in zip(sim.detections, rng.uniform(0.02, 0.05, len(sim.detections)))]
    n_frames = max(d.frame_idx for d in dets) + 1
    video = tmp_path / "cascade.mp4"
    write_test_video(video, n_frames=n_frames, fps=30.0)
    det_path = tmp_path / "dets.jsonl"
    save_detections_jsonl(dets, det_path)
    out = tmp_path / "labels_out"

    result = runner.invoke(app, [
        "label", str(video), "--out", str(out), "--detections", str(det_path),
        "--detector", "motion",
    ])

    assert result.exit_code == 0, result.output
    coco = json.loads((out / "annotations.json").read_text())
    w_px, h_px = coco["images"][0]["width"], coco["images"][0]["height"]
    inside = [a["bbox"] for a in coco["annotations"]
              if a["bbox"][0] > 0 and a["bbox"][1] > 0
              and a["bbox"][0] + a["bbox"][2] < w_px and a["bbox"][1] + a["bbox"][3] < h_px]
    assert len(inside) > 100
    sizes = {(round(bw, 6), round(bh, 6)) for _, _, bw, bh in inside}
    assert len(sizes) == 1
    (bw, bh), = sizes
    assert 0.02 * w_px <= bw <= 0.05 * w_px and 0.02 * h_px <= bh <= 0.05 * h_px


def test_label_negatives_count_every_raw_junk_box(tmp_path):
    # A static scene feature that the detector double-fires on (a box and a
    # weaker echo 0.007 apart) on every frame, and a 3 s idle tail after the
    # juggling.
    # Hard-negative mining gates on raw junk boxes (min_junk=2), so the tail
    # frames are negatives; label-side box clustering must not reach them.
    from juggletrack.cli import app
    from juggletrack.types import Detection

    fps = 30.0
    sim = simulate_cascade(n_throws=12, fps=fps, seed=1)
    n_frames = max(d.frame_idx for d in sim.detections) + 1 + int(3 * fps)
    junk = [Detection(frame_idx=f, t=f / fps, x=x, y=0.1, w=0.03, h=0.03, confidence=c)
            for f in range(n_frames) for x, c in ((0.9, 0.3), (0.907, 0.2))]
    video = tmp_path / "cascade.mp4"
    write_test_video(video, n_frames=n_frames, fps=fps)
    det_path = tmp_path / "dets.jsonl"
    save_detections_jsonl(sorted(sim.detections + junk, key=lambda d: d.frame_idx), det_path)
    neg = tmp_path / "neg"

    result = runner.invoke(app, [
        "label", str(video), "--out", str(tmp_path / "labels_out"),
        "--detections", str(det_path), "--negatives", str(neg), "--neg-person-model", "none",
    ])

    assert result.exit_code == 0, result.output
    coco = json.loads((neg / "annotations.json").read_text())
    assert len(coco["images"]) == 40
    assert coco["annotations"] == []
