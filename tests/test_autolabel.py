import json

import pytest

from juggletrack.arcs.extract import extract_arcs
from juggletrack.data.autolabel import export_video_labels, select_autolabels
from juggletrack.sim import simulate_cascade
from juggletrack.types import Detection
from tests.helpers import write_test_video


@pytest.fixture()
def sim_with_junk():
    r = simulate_cascade(n_throws=8, fps=30.0, noise=0.002, seed=3)
    junk = [Detection(frame_idx=10, t=10 / 30.0, x=0.5, y=0.02),
            Detection(frame_idx=40, t=40 / 30.0, x=0.97, y=0.6)]
    return r, r.detections + junk


def test_select_autolabels_filters_unverified(sim_with_junk):
    r, dets = sim_with_junk
    arcs = extract_arcs(dets)
    labels, review = select_autolabels(dets, arcs)
    assert 0 < len(labels) < len(dets)
    assert all(lb.w == pytest.approx(0.04) and lb.h == pytest.approx(0.04) for lb in labels)
    # the junk frames must land in the review queue
    assert 10 in review and 40 in review


def test_export_writes_coco_and_frames(tmp_path, sim_with_junk):
    r, dets = sim_with_junk
    arcs = extract_arcs(dets)
    labels, review = select_autolabels(dets, arcs)
    n_frames = max(d.frame_idx for d in dets) + 1
    video = tmp_path / "v.mp4"
    write_test_video(video, n_frames=n_frames, fps=30.0, size=(320, 240))

    out = tmp_path / "labels_out"
    stats = export_video_labels(video, labels, out, review_frames=review)

    coco = json.loads((out / "annotations.json").read_text())
    assert coco["categories"] == [{"id": 1, "name": "ball"}]
    assert stats["n_images"] == len(coco["images"])
    assert stats["n_boxes"] == len(coco["annotations"]) == len(labels)
    labeled_frames = {d.frame_idx for d in labels}
    assert stats["n_images"] == len(labeled_frames)
    # every referenced image file exists with correct size metadata
    for im in coco["images"]:
        assert (out / "images" / im["file_name"]).exists()
        assert (im["width"], im["height"]) == (320, 240)
    # bbox sanity: pixel coords inside the image, area consistent
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        assert 0 <= x <= 320 and 0 <= y <= 240 and w > 0 and h > 0
        assert x + w <= 320 + 1e-6 and y + h <= 240 + 1e-6
        assert ann["area"] == pytest.approx(w * h)
        assert ann["iscrowd"] == 0 and ann["category_id"] == 1
    manifest = json.loads((out / "review_manifest.json").read_text())
    assert manifest["review_frames"] == sorted(set(review))


def test_export_bbox_matches_normalized_center(tmp_path):
    lb = Detection(frame_idx=0, t=0.0, x=0.5, y=0.5, w=0.1, h=0.2)
    video = tmp_path / "v.mp4"
    write_test_video(video, n_frames=1, fps=30.0, size=(320, 240))
    out = tmp_path / "one"
    export_video_labels(video, [lb], out)
    coco = json.loads((out / "annotations.json").read_text())
    assert coco["annotations"][0]["bbox"] == pytest.approx([144.0, 96.0, 32.0, 48.0])
