import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from juggletrack.data.crop import crop_coco_source


def make_source(root: Path, name: str, image_specs: list[tuple[int, int, list[tuple[float, float, float, float]]]]):
    """Canonical COCO source dir with black jpgs.

    image_specs: list of (width, height, boxes) where boxes is a list of
    (x, y, w, h) in COCO bbox convention. Empty boxes -> negative sample image.
    """
    src = root / name
    (src / "images").mkdir(parents=True)
    images, annotations = [], []
    ann_id = 1
    for i, (w, h, boxes) in enumerate(image_specs):
        fn = f"frame_{i:06d}.jpg"
        cv2.imwrite(str(src / "images" / fn), np.zeros((h, w, 3), dtype=np.uint8))
        images.append({"id": i + 1, "file_name": fn, "width": w, "height": h})
        for (x, y, bw, bh) in boxes:
            annotations.append({
                "id": ann_id, "image_id": i + 1, "category_id": 1,
                "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0,
            })
            ann_id += 1
    (src / "annotations.json").write_text(json.dumps({
        "images": images, "annotations": annotations,
        "categories": [{"id": 1, "name": "ball"}],
    }))
    return src


def load_out(out: Path):
    return json.loads((out / "annotations.json").read_text())


def test_crop_contains_box_with_size_preserved(tmp_path):
    # off-center box near the bottom of a 640x1136 portrait-ish image
    src = make_source(tmp_path, "vid", [(640, 1136, [(450.0, 1000.0, 17.0, 20.0)])])
    out = tmp_path / "out"
    crop_coco_source(src, out, crop=640, margin=0.35, seed=0)
    coco = load_out(out)
    assert len(coco["images"]) == 1
    assert len(coco["annotations"]) == 1
    im = coco["images"][0]
    ann = coco["annotations"][0]

    # crop is square and within the original image's bounds
    assert im["width"] == im["height"]
    assert im["width"] <= 640 and im["height"] <= 1136

    # box size unchanged by the crop (translation only, no rescale)
    bx, by, bw, bh = ann["bbox"]
    assert bw == pytest.approx(17.0)
    assert bh == pytest.approx(20.0)

    # box fully contained within the crop
    assert bx >= 0 and by >= 0
    assert bx + bw <= im["width"]
    assert by + bh <= im["height"]

    # the on-disk crop matches the declared dimensions
    crop_img = cv2.imread(str(out / "images" / im["file_name"]))
    assert crop_img.shape[0] == im["height"]
    assert crop_img.shape[1] == im["width"]


def test_crop_offset_within_original_bounds(tmp_path):
    src = make_source(tmp_path, "vid", [(900, 1400, [(120.0, 1300.0, 18.0, 22.0)])])
    out = tmp_path / "out"
    crop_coco_source(src, out, crop=640, margin=0.35, seed=1)
    coco = load_out(out)
    im = coco["images"][0]
    ann = coco["annotations"][0]
    # infer the crop's offset in the original image from box translation
    orig_x, orig_y = 120.0, 1300.0
    offset_x = orig_x - ann["bbox"][0]
    offset_y = orig_y - ann["bbox"][1]
    assert 0 <= offset_x <= 900 - im["width"]
    assert 0 <= offset_y <= 1400 - im["height"]


def test_large_box_spread_grows_square_beyond_default_crop(tmp_path):
    # two boxes far apart force a union bigger than the default 640 crop
    src = make_source(tmp_path, "vid", [
        (1200, 1600, [(100.0, 100.0, 20.0, 20.0), (900.0, 900.0, 20.0, 20.0)]),
    ])
    out = tmp_path / "out"
    crop_coco_source(src, out, crop=640, margin=0.1, seed=0)
    coco = load_out(out)
    im = coco["images"][0]
    assert im["width"] == im["height"]
    assert im["width"] > 640
    for ann in coco["annotations"]:
        bx, by, bw, bh = ann["bbox"]
        assert bx >= 0 and by >= 0
        assert bx + bw <= im["width"]
        assert by + bh <= im["height"]


def test_negative_image_passes_through_as_center_crop(tmp_path):
    src = make_source(tmp_path, "vid", [(1080, 1920, [])])
    out = tmp_path / "out"
    stats = crop_coco_source(src, out, crop=640, seed=0)
    coco = load_out(out)
    assert len(coco["images"]) == 1
    assert coco["annotations"] == []
    im = coco["images"][0]
    assert im["width"] == 640 and im["height"] == 640
    crop_img = cv2.imread(str(out / "images" / im["file_name"]))
    assert crop_img.shape[:2] == (640, 640)
    assert stats["n_images"] == 1
    assert stats["n_boxes"] == 0


def test_stats_fields_sane(tmp_path):
    src = make_source(tmp_path, "vid", [
        (1080, 1920, [(500.0, 900.0, 17.0, 24.0)]),
        (1080, 1920, [(200.0, 300.0, 19.0, 22.0)]),
        (1080, 1920, []),
    ])
    out = tmp_path / "out"
    stats = crop_coco_source(src, out, crop=640, seed=0)
    assert stats["n_images"] == 3
    assert stats["n_boxes"] == 2
    # no rescale: absolute ball pixel size is unchanged by cropping
    assert stats["median_ball_px_before"] == pytest.approx(stats["median_ball_px_after"])
    assert stats["median_crop_px"] > 0
    # relative size grows because the crop is smaller than the full frame
    assert stats["median_ball_rel_after"] > stats["median_ball_rel_before"]


def test_union_taller_than_image_width_drops_outliers_without_crashing(tmp_path):
    # Portrait frame where ball boxes are spread over more vertical range than
    # the image is wide (plausible for a 3-ball cascade shot close-up): a
    # square crop can't contain the full union while staying in image bounds.
    # This should degrade gracefully (drop the outlier boxes that can't fit,
    # tracked in stats) rather than crash.
    boxes = [
        (550.0, 240.0, 20.0, 20.0),
        (560.0, 700.0, 20.0, 20.0),
        (555.0, 900.0, 20.0, 20.0),
        (565.0, 1470.0, 20.0, 20.0),
    ]
    src = make_source(tmp_path, "vid", [(1080, 1920, boxes)])
    out = tmp_path / "out"
    stats = crop_coco_source(src, out, crop=640, margin=0.35, seed=0)
    coco = load_out(out)
    im = coco["images"][0]
    assert im["width"] == im["height"]
    # some boxes near the clustered middle are kept
    assert len(coco["annotations"]) >= 2
    assert len(coco["annotations"]) < len(boxes)
    for ann in coco["annotations"]:
        bx, by, bw, bh = ann["bbox"]
        assert bx >= 0 and by >= 0
        assert bx + bw <= im["width"]
        assert by + bh <= im["height"]
    assert stats["n_boxes_dropped_oob"] == len(boxes) - len(coco["annotations"])
    assert stats["n_boxes_dropped_oob"] > 0


def test_deterministic_output(tmp_path):
    src = make_source(tmp_path, "vid", [
        (1080, 1920, [(500.0, 900.0, 17.0, 24.0)]),
        (1080, 1920, [(200.0, 300.0, 19.0, 22.0)]),
    ])
    out_a = tmp_path / "outA"
    out_b = tmp_path / "outB"
    stats_a = crop_coco_source(src, out_a, crop=640, seed=42)
    stats_b = crop_coco_source(src, out_b, crop=640, seed=42)
    assert stats_a == stats_b
    coco_a = load_out(out_a)
    coco_b = load_out(out_b)
    assert coco_a == coco_b
    for im in coco_a["images"]:
        img_a = cv2.imread(str(out_a / "images" / im["file_name"]))
        img_b = cv2.imread(str(out_b / "images" / im["file_name"]))
        assert np.array_equal(img_a, img_b)
