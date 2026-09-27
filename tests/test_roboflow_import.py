import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from juggletrack.data.roboflow_import import import_roboflow_dataset

REAL_SRC = Path(__file__).resolve().parents[1] / "data/raw/roboflow/universe-juggling-balls"


def _write_image(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((48, 64, 3), dtype=np.uint8))


def make_roboflow_source(root: Path, name: str = "rf") -> Path:
    """Synthetic Roboflow-style export: two splits, a dummy id-0 supercategory,
    a "juggling-ball" class and a "hand" class, one image with only hand boxes.
    """
    src = root / name
    categories = [
        {"id": 0, "name": "objects", "supercategory": "none"},
        {"id": 1, "name": "juggling-ball", "supercategory": "objects"},
        {"id": 2, "name": "hand", "supercategory": "objects"},
    ]

    # train split: img_a has 2 ball boxes + 1 hand box (kept, hand dropped);
    #              img_b has only 2 hand boxes (dropped entirely)
    train_images = [
        {"id": 0, "file_name": "img_a.jpg", "width": 64, "height": 48},
        {"id": 1, "file_name": "img_b.jpg", "width": 64, "height": 48},
    ]
    train_annotations = [
        {"id": 0, "image_id": 0, "category_id": 1, "bbox": [10.0, 10.0, 5.0, 5.0], "area": 25.0, "iscrowd": 0},
        {"id": 1, "image_id": 0, "category_id": 1, "bbox": [20.0, 20.0, 6.0, 6.0], "area": 36.0, "iscrowd": 0},
        {"id": 2, "image_id": 0, "category_id": 2, "bbox": [30.0, 5.0, 8.0, 8.0], "area": 64.0, "iscrowd": 0},
        {"id": 3, "image_id": 1, "category_id": 2, "bbox": [1.0, 1.0, 4.0, 4.0], "area": 16.0, "iscrowd": 0},
        {"id": 4, "image_id": 1, "category_id": 2, "bbox": [2.0, 2.0, 4.0, 4.0], "area": 16.0, "iscrowd": 0},
    ]
    for im in train_images:
        _write_image(src / "train" / im["file_name"])
    (src / "train" / "_annotations.coco.json").write_text(json.dumps({
        "categories": categories, "images": train_images, "annotations": train_annotations,
    }))

    # valid split: img_c has 1 ball box
    valid_images = [{"id": 0, "file_name": "img_c.jpg", "width": 64, "height": 48}]
    valid_annotations = [
        {"id": 0, "image_id": 0, "category_id": 1, "bbox": [15.0, 15.0, 7.0, 7.0], "area": 49.0, "iscrowd": 0},
    ]
    _write_image(src / "valid" / "img_c.jpg")
    (src / "valid" / "_annotations.coco.json").write_text(json.dumps({
        "categories": categories, "images": valid_images, "annotations": valid_annotations,
    }))
    return src


def test_filters_non_ball_classes_and_reports_dropped(tmp_path):
    src = make_roboflow_source(tmp_path)
    out = tmp_path / "out"
    stats = import_roboflow_dataset(src, out)

    assert stats["n_images"] == 2  # img_a, img_c (img_b dropped: zero ball boxes)
    assert stats["n_boxes"] == 3  # 2 from img_a + 1 from img_c
    assert stats["n_dropped_images"] == 1
    assert stats["classes_dropped"] == {"hand": 3}


def test_images_copied_with_split_prefix(tmp_path):
    src = make_roboflow_source(tmp_path)
    out = tmp_path / "out"
    import_roboflow_dataset(src, out)

    assert (out / "images" / "train_img_a.jpg").exists()
    assert (out / "images" / "valid_img_c.jpg").exists()
    assert not (out / "images" / "train_img_b.jpg").exists()  # dropped, never copied


def test_canonical_annotations_shape(tmp_path):
    src = make_roboflow_source(tmp_path)
    out = tmp_path / "out"
    import_roboflow_dataset(src, out)

    coco = json.loads((out / "annotations.json").read_text())
    assert coco["categories"] == [{"id": 1, "name": "ball"}]

    # sequential ids from 1
    assert [im["id"] for im in coco["images"]] == list(range(1, len(coco["images"]) + 1))
    assert [a["id"] for a in coco["annotations"]] == list(range(1, len(coco["annotations"]) + 1))

    # all kept annotations point to category id 1 and reference a valid image id
    image_ids = {im["id"] for im in coco["images"]}
    for ann in coco["annotations"]:
        assert ann["category_id"] == 1
        assert ann["image_id"] in image_ids

    # width/height carried over from source, bbox/area/iscrowd carried over verbatim
    img_a = next(im for im in coco["images"] if im["file_name"] == "train_img_a.jpg")
    assert img_a["width"] == 64 and img_a["height"] == 48
    ann_boxes = sorted(
        tuple(a["bbox"]) for a in coco["annotations"] if a["image_id"] == img_a["id"]
    )
    assert ann_boxes == [(10.0, 10.0, 5.0, 5.0), (20.0, 20.0, 6.0, 6.0)]
    for a in coco["annotations"]:
        assert "area" in a and "iscrowd" in a


def test_unannotated_background_image_is_kept_not_dropped(tmp_path):
    """An image with zero annotations to begin with (a negative/background
    frame) is a legitimate training example and must be kept — distinct from
    an image whose annotations were all filtered out as non-ball classes.
    """
    src = tmp_path / "with_negative"
    categories = [
        {"id": 0, "name": "objects", "supercategory": "none"},
        {"id": 1, "name": "juggling-ball", "supercategory": "objects"},
    ]
    images = [
        {"id": 0, "file_name": "ball.jpg", "width": 64, "height": 48},
        {"id": 1, "file_name": "empty.jpg", "width": 64, "height": 48},
    ]
    annotations = [
        {"id": 0, "image_id": 0, "category_id": 1, "bbox": [1.0, 1.0, 2.0, 2.0], "area": 4.0, "iscrowd": 0},
    ]
    for im in images:
        _write_image(src / "train" / im["file_name"])
    (src / "train" / "_annotations.coco.json").write_text(json.dumps({
        "categories": categories, "images": images, "annotations": annotations,
    }))
    stats = import_roboflow_dataset(src, tmp_path / "out")
    assert stats["n_images"] == 2
    assert stats["n_boxes"] == 1
    assert stats["n_dropped_images"] == 0
    assert (tmp_path / "out" / "images" / "train_empty.jpg").exists()


def test_default_ball_names_matches_plain_ball(tmp_path):
    src = tmp_path / "plain"
    categories = [{"id": 1, "name": "ball"}]
    images = [{"id": 0, "file_name": "x.jpg", "width": 64, "height": 48}]
    annotations = [
        {"id": 0, "image_id": 0, "category_id": 1, "bbox": [1.0, 1.0, 2.0, 2.0], "area": 4.0, "iscrowd": 0},
    ]
    _write_image(src / "train" / "x.jpg")
    (src / "train" / "_annotations.coco.json").write_text(json.dumps({
        "categories": categories, "images": images, "annotations": annotations,
    }))
    stats = import_roboflow_dataset(src, tmp_path / "out")
    assert stats["n_images"] == 1
    assert stats["n_boxes"] == 1


def test_raises_when_no_split_subdir_found(tmp_path):
    src = tmp_path / "empty"
    src.mkdir()
    (src / "README.txt").write_text("nothing here")
    with pytest.raises(ValueError):
        import_roboflow_dataset(src, tmp_path / "out")


def test_raises_when_no_ball_category_matches(tmp_path):
    src = tmp_path / "no_balls"
    categories = [
        {"id": 0, "name": "objects", "supercategory": "none"},
        {"id": 1, "name": "hand", "supercategory": "objects"},
        {"id": 2, "name": "person", "supercategory": "objects"},
    ]
    images = [{"id": 0, "file_name": "x.jpg", "width": 64, "height": 48}]
    annotations = [
        {"id": 0, "image_id": 0, "category_id": 1, "bbox": [1.0, 1.0, 2.0, 2.0], "area": 4.0, "iscrowd": 0},
    ]
    _write_image(src / "train" / "x.jpg")
    (src / "train" / "_annotations.coco.json").write_text(json.dumps({
        "categories": categories, "images": images, "annotations": annotations,
    }))
    with pytest.raises(ValueError, match="hand"):
        import_roboflow_dataset(src, tmp_path / "out")


def test_real_universe_juggling_balls_dataset(tmp_path):
    if not REAL_SRC.exists():
        pytest.skip("real Roboflow universe-juggling-balls dataset not available on this machine")
    stats = import_roboflow_dataset(REAL_SRC, tmp_path / "out")
    assert stats["n_boxes"] == 1042
    assert stats["n_images"] == 403
    coco = json.loads((tmp_path / "out" / "annotations.json").read_text())
    for im in coco["images"]:
        assert (tmp_path / "out" / "images" / im["file_name"]).exists()
