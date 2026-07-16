import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from juggletrack.data.dataset import assemble_dataset


def make_coco_source(root: Path, name: str, n_images: int, boxes_per_image: int = 1):
    """Minimal valid COCO source dir with black 64x48 jpgs."""
    import cv2

    src = root / name
    (src / "images").mkdir(parents=True)
    images, annotations = [], []
    for i in range(n_images):
        fn = f"frame_{i:06d}.jpg"
        cv2.imwrite(str(src / "images" / fn), np.zeros((48, 64, 3), dtype=np.uint8))
        images.append({"id": i + 1, "file_name": fn, "width": 64, "height": 48})
        for b in range(boxes_per_image):
            annotations.append({
                "id": len(annotations) + 1, "image_id": i + 1, "category_id": 1,
                "bbox": [16.0 + b, 12.0, 8.0, 6.0], "area": 48.0, "iscrowd": 0,
            })
    (src / "annotations.json").write_text(json.dumps({
        "images": images, "annotations": annotations,
        "categories": [{"id": 1, "name": "ball"}],
    }))
    return src


def test_assemble_splits_by_source(tmp_path):
    sources = [make_coco_source(tmp_path, f"vid{i}", n_images=3) for i in range(4)]
    out = tmp_path / "ds"
    stats = assemble_dataset(sources, out, val_fraction=0.25, seed=0)
    assert len(stats["val_sources"]) == 1
    assert len(stats["train_sources"]) == 3
    assert stats["n_train_images"] == 9 and stats["n_val_images"] == 3
    # no source appears in both splits
    assert not set(stats["train_sources"]) & set(stats["val_sources"])
    # every train image has a matching label file
    for img in (out / "images" / "train").iterdir():
        assert (out / "labels" / "train" / (img.stem + ".txt")).exists()


def test_yolo_label_contents(tmp_path):
    src = make_coco_source(tmp_path, "only", n_images=1)
    out = tmp_path / "ds"
    assemble_dataset([src], out, seed=0)
    txts = list((out / "labels" / "train").glob("*.txt"))
    assert len(txts) == 1
    parts = txts[0].read_text().split()
    # bbox [16, 12, 8, 6] in 64x48 -> cx=20/64, cy=15/48, w=8/64, h=6/48
    assert parts[0] == "0"
    assert [float(p) for p in parts[1:]] == pytest.approx([0.3125, 0.3125, 0.125, 0.125])


def test_data_yaml(tmp_path):
    sources = [make_coco_source(tmp_path, f"v{i}", 2) for i in range(2)]
    out = tmp_path / "ds"
    assemble_dataset(sources, out, val_fraction=0.5, seed=1)
    cfg = yaml.safe_load((out / "data.yaml").read_text())
    assert cfg["names"] == {0: "ball"}
    assert cfg["path"] == str(out.resolve())
    assert cfg["train"] == "images/train" and cfg["val"] == "images/val"


def test_single_source_degenerate_val(tmp_path):
    src = make_coco_source(tmp_path, "solo", 2)
    out = tmp_path / "ds"
    stats = assemble_dataset([src], out, seed=0)
    assert stats["val_sources"] == []
    cfg = yaml.safe_load((out / "data.yaml").read_text())
    assert cfg["val"] == "images/train"  # degenerate fallback, flagged in stats


def test_deterministic_split(tmp_path):
    sources = [make_coco_source(tmp_path, f"d{i}", 1) for i in range(5)]
    a = assemble_dataset(sources, tmp_path / "dsA", val_fraction=0.4, seed=7)
    b = assemble_dataset(sources, tmp_path / "dsB", val_fraction=0.4, seed=7)
    assert a["val_sources"] == b["val_sources"]
