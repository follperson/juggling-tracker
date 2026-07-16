"""Assemble ultralytics-ready datasets from COCO source dirs.

COCO stays canonical (spec: framework-neutral); this derives the YOLO layout.
Split is BY SOURCE VIDEO — never by frame (spec §6).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import yaml


def assemble_dataset(
    source_dirs: list[Path],
    out_dir: Path,
    *,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> dict:
    sources = [Path(s) for s in source_dirs]
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(sources)))
    n_val = max(1, round(val_fraction * len(sources))) if len(sources) >= 2 else 0
    val_idx = set(order[:n_val])

    out = Path(out_dir)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"train_sources": [], "val_sources": [],
             "n_train_images": 0, "n_val_images": 0,
             "n_train_boxes": 0, "n_val_boxes": 0}

    for i, src in enumerate(sources):
        split = "val" if i in val_idx else "train"
        stats[f"{split}_sources"].append(src.name)
        coco = json.loads((src / "annotations.json").read_text())
        anns_by_image: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)
        for im in coco["images"]:
            stem = f"{src.name}_{Path(im['file_name']).stem}"
            shutil.copyfile(src / "images" / im["file_name"],
                            out / "images" / split / f"{stem}.jpg")
            lines = []
            for ann in anns_by_image.get(im["id"], []):
                x, y, w, h = ann["bbox"]
                cx, cy = (x + w / 2) / im["width"], (y + h / 2) / im["height"]
                lines.append(
                    f"0 {cx:.6f} {cy:.6f} {w / im['width']:.6f} {h / im['height']:.6f}"
                )
            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            stats[f"n_{split}_images"] += 1
            stats[f"n_{split}_boxes"] += len(lines)

    (out / "data.yaml").write_text(yaml.safe_dump({
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val" if stats["n_val_images"] else "images/train",
        "names": {0: "ball"},
    }, sort_keys=False))
    return stats
