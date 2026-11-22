"""Assemble ultralytics-ready datasets from COCO source dirs.

COCO stays canonical (spec: framework-neutral); this derives the YOLO layout.
Split is BY RECORDING — never by frame (spec §6).
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import yaml


def assemble_dataset(
    source_dirs: list[Path],
    out_dir: Path,
    *,
    val_fraction: float = 0.2,
    seed: int = 0,
    recordings: dict[str, str] | None = None,
) -> dict:
    """Write a dataset into an empty directory, splitting by recording.

    `recordings` maps a source directory's name to its recording id, so the
    labels, negatives and crops of one recording land on the same side. A
    source it does not name is its own recording. Validation takes the
    `n_val` recordings that rank first by a hash of the seed and the
    recording id, so reordering sources moves nothing and adding one can
    displace at most the recording at the edge of the validation set.

    Reusing a populated output can leave old images in the opposite split.
    Refuse that operation before writing; use a new versioned directory.
    A single source remains useful for training smoke tests, but its
    validation path aliases training and is not an accuracy measurement.
    """
    sources = [Path(s) for s in source_dirs]
    if not sources:
        raise ValueError("provide at least one source directory")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1 (exclusive)")
    if (len({s.name for s in sources}) != len(sources)
            or len({s.resolve() for s in sources}) != len(sources)):
        raise ValueError("source directories must have unique names and unique resolved paths")

    out = Path(out_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"use an empty output directory: {out} already contains data")

    recording_of = {s.name: (recordings or {}).get(s.name, s.name) for s in sources}
    rec_ids = set(recording_of.values())
    n_val = min(len(rec_ids) - 1, max(1, round(val_fraction * len(rec_ids))))
    ranked = sorted(rec_ids, key=lambda r: hashlib.sha256(f"{seed}:{r}".encode()).hexdigest())
    val_recs = set(ranked[:n_val])

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"train_sources": [], "val_sources": [],
             "n_train_images": 0, "n_val_images": 0,
             "n_train_boxes": 0, "n_val_boxes": 0}

    for src in sources:
        split = "val" if recording_of[src.name] in val_recs else "train"
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
