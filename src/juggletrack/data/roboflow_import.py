"""Import Roboflow COCO exports into our canonical source-dir shape.

Roboflow exports split subdirs (train|valid|test), each holding images plus
a `_annotations.coco.json` with arbitrary category ids/names (often a dummy
supercategory at id 0, and sometimes multi-class labels like hand/person
alongside the ball class). This adapter keeps only ball-class boxes, drops
images left with zero boxes, and re-numbers everything into the canonical
single-category COCO shape `assemble_dataset` expects.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

SPLIT_NAMES = ("train", "valid", "test")
BALL_CATEGORY = {"id": 1, "name": "ball"}


def import_roboflow_dataset(
    src_dir: str | Path,
    out_dir: str | Path,
    *,
    ball_names: tuple[str, ...] = ("ball", "juggling-ball", "juggling ball"),
) -> dict:
    src = Path(src_dir)
    out = Path(out_dir)
    ball_names_lower = {n.lower() for n in ball_names}

    splits = [s for s in SPLIT_NAMES if (src / s / "_annotations.coco.json").exists()]
    if not splits:
        raise ValueError(
            f"No Roboflow split subdirectories with _annotations.coco.json found under {src}"
        )

    all_category_names: set[str] = set()
    split_data = []
    for split in splits:
        coco = json.loads((src / split / "_annotations.coco.json").read_text())
        cat_id_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}
        all_category_names.update(cat_id_to_name.values())
        split_data.append((split, coco, cat_id_to_name))

    ball_cat_ids_by_split = []
    for split, coco, cat_id_to_name in split_data:
        ball_ids = {cid for cid, name in cat_id_to_name.items() if name.lower() in ball_names_lower}
        ball_cat_ids_by_split.append(ball_ids)

    if not any(ball_cat_ids_by_split):
        raise ValueError(
            f"No category name matched ball_names={ball_names!r}; "
            f"found categories: {sorted(all_category_names)}"
        )

    (out / "images").mkdir(parents=True, exist_ok=True)

    out_images: list[dict] = []
    out_annotations: list[dict] = []
    classes_dropped: Counter[str] = Counter()
    n_dropped_images = 0
    next_image_id = 1
    next_ann_id = 1

    for (split, coco, cat_id_to_name), ball_ids in zip(split_data, ball_cat_ids_by_split):
        anns_by_image: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        for im in coco["images"]:
            anns = anns_by_image.get(im["id"], [])
            kept = [a for a in anns if a["category_id"] in ball_ids]
            dropped = [a for a in anns if a["category_id"] not in ball_ids]
            for a in dropped:
                classes_dropped[cat_id_to_name[a["category_id"]]] += 1

            # Only drop images whose annotations were entirely filtered away
            # (e.g. hand/person-only frames). Images that never had any
            # annotation to begin with are legitimate negative examples and
            # are kept with zero boxes.
            if anns and not kept:
                n_dropped_images += 1
                continue

            orig_name = Path(im["file_name"]).name
            new_name = f"{split}_{orig_name}"
            shutil.copyfile(src / split / orig_name, out / "images" / new_name)

            out_images.append({
                "id": next_image_id,
                "file_name": new_name,
                "width": im["width"],
                "height": im["height"],
            })
            for a in kept:
                out_annotations.append({
                    "id": next_ann_id,
                    "image_id": next_image_id,
                    "category_id": BALL_CATEGORY["id"],
                    "bbox": a["bbox"],
                    "area": a["area"],
                    "iscrowd": a.get("iscrowd", 0),
                })
                next_ann_id += 1
            next_image_id += 1

    (out / "annotations.json").write_text(json.dumps({
        "images": out_images,
        "annotations": out_annotations,
        "categories": [BALL_CATEGORY],
    }))

    return {
        "n_images": len(out_images),
        "n_boxes": len(out_annotations),
        "n_dropped_images": n_dropped_images,
        "classes_dropped": dict(classes_dropped),
    }
