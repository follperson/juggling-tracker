"""Action-ROI cropping for training-data export.

Portrait 1080x1920 footage has ~15-19px balls; at YOLO's 640 input the long
side shrinks 3x, pushing balls below reliable detection/learning resolution.
This crops each dataset image to a square around the labeled action (the
union of its ball boxes) so balls keep native pixel resolution — the win is
the ball's size RELATIVE to the training image, not its absolute pixel size
(spec §4's ROI-cropping principle, applied to the training-data export
instead of runtime detection).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

MIN_EXPANSION_PX = 64.0
JITTER_FRACTION = 0.15


def _union_rect(boxes: list[list[float]]) -> tuple[float, float, float, float]:
    x_min = min(b[0] for b in boxes)
    y_min = min(b[1] for b in boxes)
    x_max = max(b[0] + b[2] for b in boxes)
    y_max = max(b[1] + b[3] for b in boxes)
    return x_min, y_min, x_max, y_max


def _axis_position(
    center: float, side: int, img_dim: int,
    union_lo: float, union_hi: float, jitter: float,
) -> float:
    """Clamp a jittered axis position so the square stays within the image
    and still fully contains the union rect on this axis."""
    lo = max(0.0, union_hi - side)
    hi = min(float(img_dim - side), union_lo)
    assert lo <= hi + 1e-6, "expanded square too small to contain box union"
    base = min(max(center - side / 2.0, lo), hi)
    return min(max(base + jitter, lo), hi)


def _positive_crop_rect(
    boxes: list[list[float]], img_w: int, img_h: int,
    crop: int, margin: float, rng: np.random.Generator,
) -> tuple[int, int, int]:
    x_min, y_min, x_max, y_max = _union_rect(boxes)
    union_w, union_h = x_max - x_min, y_max - y_min
    expansion = max(margin * max(union_w, union_h), MIN_EXPANSION_PX)
    ex_x_min, ex_x_max = x_min - expansion, x_max + expansion
    ex_y_min, ex_y_max = y_min - expansion, y_max + expansion

    side_f = max(crop, ex_x_max - ex_x_min, ex_y_max - ex_y_min)
    side_f = min(side_f, min(img_w, img_h))
    side = int(math.floor(side_f))

    cx = (ex_x_min + ex_x_max) / 2.0
    cy = (ex_y_min + ex_y_max) / 2.0
    jitter_max = JITTER_FRACTION * crop
    dx = rng.uniform(-jitter_max, jitter_max)
    dy = rng.uniform(-jitter_max, jitter_max)

    x = _axis_position(cx, side, img_w, x_min, x_max, dx)
    y = _axis_position(cy, side, img_h, y_min, y_max, dy)

    # final integer-rounding safety clamp (side/positions must stay in bounds)
    x_i = min(max(int(round(x)), 0), img_w - side)
    y_i = min(max(int(round(y)), 0), img_h - side)
    return x_i, y_i, side


def _negative_crop_rect(img_w: int, img_h: int, crop: int) -> tuple[int, int, int]:
    side = int(min(crop, min(img_w, img_h)))
    x = (img_w - side) // 2
    y = (img_h - side) // 2
    return x, y, side


def crop_coco_source(
    src_dir: str | Path,
    out_dir: str | Path,
    *,
    crop: int = 640,
    margin: float = 0.35,
    seed: int = 0,
) -> dict:
    src = Path(src_dir)
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)

    coco = json.loads((src / "annotations.json").read_text())
    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    rng = np.random.default_rng(seed)

    out_images: list[dict] = []
    out_annotations: list[dict] = []
    next_ann_id = 1
    ball_px_before: list[float] = []
    ball_px_after: list[float] = []
    ball_rel_before: list[float] = []
    ball_rel_after: list[float] = []
    crop_sides: list[int] = []

    for im in coco["images"]:
        img_w, img_h = im["width"], im["height"]
        img = cv2.imread(str(src / "images" / im["file_name"]))
        anns = anns_by_image.get(im["id"], [])

        if anns:
            boxes = [a["bbox"] for a in anns]
            x, y, side = _positive_crop_rect(boxes, img_w, img_h, crop, margin, rng)
        else:
            x, y, side = _negative_crop_rect(img_w, img_h, crop)

        cropped = img[y:y + side, x:x + side]
        assert cropped.shape[0] == side and cropped.shape[1] == side, (
            "crop fell outside image bounds"
        )
        cv2.imwrite(str(out / "images" / im["file_name"]), cropped)

        out_images.append({
            "id": im["id"], "file_name": im["file_name"],
            "width": side, "height": side,
        })
        crop_sides.append(side)

        for a in anns:
            bx, by, bw, bh = a["bbox"]
            nx, ny = bx - x, by - y
            assert nx >= -1e-6 and ny >= -1e-6, "box fell outside crop"
            assert nx + bw <= side + 1e-6 and ny + bh <= side + 1e-6, "box fell outside crop"
            out_annotations.append({
                "id": next_ann_id,
                "image_id": im["id"],
                "category_id": a["category_id"],
                "bbox": [nx, ny, bw, bh],
                "area": bw * bh,
                "iscrowd": a.get("iscrowd", 0),
            })
            next_ann_id += 1
            ball_px_before.append(bw)
            ball_px_after.append(bw)
            ball_rel_before.append(bw / img_w)
            ball_rel_after.append(bw / side)

    (out / "annotations.json").write_text(json.dumps({
        "images": out_images,
        "annotations": out_annotations,
        "categories": coco["categories"],
    }))

    def median(xs: list[float]) -> float:
        return float(np.median(xs)) if xs else 0.0

    return {
        "n_images": len(out_images),
        "n_boxes": len(out_annotations),
        "median_ball_px_before": median(ball_px_before),
        "median_ball_px_after": median(ball_px_after),
        "median_crop_px": median(crop_sides),
        "median_ball_rel_before": median(ball_rel_before),
        "median_ball_rel_after": median(ball_rel_after),
    }
