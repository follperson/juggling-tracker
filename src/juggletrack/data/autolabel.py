"""Arc-verified auto-labeling: physics-surviving detections become COCO labels.

Precision comes from the parabola gate (spec §5): a detection only becomes a
label if it lies on an extracted arc. Frames holding detections that did NOT
verify go to the review manifest for a human pass.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2

from juggletrack.arcs.extract import assign_detections
from juggletrack.types import Arc, Detection

BALL_CATEGORY = {"id": 1, "name": "ball"}


def select_autolabels(
    dets: list[Detection],
    arcs: list[Arc],
    *,
    resid_tol: float = 0.02,
    default_box: float = 0.04,
) -> tuple[list[Detection], list[int]]:
    assignment = assign_detections(dets, arcs, resid_tol=resid_tol)
    labels: list[Detection] = []
    review_frames: set[int] = set()
    for d, arc_id in zip(dets, assignment):
        if arc_id == -1:
            review_frames.add(d.frame_idx)
            continue
        labels.append(d.model_copy(update={
            "w": d.w if d.w > 0 else default_box,
            "h": d.h if d.h > 0 else default_box,
        }))
    return labels, sorted(review_frames)


def export_video_labels(
    video_path: str | Path,
    labels: list[Detection],
    out_dir: str | Path,
    *,
    review_frames: list[int] | None = None,
    jpeg_quality: int = 90,
) -> dict:
    from juggletrack.video.reader import VideoReader

    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)

    by_frame: dict[int, list[Detection]] = {}
    for lb in labels:
        by_frame.setdefault(lb.frame_idx, []).append(lb)

    images: list[dict] = []
    annotations: list[dict] = []
    with VideoReader(video_path) as reader:
        w_px, h_px = reader.info.width, reader.info.height
        for idx, _t, frame in reader.frames():
            frame_labels = by_frame.get(idx)
            if not frame_labels:
                continue
            file_name = f"frame_{idx:06d}.jpg"
            cv2.imwrite(
                str(out / "images" / file_name), frame,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            )
            image_id = len(images) + 1
            images.append({"id": image_id, "file_name": file_name,
                           "width": w_px, "height": h_px})
            for lb in frame_labels:
                bw, bh = lb.w * w_px, lb.h * h_px
                x_min = min(max(lb.x * w_px - bw / 2, 0.0), w_px - 1.0)
                y_min = min(max(lb.y * h_px - bh / 2, 0.0), h_px - 1.0)
                bw = min(bw, w_px - x_min)
                bh = min(bh, h_px - y_min)
                annotations.append({
                    "id": len(annotations) + 1, "image_id": image_id,
                    "category_id": BALL_CATEGORY["id"],
                    "bbox": [x_min, y_min, bw, bh],
                    "area": bw * bh, "iscrowd": 0,
                })

    (out / "annotations.json").write_text(json.dumps({
        "images": images, "annotations": annotations,
        "categories": [BALL_CATEGORY],
    }, indent=2))
    (out / "review_manifest.json").write_text(json.dumps({
        "video": str(video_path),
        "review_frames": sorted(set(review_frames or [])),
    }, indent=2))
    return {"n_images": len(images), "n_boxes": len(annotations),
            "n_review_frames": len(set(review_frames or []))}
