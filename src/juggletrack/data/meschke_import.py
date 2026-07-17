"""Import the Meschke Juggling Data Set into juggletrack's shapes.

Source corpus (mirrored under data/raw/meschke/): 159 CSVs, one row per video
FRAME, columns interleaved per-ball integer pixel coordinates (x1,y1,x2,y2,...).
The header row is inconsistent across files (sometimes a repeated
"x_position,y_position,..." label, sometimes a bare numeric index) so line 1 is
skipped unconditionally. Sentinel values -- an exact (0,0) pair or any negative
coordinate -- mean the ball wasn't visible in that frame and are dropped.

Two consumers of this same corpus:

- `load_meschke_trajectories`: the EVENT-EVAL ground-truth loader. Point labels
  (no box) become zero-size Detections, one trajectory per ball column-pair,
  ready to feed `analyze_detections` directly or compare against a detector's
  output.
- `export_meschke_source`: the TRAINING-DATA exporter. Buys real, human-labeled
  ball centers (no model-generated boxes to trust or distrust) at the cost of
  needing a plausible box size, since the corpus only ever labels a point.

Citation (required in any output manifest per the dataset's site):
"Citation: Stephen Meschke - Juggling Data Set - https://sites.google.com/view/jugglingdataset"
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from juggletrack.types import Detection

CITATION = (
    "Citation: Stephen Meschke - Juggling Data Set - "
    "https://sites.google.com/view/jugglingdataset"
)
BALL_CATEGORY = {"id": 1, "name": "ball"}

# Default ball box size, normalized to frame WIDTH (applied as a fixed pixel
# square, not stretched per-axis -- see export_meschke_source).
#
# Measured empirically, not derived per-video: extracted frame 1000 of the
# mirrored sample video data/raw/meschke/videos/ss3_id_016.MP4 (480x848),
# cropped a 120x120px window centered on that frame's labeled point (row 1000
# of ss3_id_016.csv, ball 1: (325, 654)), and inspected the crop -- a blue
# ball held at the catch, clearly resolved. Visual inspection showed the ball
# filling roughly half the 120px window; a color-segmentation check (HSV
# threshold + largest connected component, isolating the ball's blob from a
# stray same-hue speck elsewhere in the crop) measured a 30x34px bounding box,
# area-equivalent diameter ~31px. That confirms the corpus's balls are in the
# expected ~25-35px range at 480px width. 31/480 = 0.0646, rounded to 0.065.
DEFAULT_BOX = 0.065


def _read_csv_rows(csv_path: str | Path) -> list[list[int]]:
    """Read a Meschke CSV, unconditionally skipping the inconsistent header line."""
    path = Path(csv_path)
    with path.open(newline="") as f:
        all_rows = list(csv.reader(f))
    data_rows = all_rows[1:]
    return [[int(v) for v in row] for row in data_rows if row]


def _is_sentinel(px: int, py: int) -> bool:
    return (px == 0 and py == 0) or px < 0 or py < 0


def _trajectories_from_rows(
    rows: list[list[int]], *, fps: float, width: int, height: int,
) -> list[list[Detection]]:
    if not rows:
        return []
    n_cols = len(rows[0])
    if n_cols % 2 != 0:
        raise ValueError(f"expected an even number of x,y columns, got {n_cols}")
    n_balls = n_cols // 2

    trajectories: list[list[Detection]] = [[] for _ in range(n_balls)]
    for row_idx, row in enumerate(rows):
        for b in range(n_balls):
            px, py = row[2 * b], row[2 * b + 1]
            if _is_sentinel(px, py):
                continue
            trajectories[b].append(Detection(
                frame_idx=row_idx,
                t=row_idx / fps,
                x=px / width,
                y=py / height,
                w=0.0,
                h=0.0,
                confidence=1.0,
            ))
    return trajectories


def load_meschke_trajectories(
    csv_path: str | Path, *, fps: float, width: int, height: int,
) -> list[list[Detection]]:
    """Load one Detection trajectory per ball column-pair (event-eval ground truth).

    Skips sentinel points (exact (0,0) or negative coordinates = ball not
    visible that frame). frame_idx is the CSV row position (0-based); t is
    frame_idx / fps; x, y are normalized to [0, 1] by width/height. Boxes are
    zero (point labels only).
    """
    rows = _read_csv_rows(csv_path)
    return _trajectories_from_rows(rows, fps=fps, width=width, height=height)


def export_meschke_source(
    csv_path: str | Path,
    video_path: str | Path,
    out_dir: str | Path,
    *,
    frame_stride: int = 5,
    max_frames: int = 120,
    box: float | None = None,
    seed: int = 0,
) -> dict:
    """Export a training-data COCO source dir from one Meschke CSV+video pair.

    Selects up to `max_frames` frame indices spaced `frame_stride` apart (a
    seeded uniform subsample when more candidates exist than the cap), keeps
    only chosen frames with at least one visible ball, and writes each kept
    frame's jpg plus a canonical single-category ("ball", id 1) COCO
    annotations.json in one VideoReader pass. Box size defaults to
    `DEFAULT_BOX` (normalized to width) if `box` is None; the same pixel size
    is used for both width and height so the box stays square in pixel space
    regardless of the (often non-square, e.g. 480x848 portrait) frame aspect
    ratio.

    Raises ValueError if the CSV's row count and the video's frame count
    differ by more than 2 (something is misaligned -- wrong pairing, a
    truncated download, etc). A difference of 2 or fewer is tolerated by
    truncating to the shorter of the two.
    """
    import cv2

    from juggletrack.video.reader import VideoReader

    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)

    rows = _read_csv_rows(csv_path)

    with VideoReader(video_path) as reader:
        info = reader.info
        diff = abs(len(rows) - info.frame_count)
        if diff > 2:
            raise ValueError(
                f"{csv_path}: row count {len(rows)} and video frame count "
                f"{info.frame_count} ({video_path}) differ by {diff} (> 2 tolerance)"
            )
        n_frames_eff = min(len(rows), info.frame_count)
        rows = rows[:n_frames_eff]

        trajectories = _trajectories_from_rows(
            rows, fps=info.fps, width=info.width, height=info.height
        )
        by_frame: dict[int, list[tuple[float, float]]] = {}
        for traj in trajectories:
            for d in traj:
                by_frame.setdefault(d.frame_idx, []).append((d.x, d.y))

        candidates = list(range(0, n_frames_eff, frame_stride))
        if len(candidates) > max_frames:
            rng = np.random.default_rng(seed)
            pick = rng.choice(len(candidates), size=max_frames, replace=False)
            candidates = sorted(candidates[i] for i in pick)

        chosen_set = {c for c in candidates if by_frame.get(c)}
        last_needed = max(chosen_set) if chosen_set else -1

        box_norm = DEFAULT_BOX if box is None else box
        box_px = box_norm * info.width

        images: list[dict] = []
        annotations: list[dict] = []
        for idx, _t, frame in reader.frames():
            if idx > last_needed:
                break
            if idx not in chosen_set:
                continue
            file_name = f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(out / "images" / file_name), frame)
            image_id = len(images) + 1
            images.append({
                "id": image_id, "file_name": file_name,
                "width": info.width, "height": info.height,
            })
            for x_norm, y_norm in by_frame[idx]:
                px, py = x_norm * info.width, y_norm * info.height
                bw = bh = box_px
                x_min = min(max(px - bw / 2, 0.0), info.width - bw)
                y_min = min(max(py - bh / 2, 0.0), info.height - bh)
                bw = min(bw, info.width - x_min)
                bh = min(bh, info.height - y_min)
                annotations.append({
                    "id": len(annotations) + 1, "image_id": image_id,
                    "category_id": BALL_CATEGORY["id"],
                    "bbox": [x_min, y_min, bw, bh],
                    "area": bw * bh, "iscrowd": 0,
                })

        stats = {
            "n_images": len(images), "n_boxes": len(annotations),
            "fps": info.fps, "width": info.width, "height": info.height,
            "box_px": box_px,
        }

    (out / "annotations.json").write_text(json.dumps({
        "images": images, "annotations": annotations,
        "categories": [BALL_CATEGORY],
    }, indent=2))
    (out / "manifest.json").write_text(json.dumps({
        "citation": CITATION,
        "source_csv": str(csv_path),
        "source_video": str(video_path),
        **stats,
    }, indent=2))
    return stats
