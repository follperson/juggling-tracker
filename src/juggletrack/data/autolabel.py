"""Arc-verified auto-labeling: physics-surviving detections become COCO labels.

Precision comes from the parabola gate (spec §5): a detection only becomes a
label if it lies on an extracted arc. Frames holding detections that did NOT
verify go to the review manifest for a human pass.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import cv2
import numpy as np

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


def calibrate_label_boxes(
    labels: list[Detection], arcs: list[Arc], *, slow_quantile: float = 0.25,
) -> list[Detection]:
    """Replace motion-derived box sizes with a physics-calibrated canonical size.

    Diagnosis: motion blobs systematically underestimate ball size, and the
    error varies with speed -- a fast-moving ball smears across the frame
    during its exposure window (motion blur / MOG2 tail), so its detected
    blob is narrower than the ball's true extent, while a ball near its
    arc's apex (low |velocity|) is essentially stationary for that frame and
    yields a blob close to the true size. A ball's physical size is ~constant
    within one video, so the apex-region boxes are the trustworthy sample.

    Method: assign each label to its arc (same acceptance rule as
    `select_autolabels`'s precision gate), compute each assigned label's
    speed from the arc model (`hypot(vy_at(t), bx)`), pool speeds globally
    across all arcs (not per-arc), and take the slowest `slow_quantile`
    fraction. The canonical box is the median w and median h of that slow
    subset. Every label's box (assigned or not) is replaced with this
    canonical size; centers (x, y) are never touched.

    Needs a quorum to trust the estimate: fewer than 8 assigned labels, or
    no arcs at all, returns `labels` unchanged.
    """
    if not arcs:
        return labels

    assignment = assign_detections(labels, arcs)
    arc_by_id = {a.id: a for a in arcs}

    # (speed, w, h) for every label that landed on an arc -- pooled globally
    # across arcs, per the spec, not bucketed per-arc.
    assigned: list[tuple[float, float, float]] = []
    for lb, arc_id in zip(labels, assignment):
        arc = arc_by_id.get(arc_id)
        if arc is None:
            continue
        speed = math.hypot(arc.vy_at(lb.t), arc.bx)
        assigned.append((speed, lb.w, lb.h))

    if len(assigned) < 8:
        return labels

    assigned.sort(key=lambda s: s[0])
    n_slow = max(1, round(slow_quantile * len(assigned)))
    slow = assigned[:n_slow]
    canon_w = statistics.median(w for _, w, _ in slow)
    canon_h = statistics.median(h for _, _, h in slow)

    # Uniform size applies to ALL labels, including any with arc_id == -1
    # (shouldn't exist post-select_autolabels, but handled here too).
    return [lb.model_copy(update={"w": canon_w, "h": canon_h}) for lb in labels]


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


def detect_person_boxes(
    video_path: str | Path,
    frame_indices: list[int],
    *,
    model: str = "yolo11n.pt",
    conf: float = 0.15,
) -> dict[int, list[tuple[float, float, float, float]]]:
    """Normalized-xyxy person boxes (COCO class 0) for the requested frames.

    conf=0.15 is deliberately permissive: a false-positive person only costs
    negative-mining yield (fail-safe), while a missed person can pass a frame
    whose real balls the ball detector also missed (the 183035587 frame-221
    field case sat at conf 0.246 -- under the usual 0.25).

    Feeds `export_hard_negatives`'s person-region veto. Runs the stock COCO
    model only on `frame_indices` (candidate negatives are a few dozen frames
    per video, so this stays cheap). Frames with no detected person are
    simply absent from the result.
    """
    from ultralytics import YOLO  # lazy: torch loads only when actually mining

    from juggletrack.video.reader import VideoReader

    wanted = set(frame_indices)
    if not wanted:
        return {}
    last_wanted = max(wanted)
    yolo = YOLO(model)
    out: dict[int, list[tuple[float, float, float, float]]] = {}
    with VideoReader(video_path) as reader:
        for idx, _t, frame in reader.frames():
            if idx > last_wanted:
                break
            if idx not in wanted:
                continue
            result = yolo.predict(frame, conf=conf, classes=[0], verbose=False)[0]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            out[idx] = [tuple(float(v) for v in row)
                        for row in boxes.xyxyn.cpu().numpy()]
    return out


def _trusted_static_clusters(
    idle_dets: list[Detection],
    all_unassigned: list[Detection],
    total_span: float,
    *,
    radius: float,
    min_frac: float,
    min_count: int,
) -> tuple[list[int], list[dict]]:
    """Greedy nearest-centroid clustering of idle-frame junk, plus trust.

    Returns (trusted flag per idle det, in input order; trusted clusters as
    manifest dicts). A cluster is trusted junk iff its idle membership has
    >= `min_count` detections AND the time span of ALL unassigned detections
    within `radius` of its centroid (any frame -- a wall picture's
    during-run firings are evidence too) covers >= `min_frac` of
    `total_span`.

    Cluster membership alone is NOT enough to trust a detection: greedy
    radius membership would let a real, slowly-moving ball inherit a trusted
    cluster's status just by drifting through its neighborhood
    (adversarial-review finding, demonstrated live). So each member must
    ALSO carry its own staticness evidence -- unassigned detections
    recurring within `radius / 2` of the detection's OWN position across
    >= `min_frac` of `total_span` (and >= `min_count` of them). A moving
    object's tight neighborhood only ever spans its crossing moment. The
    irreducible residual is a ball momentarily colocated with the junk's
    own footprint: geometrically indistinguishable, accepted and documented
    in the design spec.
    """
    # deterministic processing order regardless of caller's det ordering
    order = sorted(range(len(idle_dets)),
                   key=lambda i: (idle_dets[i].frame_idx, idle_dets[i].x, idle_dets[i].y))
    sums: list[list[float]] = []  # [sum_x, sum_y, n] per cluster
    cluster_of = [0] * len(idle_dets)
    for i in order:
        d = idle_dets[i]
        best, best_dist = -1, radius
        for ci, (sx, sy, n) in enumerate(sums):
            dist = math.hypot(sx / n - d.x, sy / n - d.y)
            if dist < best_dist:
                best, best_dist = ci, dist
        if best == -1:
            best = len(sums)
            sums.append([d.x, d.y, 1])
        else:
            sums[best][0] += d.x
            sums[best][1] += d.y
            sums[best][2] += 1
        cluster_of[i] = best

    trusted: list[dict] = []
    trusted_ids: set[int] = set()
    for ci, (sx, sy, n) in enumerate(sums):
        if n < min_count:
            continue
        cx, cy = sx / n, sy / n
        ts = [d.t for d in all_unassigned
              if math.hypot(d.x - cx, d.y - cy) < radius]
        if not ts:  # running centroid drifted past every evidence point
            continue
        span = max(ts) - min(ts)
        if total_span > 0 and span >= min_frac * total_span:
            trusted_ids.add(ci)
            trusted.append({"x": cx, "y": cy, "n": n,
                            "t_min": min(ts), "t_max": max(ts)})

    tight = radius / 2.0
    flags: list[bool] = []
    for d, ci in zip(idle_dets, cluster_of):
        if ci not in trusted_ids:
            flags.append(False)
            continue
        ts = [e.t for e in all_unassigned
              if math.hypot(e.x - d.x, e.y - d.y) < tight]
        flags.append(
            len(ts) >= min_count
            and total_span > 0
            and max(ts) - min(ts) >= min_frac * total_span
        )
    return flags, trusted


def export_hard_negatives(
    video_path: str | Path,
    dets: list[Detection],
    arcs: list[Arc],
    out_dir: str | Path,
    *,
    max_frames: int = 40,
    min_junk: int = 2,
    seed: int = 0,
    jpeg_quality: int = 90,
    pad: float = 1.0,
    persist_radius: float = 0.04,
    min_persist_frac: float = 0.5,
    min_persist_count: int = 8,
    floor_band_y: float = 0.85,
    person_boxes: dict[int, list[tuple[float, float, float, float]]] | None = None,
    person_model: str | None = None,
    person_dilate_frames: int = 5,
) -> dict:
    """Mine arc-rejected detections into zero-box negative training images.

    The arc gate (`assign_detections`) already tells the pipeline which
    detections are junk -- physics-implausible false positives (e.g. framed
    wall pictures the detector keeps firing on). That knowledge never made
    it back into training before this: negative (zero-box) images teach
    YOLO what NOT to detect, and `assemble_dataset` already passes
    zero-annotation images through untouched with empty label files, so all
    that was missing was generating them.

    But "arc-unassigned" is NOT "not a ball", and a zero-box negative
    containing a real ball actively teaches the model to miss balls -- worse
    than not training on the frame at all. The turn-4 postmortem (v4 trained
    on contaminated negatives and lost outdoor recall; not promoted) found
    every leak class in the field, and each gate below rejects one of them.
    A frame must pass ALL gates to export; every gate fails safe (rejects
    yield, never admits contamination):

    0. No arcs at all -> no negatives (`n_skipped_no_arcs`). Junk-ness is
       defined by arc rejection; a no-arc juggling video is exactly the
       maximum-contamination case (extraction failed everywhere).
    1. Any arc-assigned detection on the frame -> `n_skipped_ambiguous`
       (a zero-box negative would un-teach that verified ball).
    2. Frame time within `pad` s of any arc's [t_start, t_end] ->
       `n_skipped_active`. HELD balls are arc-unassigned by design (no
       parabola in a hand) and exist around flights, so near-activity
       frames can't be trusted at all.
    3. Any detection outside every trusted static cluster ->
       `n_skipped_transient`. Rejects MOVING unassigned objects: real
       flights the linker failed to stitch (turn-4: a ball crossing the
       juggler's face, frames 108/630, mistaken for "pupil FPs"). Trusted
       junk must be a persistent static scene feature, evidenced both at
       cluster level and in each detection's own tight neighborhood: see
       `_trusted_static_clusters` (`persist_radius`, `min_persist_frac`,
       `min_persist_count`). Residual: a ball momentarily colocated with a
       trusted junk spot is geometrically indistinguishable and passes.
    4. Any detection with y > `floor_band_y` -> `n_skipped_floor`. A ball
       RESTING on the floor is static and persistent -- geometrically
       indistinguishable from background junk (turn-4: clusters at y~0.96
       spanning 82-100% of the video) -- so the floor band is excluded
       wholesale. Genuine floor clutter is sacrificed; camera angles where
       the floor sits higher in frame are a documented residual risk.
    5. Any person detected on the frame OR within `person_dilate_frames`
       of it -> `n_skipped_person`. Rejects balls held in hands or
       crossing the body/face during idle stretches beyond `pad` (turn-4
       frame 87: two balls held at the hip 2.3 s before the first arc).
       PRESENCE is the veto, not junk-inside-box overlap: the final-audit
       field case (183035587 frame 221) had a person carrying balls that
       produced NO detection at all (motion blur), so detection-space
       geometry is blind to them -- the person box is the only visible
       evidence, wherever the frame's junk sits. And presence is dilated
       in time because the same field case showed the person detector
       itself missing the blurred entry frame (conf 0.246) while nailing
       its neighbors (0.57-0.86) -- a person cannot teleport, so a
       detection within +-5 frames vetoes too. Boxes come injected
       (`person_boxes`, normalized xyxy per frame) or from the stock COCO
       model (`person_model`, run on the dilated neighborhood of frames
       that survived gates 0-4 plus `min_junk`). Frames with no detected
       person anywhere nearby pass: an empty scene is the safest yield
       there is -- but a person-detector miss across the whole window
       still fails open for this gate (gates 3-4 still apply).

    `min_junk` stays what it always was: at least that many junk detections
    on the frame, applied after gate 4, uncounted.

    WARNING: the library defaults (`person_boxes=None, person_model=None`)
    run WITHOUT gate 5 -- held-ball frames like turn-4's frame 87 then pass
    gates 0-4 by construction. Any mining whose output feeds training must
    pass `person_model=` (the CLI default does) or inject `person_boxes`.
    """
    from juggletrack.video.reader import VideoReader

    assignment = assign_detections(dets, arcs)
    by_frame: dict[int, list[Detection]] = {}
    assigned_by_frame: dict[int, list[bool]] = {}
    frame_t: dict[int, float] = {}
    for d, arc_id in zip(dets, assignment):
        by_frame.setdefault(d.frame_idx, []).append(d)
        assigned_by_frame.setdefault(d.frame_idx, []).append(arc_id != -1)
        frame_t.setdefault(d.frame_idx, d.t)

    windows = [(a.t_start - pad, a.t_end + pad) for a in arcs]
    n_skipped_ambiguous = 0
    n_skipped_active = 0
    n_skipped_transient = 0
    n_skipped_floor = 0
    n_skipped_person = 0
    n_skipped_no_arcs = 0
    trusted_clusters: list[dict] = []
    candidates: dict[int, list[Detection]] = {}

    if not arcs:
        n_skipped_no_arcs = len(by_frame)
    else:
        idle_frames = []
        for frame_idx, flags in assigned_by_frame.items():
            if any(flags):
                n_skipped_ambiguous += 1
                continue
            if any(lo <= frame_t[frame_idx] <= hi for lo, hi in windows):
                n_skipped_active += 1
                continue
            idle_frames.append(frame_idx)

        idle_dets = [d for fi in idle_frames for d in by_frame[fi]]
        all_unassigned = [d for d, a in zip(dets, assignment) if a == -1]
        t_all = [d.t for d in dets]
        total_span = max(t_all) - min(t_all) if dets else 0.0
        trusted_flags, trusted_clusters = _trusted_static_clusters(
            idle_dets, all_unassigned, total_span,
            radius=persist_radius, min_frac=min_persist_frac,
            min_count=min_persist_count,
        )
        det_trusted = {id(d): ok for d, ok in zip(idle_dets, trusted_flags)}

        for frame_idx in idle_frames:
            frame_dets = by_frame[frame_idx]
            if not all(det_trusted[id(d)] for d in frame_dets):
                n_skipped_transient += 1
                continue
            if any(d.y > floor_band_y for d in frame_dets):
                n_skipped_floor += 1
                continue
            if len(frame_dets) >= min_junk:
                candidates[frame_idx] = frame_dets

        dilate = range(-person_dilate_frames, person_dilate_frames + 1)
        if person_model is not None and person_boxes is None:
            query = sorted({f + off for f in candidates for off in dilate
                            if f + off >= 0})
            person_boxes = detect_person_boxes(
                video_path, query, model=person_model,
            )
        if person_boxes:
            person_frames = {f for f, boxes in person_boxes.items() if boxes}
            kept: dict[int, list[Detection]] = {}
            for frame_idx, frame_dets in candidates.items():
                if any(frame_idx + off in person_frames for off in dilate):
                    n_skipped_person += 1
                    continue
                kept[frame_idx] = frame_dets
            candidates = kept

    n_candidate_frames = len(candidates)
    frame_list = sorted(candidates)
    rng = np.random.default_rng(seed)
    if len(frame_list) > max_frames:
        chosen = rng.choice(len(frame_list), size=max_frames, replace=False)
        chosen_frames = sorted(frame_list[i] for i in chosen)
    else:
        chosen_frames = frame_list
    chosen_set = set(chosen_frames)

    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)

    images: list[dict] = []
    manifest_frames: list[dict] = []
    with VideoReader(video_path) as reader:
        w_px, h_px = reader.info.width, reader.info.height
        for idx, _t, frame in reader.frames():
            if idx not in chosen_set:
                continue
            file_name = f"frame_{idx:06d}.jpg"
            cv2.imwrite(
                str(out / "images" / file_name), frame,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            )
            image_id = len(images) + 1
            images.append({"id": image_id, "file_name": file_name,
                           "width": w_px, "height": h_px})
            junk_dets = candidates[idx]
            manifest_frames.append({
                "frame_idx": idx,
                "n_junk": len(junk_dets),
                "positions": [[d.x, d.y] for d in junk_dets],
            })

    (out / "annotations.json").write_text(json.dumps({
        "images": images, "annotations": [],
        "categories": [BALL_CATEGORY],
    }, indent=2))
    (out / "negatives_manifest.json").write_text(json.dumps({
        "video": str(video_path),
        "frames": manifest_frames,
        "trusted_clusters": trusted_clusters,
    }, indent=2))

    return {
        "n_images": len(images),
        "n_candidate_frames": n_candidate_frames,
        "n_skipped_ambiguous": n_skipped_ambiguous,
        "n_skipped_active": n_skipped_active,
        "n_skipped_transient": n_skipped_transient,
        "n_skipped_floor": n_skipped_floor,
        "n_skipped_person": n_skipped_person,
        "n_skipped_no_arcs": n_skipped_no_arcs,
    }
