"""Ball-center localization metrics independent of arc extraction and event counting."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from juggletrack.types import Detection


def match_centers(
    predictions: list[tuple[float, float]], targets: list[tuple[float, float]],
    tolerance_px: float,
) -> list[tuple[int, int, float]]:
    """Return deterministic gated (prediction, target, distance) assignments.

    Inputs are pixel coordinates in stable input order. Maximize the number
    of gated matches before minimizing their total distance. SciPy's stable
    assignment on this fixed matrix resolves equal-cost choices consistently.
    """
    if not predictions or not targets:
        return []
    distances = np.linalg.norm(
        np.asarray(predictions)[:, None] - np.asarray(targets)[None, :], axis=2,
    )
    return _match_distances(distances, tolerance_px)


def _match_distances(distances: np.ndarray, tolerance_px: float) -> list[tuple[int, int, float]]:
    """Assign an already calculated matrix, retaining dense scorer costs."""
    valid = distances <= tolerance_px
    penalty = (min(distances.shape) + 1) * tolerance_px
    if math.isfinite(penalty):
        costs = np.where(valid, distances, penalty)
    else:
        # Scale only when the normal penalty would overflow.
        costs = np.full(distances.shape, min(distances.shape) + 1.0)
        costs[valid] = distances[valid] / tolerance_px
    rows, cols = linear_sum_assignment(costs)
    return [(int(pi), int(gi), float(distances[pi, gi]))
            for pi, gi in zip(rows, cols) if valid[pi, gi]]


def evaluate_ball_centers(
    predictions: list[Detection], targets: list[Detection], *,
    width: int, height: int, frame_count: int, tolerance_px: float = 10.0, stride: int = 1,
    min_conf: float = 0.0,
) -> dict:
    """Match visible labeled centers one-to-one within a pixel-distance gate.

    Maximize valid matches first, then minimize localization distance. Extra
    predictions near a labeled center are duplicate candidates; other extra
    predictions are background/localization errors. These are center metrics,
    not box IoU or mAP. Empty annotated frames contribute false positives.
    """
    if width <= 0 or height <= 0 or frame_count <= 0 or stride < 1:
        raise ValueError("width, height, frame_count and stride must be positive")
    if not math.isfinite(tolerance_px) or tolerance_px <= 0:
        raise ValueError("tolerance_px must be finite and positive")
    if not 0 <= min_conf <= 1:
        raise ValueError("min_conf must be between 0 and 1")
    if any(not 0 <= d.confidence <= 1 for d in predictions):
        raise ValueError("prediction confidences must be finite and between 0 and 1")

    def group(dets):
        by_frame = defaultdict(list)
        for d in dets:
            if (not 0 <= d.frame_idx < frame_count
                    or not math.isfinite(d.x) or not math.isfinite(d.y)):
                raise ValueError(f"invalid center at frame {d.frame_idx}")
            if d.frame_idx % stride == 0:
                by_frame[d.frame_idx].append((d.x * width, d.y * height))
        return by_frame

    pred_frames = group([d for d in predictions if d.confidence >= min_conf])
    target_frames = group(targets)
    tp = fp = fn = duplicate_fp = 0
    errors: list[float] = []
    error_frames: list[dict] = []
    # Empty-on-both frames contribute no errors; include them in the frame denominator.
    for frame_idx in sorted(pred_frames.keys() | target_frames.keys()):
        pred, gt = pred_frames[frame_idx], target_frames[frame_idx]
        matched_p: set[int] = set()
        matched_g: set[int] = set()
        duplicates = 0
        if pred and gt:
            distances = np.linalg.norm(np.asarray(pred)[:, None] - np.asarray(gt)[None, :], axis=2)
            for pi, gi, distance in _match_distances(distances, tolerance_px):
                matched_p.add(pi)
                matched_g.add(gi)
                errors.append(distance)
            duplicates = sum(
                bool(np.any(distances[pi] <= tolerance_px))
                for pi in range(len(pred)) if pi not in matched_p
            )
        frame_fp, frame_fn = len(pred) - len(matched_p), len(gt) - len(matched_g)
        tp += len(matched_p)
        fp += frame_fp
        fn += frame_fn
        duplicate_fp += duplicates
        if frame_fp or frame_fn:
            error_frames.append({"frame_idx": frame_idx, "fp": frame_fp,
                                 "fn": frame_fn, "duplicate_fp": duplicates})

    return {
        "metric": "ball_center_distance", "tolerance_px": tolerance_px, "min_conf": min_conf,
        "width": width, "height": height, "frame_count": frame_count, "stride": stride,
        "n_sampled_frames": math.ceil(frame_count / stride),
        "tp": tp, "fp": fp, "fn": fn,
        "duplicate_fp": duplicate_fp, "background_fp": fp - duplicate_fp,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "mean_center_error_px": float(np.mean(errors)) if errors else None,
        "error_frames": error_frames,
    }
