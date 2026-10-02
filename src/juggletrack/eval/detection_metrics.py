"""Ball-center localization metrics independent of arc extraction and event counting."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from juggletrack.types import Detection


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
            # One invalid assignment must cost more than all valid distances combined.
            penalty = (min(len(pred), len(gt)) + 1) * tolerance_px
            rows, cols = linear_sum_assignment(
                np.where(distances <= tolerance_px, distances, penalty)
            )
            for pi, gi in zip(rows, cols):
                if distances[pi, gi] <= tolerance_px:
                    matched_p.add(int(pi))
                    matched_g.add(int(gi))
                    errors.append(float(distances[pi, gi]))
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
