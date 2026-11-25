"""Sparse labels for ball centers, independent of detector or arc output."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from juggletrack.eval.detection_metrics import match_centers
from juggletrack.types import Detection

PTS_TOLERANCE_S = 0.001


def same_frame_time(first: float, second: float) -> bool:
    # The tiny margin keeps an exact millisecond at the floating-point boundary valid.
    return abs(first - second) <= PTS_TOLERANCE_S + 1e-12


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ReviewedCenter(_StrictModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    state: Literal["airborne", "held", "ambiguous"]


class ReviewedFrame(_StrictModel):
    frame_idx: int = Field(ge=0)
    t: float = Field(ge=0)
    # Required: an empty list explicitly records a reviewed empty frame.
    centers: list[ReviewedCenter]


class ReviewedCenters(_StrictModel):
    """Version 1 JSON contract. Only the listed frames have been reviewed."""

    schema_version: Literal[1]
    review_method: Literal["human", "model_assisted", "synthetic"]
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frames: list[ReviewedFrame] = Field(min_length=1)

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value):
        if type(value) is not int:
            raise ValueError("schema_version must be integer 1")
        return value

    @model_validator(mode="after")
    def ordered_frames(self) -> "ReviewedCenters":
        for previous, current in zip(self.frames, self.frames[1:]):
            if current.frame_idx <= previous.frame_idx:
                raise ValueError("reviewed frame indices must be unique and increasing")
            if current.t <= previous.t:
                raise ValueError("reviewed frame PTS must be strictly increasing")
        return self


def _state_counts(centers, matched):
    counts = {}
    for state in ("airborne", "held", "ambiguous"):
        n = sum(c.state == state for c in centers)
        found = sum(centers[i].state == state for i in matched)
        counts[state] = {"n": n, "matched": found,
                         "recall": found / n if n and state != "ambiguous" else None}
    return counts


def evaluate_reviewed_centers(
    predictions: list[Detection], labels: ReviewedCenters, *,
    tolerance_px: float, min_conf: float = 0.0,
) -> dict:
    """Score definite centers first, then let ambiguous centers absorb residuals.

    Unlisted frames are unreviewed, never empty ground truth. Every input row
    is validated before confidence filtering, including reviewed-frame PTS.
    Ambiguous coverage is reported separately from judged precision/recall.
    """
    if (isinstance(tolerance_px, bool) or not math.isfinite(tolerance_px)
            or tolerance_px <= 0):
        raise ValueError("tolerance_px must be finite and positive")
    if isinstance(min_conf, bool) or not math.isfinite(min_conf) or not 0 <= min_conf <= 1:
        raise ValueError("min_conf must be finite and between 0 and 1")
    reviewed = {frame.frame_idx: frame for frame in labels.frames}
    by_frame = defaultdict(list)
    for pred in predictions:
        if (type(pred.frame_idx) is not int or pred.frame_idx < 0
                or not math.isfinite(pred.t) or pred.t < 0
                or not math.isfinite(pred.x) or not 0 <= pred.x <= 1
                or not math.isfinite(pred.y) or not 0 <= pred.y <= 1
                or not math.isfinite(pred.confidence) or not 0 <= pred.confidence <= 1):
            raise ValueError(f"invalid prediction at frame {pred.frame_idx}")
        frame = reviewed.get(pred.frame_idx)
        if frame is not None:
            if not same_frame_time(pred.t, frame.t):
                raise ValueError(f"prediction PTS differs by more than 1 ms at frame {pred.frame_idx}")
            if pred.confidence >= min_conf:
                by_frame[pred.frame_idx].append(pred)

    frames = []
    all_centers, all_matched = [], set()
    errors = []
    for frame in labels.frames:
        pred = [(p.x * labels.width, p.y * labels.height) for p in by_frame[frame.frame_idx]]
        definite = [i for i, c in enumerate(frame.centers) if c.state != "ambiguous"]
        ambiguous = [i for i, c in enumerate(frame.centers) if c.state == "ambiguous"]

        def coords(indices):
            return [(frame.centers[i].x * labels.width,
                     frame.centers[i].y * labels.height) for i in indices]

        matches = match_centers(pred, coords(definite), tolerance_px)
        matched_pred = {pi for pi, _, _ in matches}
        matched_targets = {definite[gi] for _, gi, _ in matches}
        errors.extend(distance for _, _, distance in matches)
        residual = [i for i in range(len(pred)) if i not in matched_pred]
        absorbed = match_centers([pred[i] for i in residual], coords(ambiguous), tolerance_px)
        matched_targets.update(ambiguous[gi] for _, gi, _ in absorbed)
        tp, ignored = len(matches), len(absorbed)
        fp, fn = len(pred) - tp - ignored, len(definite) - tp
        frames.append({
            "frame_idx": frame.frame_idx, "t": frame.t, "n_predictions": len(pred),
            "tp": tp, "fp": fp, "fn": fn, "n_ignored_predictions": ignored,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "states": _state_counts(frame.centers, matched_targets),
        })
        all_matched.update(len(all_centers) + i for i in matched_targets)
        all_centers.extend(frame.centers)
    tp, fp, fn = (sum(frame[key] for frame in frames) for key in ("tp", "fp", "fn"))
    return {
        "metric": "reviewed_ball_center_distance", "schema_version": 1,
        "source_sha256": labels.source_sha256, "review_method": labels.review_method,
        "width": labels.width, "height": labels.height,
        "tolerance_px": tolerance_px, "min_conf": min_conf,
        "n_reviewed_frames": len(frames), "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "states": _state_counts(all_centers, all_matched),
        "n_ignored_predictions": sum(frame["n_ignored_predictions"] for frame in frames),
        "mean_center_error_px": sum(errors) / len(errors) if errors else None,
        "frames": frames,
    }
