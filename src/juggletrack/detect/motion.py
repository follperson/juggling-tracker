"""Motion-based ball detector via MOG2 background subtraction.

Cold-start bootstrap: on new-environment footage where the appearance-based
YOLO detector fails (dark balls against foliage/dark clothing), classical
background subtraction sees anything that MOVES regardless of appearance.
Precision is supplied downstream by the existing arc-verification gate
(extract_arcs / select_autolabels), not by this detector.

STATEFUL: `MotionDetector.detect` must be called once per frame, in strict
temporal order, for a single video -- the background model accumulates across
calls. `pipeline.offline.detect_video` already feeds frames this way; do not
call `detect` out of order or reuse one instance across unrelated videos.

cv2 is imported lazily inside `__init__` (not at module load), matching the
constraint documented in `juggletrack.detect`: importing `juggletrack.detect`
must not pull in cv2/torch.
"""
from __future__ import annotations

import numpy as np

from juggletrack.types import Detection


class MotionDetector:
    def __init__(
        self,
        *,
        history: int = 300,
        var_threshold: float = 16.0,
        min_area: float = 2e-5,
        max_area: float = 4e-3,
        warmup_frames: int = 10,
        max_blobs: int = 12,
    ):
        import cv2  # lazy: keep cv2 out of module load, per detect/ package constraint

        self._cv2 = cv2
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.min_area = min_area
        self.max_area = max_area
        self.warmup_frames = warmup_frames
        self.max_blobs = max_blobs
        self._n_calls = 0

    def detect(self, frame: np.ndarray, frame_idx: int, t: float) -> list[Detection]:
        cv2 = self._cv2
        fgmask = self._bg.apply(frame)
        self._n_calls += 1
        if self._n_calls <= self.warmup_frames:
            return []  # background model still learning

        _, mask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        h, w = frame.shape[:2]
        frame_area = float(w * h)
        n_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        out: list[Detection] = []
        for i in range(1, n_labels):  # label 0 is background
            area_norm = stats[i, cv2.CC_STAT_AREA] / frame_area
            if not (self.min_area <= area_norm <= self.max_area):
                continue
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[i]
            out.append(Detection(
                frame_idx=frame_idx, t=t,
                x=cx / w, y=cy / h,
                w=bw / w, h=bh / h,
                confidence=0.5,
            ))

        # Busy-frame guard: a MOVING camera makes MOG2 fire on nearly the
        # whole frame (hundreds of blobs on harvested footage), which floods
        # downstream arc extraction with junk detections -- both quality
        # (garbage arcs) and performance (extract_arcs' pairwise merge pass
        # is quadratic in arc count, and thousands of spurious short-lived
        # fragments from a busy scene blow that up to minutes). A frame this
        # cluttered can't be trusted as ball motion, so drop it entirely
        # rather than pass junk downstream.
        if len(out) > self.max_blobs:
            return []
        return out
