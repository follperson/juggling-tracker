"""Deterministic detector double: replays pre-supplied detections by frame index."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from juggletrack.types import Detection


class FakeDetector:
    def __init__(self, detections: list[Detection]):
        self._by_frame: dict[int, list[Detection]] = defaultdict(list)
        for d in detections:
            self._by_frame[d.frame_idx].append(d)

    def detect(self, frame: np.ndarray, frame_idx: int, t: float) -> list[Detection]:
        return list(self._by_frame.get(frame_idx, []))
