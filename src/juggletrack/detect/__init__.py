"""Ball detection interfaces. Implementations must NOT import cv2/torch at module load."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from juggletrack.types import Detection


@runtime_checkable
class BallDetector(Protocol):
    def detect(self, frame: np.ndarray, frame_idx: int, t: float) -> list[Detection]:
        """Return ball detections for one BGR frame (normalized coordinates)."""
        ...
