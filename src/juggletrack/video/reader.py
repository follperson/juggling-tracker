"""cv2-backed video reading. The only module (with pipeline/) allowed to import cv2.

Phone videos carry rotation metadata; OpenCV >= 4.5 applies it automatically
(CAP_PROP_ORIENTATION_AUTO defaults on), so width/height are post-rotation.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel


class VideoInfo(BaseModel):
    path: str
    fps: float
    width: int
    height: int
    frame_count: int
    duration: float


class VideoReader:
    def __init__(self, path: str | Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"video not found: {path}")
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise ValueError(f"cv2 could not open video: {path}")
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
            raise ValueError(f"video has no readable frames/fps: {path}")
        self.info = VideoInfo(
            path=str(path),
            fps=fps,
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=frame_count,
            duration=frame_count / fps,
        )

    def frames(self) -> Iterator[tuple[int, float, np.ndarray]]:
        """Yield (frame_idx, timestamp_seconds, BGR frame) from the start."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        idx = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield idx, idx / self.info.fps, frame
            idx += 1

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
