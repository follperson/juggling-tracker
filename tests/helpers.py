"""Shared test utilities for video-facing tests."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def write_test_video(
    path: str | Path,
    n_frames: int = 60,
    fps: float = 30.0,
    size: tuple[int, int] = (320, 240),
) -> list[tuple[float, float]]:
    """Write an mp4 of a white circle orbiting on black.

    Returns the circle's normalized (x, y) center for each frame.
    """
    w, h = size
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter failed to open (mp4v codec missing?)")
    centers: list[tuple[float, float]] = []
    for i in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        phase = 2 * np.pi * i / n_frames
        cx = 0.5 + 0.3 * np.cos(phase)
        cy = 0.5 + 0.3 * np.sin(phase)
        cv2.circle(frame, (int(cx * w), int(cy * h)), 10, (255, 255, 255), -1)
        writer.write(frame)
        centers.append((cx, cy))
    writer.release()
    return centers
