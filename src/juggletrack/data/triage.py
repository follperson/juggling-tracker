"""Camera-motion triage: classify harvested clips as static- or moving-camera.

Motivation: `detect.motion.MotionDetector` assumes a static camera -- MOG2
background subtraction only isolates a moving ball if the *background* holds
still. Harvested corpora (Kinetics, commons, external scrapes) frequently
have a MOVING camera instead, which makes MOG2 fire on nearly the whole
frame every frame (hundreds of blobs), and the busy-frame guard in
`motion.py` then blanks most of the clip's detections outright. Triage lets
the bulk-labeling pipeline sort clips BEFORE spending a motion-detection
pass on them, instead of discovering a moving-camera clip only after it
produces near-empty output.

Method: sample a handful of frame pairs spaced through the video, ~0.5s
apart each. For each pair, downscale both frames to a fixed small width and
run phase correlation -- this recovers the dominant (background) global
translational shift between the two frames, which tracks camera motion and
is largely insensitive to a small foreground subject (e.g. a juggler)
moving through an otherwise static scene, since that subject occupies only a
minority of the frame's spectral energy. The per-pair shift is normalized by
the (downscaled) frame width so the estimate is resolution-independent.

The video-level estimate is the MEDIAN of the per-pair shifts, not the mean:
a juggler sweeping across the frame during one or two sampled pairs could
inflate a mean, but must not by itself flip a genuinely static-camera video
to "moving" -- the median is robust to that handful of outlier pairs.
"""
from __future__ import annotations

import math
import statistics
from pathlib import Path

import cv2
import numpy as np

_SAMPLE_WIDTH = 320
_PAIR_GAP_S = 0.5


def _grab_gray(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    """Seek to `frame_idx`, read it, and return a downscaled float32 grayscale
    copy (as phaseCorrelate requires), or None if the read failed."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        return None
    h, w = frame.shape[:2]
    scale = _SAMPLE_WIDTH / w
    target_h = max(1, round(h * scale))
    small = cv2.resize(frame, (_SAMPLE_WIDTH, target_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32)


def estimate_camera_motion(video_path: str | Path, *, samples: int = 10) -> float:
    """Median normalized global-shift magnitude across `samples` sampled
    frame pairs (~0.5s apart, spread through the video).

    Returns 0.0 if the video is too short to form even one pair, or if no
    pair could be read.
    """
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cv2 could not open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
            raise ValueError(f"video has no readable frames/fps: {path}")

        gap = max(1, round(_PAIR_GAP_S * fps))
        last_start = frame_count - gap - 1
        if last_start <= 0 or samples <= 1:
            starts = [0]
        else:
            n = min(samples, last_start + 1)
            starts = sorted({round(i * last_start / (n - 1)) for i in range(n)})

        shifts: list[float] = []
        for s in starts:
            g0 = _grab_gray(cap, s)
            g1 = _grab_gray(cap, s + gap)
            if g0 is None or g1 is None or g0.shape != g1.shape:
                continue
            (dx, dy), _response = cv2.phaseCorrelate(g0, g1)
            shifts.append(math.hypot(dx, dy) / _SAMPLE_WIDTH)
    finally:
        cap.release()

    if not shifts:
        return 0.0
    return float(statistics.median(shifts))


def triage_video(video_path: str | Path, *, threshold: float = 0.004) -> str:
    """Classify a video as "static" or "moving" camera.

    "moving" means `estimate_camera_motion` exceeds `threshold`: MotionDetector
    should not be trusted at face value on such a clip (see module docstring).
    """
    motion = estimate_camera_motion(video_path)
    return "moving" if motion > threshold else "static"
