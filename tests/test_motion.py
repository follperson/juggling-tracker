"""TDD tests for the motion-based bootstrap detector (MOG2 background subtraction).

Cold-start motivation: on new-environment footage (dark balls vs foliage/dark
shirt, static camera) the appearance-based fine-tuned YOLO detector collapses
to ~29% per-ball recall with 60-85% of its firings being static false
positives, so arc extraction gets nothing to work with and auto-labeling
cold-starts. Background subtraction instead flags anything that MOVES against
a learned background, independent of appearance -- the existing
arc-verification gate (extract_arcs / select_autolabels) supplies the
precision a bare motion detector lacks on its own.
"""
from __future__ import annotations

import cv2
import numpy as np

from juggletrack.detect import BallDetector
from juggletrack.video.reader import VideoReader
from tests.helpers import write_test_video


def test_protocol_conformance():
    from juggletrack.detect.motion import MotionDetector

    assert isinstance(MotionDetector(), BallDetector)


def test_moving_circle_detected(tmp_path):
    from juggletrack.detect.motion import MotionDetector

    video_path = tmp_path / "moving.mp4"
    # Larger frame than the helper default (320x240) so the fixed-radius-10
    # circle's normalized area lands comfortably inside MotionDetector's
    # default [min_area, max_area] band instead of straddling its edge.
    centers = write_test_video(video_path, n_frames=60, size=(640, 480))

    det = MotionDetector(warmup_frames=10)
    scored = 0
    hits = 0
    with VideoReader(video_path) as reader:
        for idx, t, frame in reader.frames():
            dets = det.detect(frame, idx, t)
            if idx < det.warmup_frames:
                continue  # background model still learning
            scored += 1
            gx, gy = centers[idx]
            if any(abs(d.x - gx) <= 0.05 and abs(d.y - gy) <= 0.05 for d in dets):
                hits += 1

    assert scored > 0
    assert hits / scored >= 0.8


def test_static_object_absorbed(tmp_path):
    """A clip with BOTH a moving circle and a static one: after the background
    model has had time to learn (warmup + enough history), the static circle's
    position should rarely fire (absorbed into the background) while the
    moving circle keeps being detected.
    """
    from juggletrack.detect.motion import MotionDetector

    video_path = tmp_path / "mixed.mp4"
    # Same larger-frame reasoning as test_moving_circle_detected above.
    n_frames, fps, (w, h) = 90, 30.0, (640, 480)
    # Kept well outside the moving circle's orbit (x,y in [0.2, 0.8]) so the
    # two never overlap into one blob and confound the per-position scoring.
    static_center = (0.9, 0.1)

    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened(), "cv2.VideoWriter failed to open (mp4v codec missing?)"
    moving_centers: list[tuple[float, float]] = []
    for i in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        phase = 2 * np.pi * i / n_frames
        mx = 0.5 + 0.3 * np.cos(phase)
        my = 0.5 + 0.3 * np.sin(phase)
        cv2.circle(frame, (int(mx * w), int(my * h)), 10, (255, 255, 255), -1)
        cv2.circle(
            frame, (int(static_center[0] * w), int(static_center[1] * h)), 10,
            (255, 255, 255), -1,
        )
        writer.write(frame)
        moving_centers.append((mx, my))
    writer.release()

    history = 50  # tuned down from the default 300 for test speed
    det = MotionDetector(history=history, warmup_frames=10)
    skip = det.warmup_frames + history  # let the static blob fully absorb
    scored = 0
    static_hits = 0
    moving_hits = 0
    with VideoReader(video_path) as reader:
        for idx, t, frame in reader.frames():
            dets = det.detect(frame, idx, t)
            if idx < skip:
                continue
            scored += 1
            sx, sy = static_center
            if any(abs(d.x - sx) <= 0.05 and abs(d.y - sy) <= 0.05 for d in dets):
                static_hits += 1
            mx, my = moving_centers[idx]
            if any(abs(d.x - mx) <= 0.05 and abs(d.y - my) <= 0.05 for d in dets):
                moving_hits += 1

    assert scored > 0
    assert static_hits / scored < 0.10
    assert moving_hits / scored >= 0.8


def _feed_black_warmup(det, n: int, size: tuple[int, int]) -> None:
    """Advance a MotionDetector past warmup on an all-black background so its
    MOG2 model has learned a clean background before the frame under test."""
    w, h = size
    black = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(n):
        det.detect(black, i, i / 30.0)


def test_busy_frame_guard_suppresses_dense_scene():
    """Field motivation: harvested clips with a MOVING camera make MOG2 fire
    on nearly the whole frame (hundreds of blobs), which floods downstream arc
    extraction with junk (both quality and O(n^2)-blowup performance -- see
    arcs/extract.py's _merge_pass). A frame this busy can't be trusted, so it
    should be dropped outright rather than passed downstream.
    """
    from juggletrack.detect.motion import MotionDetector

    size = (640, 480)
    w, h = size
    det = MotionDetector(warmup_frames=5, max_blobs=12)
    _feed_black_warmup(det, det.warmup_frames + 1, size)

    busy = np.zeros((h, w, 3), dtype=np.uint8)
    n_blobs = 20  # > max_blobs
    for j in range(n_blobs):
        cx = 20 + j * 30
        cv2.circle(busy, (cx, h // 2), 8, (255, 255, 255), -1)

    idx = det.warmup_frames + 1
    dets = det.detect(busy, idx, idx / 30.0)
    assert dets == []


def test_busy_frame_guard_lets_sparse_scene_through():
    """The mirror case: a frame with <= max_blobs size-filtered blobs is not
    considered busy and passes through normally."""
    from juggletrack.detect.motion import MotionDetector

    size = (640, 480)
    w, h = size
    det = MotionDetector(warmup_frames=5, max_blobs=12)
    _feed_black_warmup(det, det.warmup_frames + 1, size)

    sparse = np.zeros((h, w, 3), dtype=np.uint8)
    n_blobs = 8  # <= max_blobs
    for j in range(n_blobs):
        cx = 40 + j * 70
        cv2.circle(sparse, (cx, h // 2), 8, (255, 255, 255), -1)

    idx = det.warmup_frames + 1
    dets = det.detect(sparse, idx, idx / 30.0)
    assert len(dets) == n_blobs
