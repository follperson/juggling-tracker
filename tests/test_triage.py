"""TDD tests for camera-motion triage (data/triage.py).

Motivation: MotionDetector (detect/motion.py) assumes a static camera --
MOG2 background subtraction only isolates a moving ball if the background
itself holds still. Harvested clips (Kinetics, commons, external) often have
a MOVING camera instead, which makes MOG2 fire on nearly the whole frame
(see the busy-frame guard added to motion.py). Triage classifies a clip as
"static" or "moving" BEFORE motion-labeling so moving-camera clips can be
routed differently instead of silently producing near-empty (busy-frame-
guarded) detections.

Method under test: sample a handful of frame pairs ~0.5s apart spread
through the video, downscale to a fixed width, and run phase correlation on
each pair to recover the dominant (background) translational shift --
independent of a small foreground subject (e.g. a juggler) moving through
an otherwise static scene. The MEDIAN pair shift (not mean) is the estimate,
specifically so that a juggler's own on-screen motion in one or two sampled
pairs can't tip a genuinely static-camera video into "moving".
"""
from __future__ import annotations

import cv2
import numpy as np


def _write_static_textured_video(
    path,
    n_frames: int = 90,
    fps: float = 30.0,
    size: tuple[int, int] = (640, 480),
) -> None:
    """Write an mp4 with a STATIC textured background (camera never moves)
    and a small white circle orbiting on top of it -- a juggler-like
    foreground subject moving through an otherwise unchanging scene.

    Deliberately NOT `tests.helpers.write_test_video`: that helper's backdrop
    is flat black, which carries zero non-DC spectral energy -- a global
    cv2.phaseCorrelate then has nothing to align on except the one moving
    circle, and reports its full orbital displacement as "camera motion"
    (empirically ~0.27 normalized, confirmed against this file's static test
    before this fixture was introduced). That's an artifact of a textureless
    fixture, not a real static-camera scene: real footage's background
    (ground, walls, foliage) actually dominates the frame's spectral content,
    which is the premise `estimate_camera_motion`'s docstring relies on
    ("phase correlation tracks the dominant/background alignment"). Giving
    the background real texture here is what makes this fixture a faithful
    stand-in for that assumption.
    """
    w, h = size
    rng = np.random.default_rng(1)
    texture = rng.integers(0, 256, size=(h, w), dtype=np.uint8).astype(np.float32)
    texture = cv2.GaussianBlur(texture, (0, 0), 3)
    bg = cv2.cvtColor(texture.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter failed to open (mp4v codec missing?)")
    for i in range(n_frames):
        frame = bg.copy()
        phase = 2 * np.pi * i / n_frames
        cx = 0.5 + 0.3 * np.cos(phase)
        cy = 0.5 + 0.3 * np.sin(phase)
        cv2.circle(frame, (int(cx * w), int(cy * h)), 10, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def _write_panning_video(
    path,
    n_frames: int = 90,
    fps: float = 30.0,
    size: tuple[int, int] = (640, 480),
    shift_px: int = 3,
) -> None:
    """Write an mp4 that pans a fixed camera window across a larger textured
    image a few px per frame -- simulates a moving camera, as opposed to
    write_test_video's static background with only a small moving subject.
    """
    w, h = size
    margin = 20
    tex_w = w + shift_px * n_frames + margin
    tex_h = h + margin
    rng = np.random.default_rng(0)
    texture = rng.integers(0, 256, size=(tex_h, tex_w), dtype=np.uint8).astype(np.float32)
    texture = cv2.GaussianBlur(texture, (0, 0), 3)
    texture = texture.astype(np.uint8)

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter failed to open (mp4v codec missing?)")
    for i in range(n_frames):
        x0 = i * shift_px
        window = texture[margin // 2: margin // 2 + h, x0: x0 + w]
        frame = cv2.cvtColor(window, cv2.COLOR_GRAY2BGR)
        writer.write(frame)
    writer.release()


def test_estimate_camera_motion_static_scene_near_zero(tmp_path):
    from juggletrack.data.triage import estimate_camera_motion

    video_path = tmp_path / "static.mp4"
    _write_static_textured_video(video_path, n_frames=90, size=(640, 480))

    motion = estimate_camera_motion(video_path)
    assert motion < 0.004


def test_estimate_camera_motion_panning_scene_above_threshold(tmp_path):
    from juggletrack.data.triage import estimate_camera_motion

    video_path = tmp_path / "panning.mp4"
    _write_panning_video(video_path, n_frames=90, size=(640, 480), shift_px=3)

    motion = estimate_camera_motion(video_path)
    assert motion > 0.004


def test_triage_video_static(tmp_path):
    from juggletrack.data.triage import triage_video

    video_path = tmp_path / "static.mp4"
    _write_static_textured_video(video_path, n_frames=90, size=(640, 480))

    assert triage_video(video_path) == "static"


def test_triage_video_moving(tmp_path):
    from juggletrack.data.triage import triage_video

    video_path = tmp_path / "panning.mp4"
    _write_panning_video(video_path, n_frames=90, size=(640, 480), shift_px=3)

    assert triage_video(video_path) == "moving"
