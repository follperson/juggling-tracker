"""Shared test utilities: synthetic videos and analytic per-ball juggling tracks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from juggletrack.types import Detection


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


@dataclass(frozen=True)
class Throw:
    """One analytic flight: the ball leaves hand height at t0 and is back at it dur later."""

    ball: int
    t0: float
    dur: float
    rise: float
    x0: float
    x1: float

    def position(self, t: float, hand_y: float) -> tuple[float, float]:
        u = (t - self.t0) / self.dur
        return self.x0 + (self.x1 - self.x0) * u, hand_y - 4.0 * self.rise * u * (1.0 - u)


def cascade_throws(
    n_balls: int = 3, per_ball: int = 6, *, dur: float = 0.6, dwell: float = 0.3,
    rise: float = 0.3, lead: float = 0.5, hands: tuple[float, float] = (0.35, 0.65),
) -> list[Throw]:
    """Evenly staggered throws that cross from hand to hand, in launch order."""
    cycle = dur + dwell
    throws = []
    for ball in range(n_balls):
        for k in range(per_ball):
            x0, x1 = hands if (ball + k) % 2 == 0 else hands[::-1]
            t0 = lead + ball * cycle / n_balls + k * cycle
            throws.append(Throw(ball=ball, t0=t0, dur=dur, rise=rise, x0=x0, x1=x1))
    return sorted(throws, key=lambda th: th.t0)


def juggle_tracks(
    throws: list[Throw], *, fps: float = 30.0, hand_y: float = 0.7, tail: float = 0.5,
) -> list[list[Detection]]:
    """Per-ball label tracks for analytic throws, one Detection per ball per frame.

    A ball follows its throw's parabola while in the air and rests at hand
    height, where it last landed, between throws.
    """
    n_frames = int((max(th.t0 + th.dur for th in throws) + tail) * fps) + 1
    tracks: list[list[Detection]] = []
    for ball in range(max(th.ball for th in throws) + 1):
        own = [th for th in throws if th.ball == ball]
        track = []
        for idx in range(n_frames):
            t = idx / fps
            x, y = own[0].x0, hand_y
            for th in own:
                if th.t0 <= t <= th.t0 + th.dur:
                    x, y = th.position(t, hand_y)
                elif t > th.t0 + th.dur:
                    x = th.x1
            track.append(Detection(frame_idx=idx, t=t, x=x, y=y))
        tracks.append(track)
    return tracks
