"""Shared frame-drawing helpers for the offline overlay and the live HUD.

cv2 is imported inside each function (not at module scope) so that
``realtime.py`` and other modules can import types/helpers from this file
without transitively pulling in cv2 through a module-level import.
"""
from __future__ import annotations

import numpy as np

from juggletrack.types import Arc, Detection

GREEN = (0, 200, 0)
YELLOW = (0, 220, 220)
RED = (0, 0, 255)
WHITE = (240, 240, 240)


def _scale(frame: np.ndarray) -> float:
    """Sizes below were tuned on 480-pixel frames; grow them with the shorter
    side and never shrink them."""
    return max(1.0, min(frame.shape[:2]) / 480)


def _px(size: float, scale: float) -> int:
    return max(1, round(size * scale))


def draw_hand_line(frame: np.ndarray, hand_line_y: float) -> None:
    import cv2

    h, w = frame.shape[:2]
    y = int(hand_line_y * h)
    cv2.line(frame, (0, y), (w, y), YELLOW, _px(1, _scale(frame)))


def draw_detections(frame: np.ndarray, dets: list[Detection]) -> None:
    import cv2

    h, w = frame.shape[:2]
    s = _scale(frame)
    for d in dets:
        cv2.circle(frame, (int(d.x * w), int(d.y * h)), _px(4, s), WHITE, _px(1, s))


def draw_arc_tails(frame: np.ndarray, arcs: list[Arc], t: float, *, tail_s: float = 0.4) -> None:
    import cv2

    h, w = frame.shape[:2]
    for arc in arcs:
        if not (arc.t_start <= t <= arc.t_end + 0.1):
            continue
        t0 = max(arc.t_start, t - tail_s)
        ts = np.arange(t0, min(t, arc.t_end) + 1e-9, 1 / 60)
        pts = np.array([[int(arc.x_at(tt) * w), int(arc.y_at(tt) * h)] for tt in ts])
        if len(pts) >= 2:
            cv2.polylines(frame, [pts], False, GREEN, _px(2, _scale(frame)))


def draw_hud(frame: np.ndarray, text: str, *, y: int = 24) -> None:
    """Draw one line of HUD text at baseline row ``y`` (default: the
    original single-line position). ``y`` lets callers stack multiple HUD
    lines (e.g. live.py's run-history line below the main status line).
    Position and size scale with the frame, so stacked lines stay apart."""
    import cv2

    s = _scale(frame)
    cv2.putText(frame, text, (_px(10, s), _px(y, s)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s,
                WHITE, _px(2, s))


def draw_drop_marker(frame: np.ndarray, x: float, y: float) -> None:
    import cv2

    h, w = frame.shape[:2]
    s = _scale(frame)
    cv2.drawMarker(frame, (int(x * w), int(y * h)), RED, cv2.MARKER_TILTED_CROSS,
                   _px(24, s), _px(3, s))
