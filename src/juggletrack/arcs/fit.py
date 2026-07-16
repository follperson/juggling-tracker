"""Weighted least-squares parabola fitting for ballistic arcs."""
from __future__ import annotations

import numpy as np

from juggletrack.types import Arc, Detection


def points_array(dets: list[Detection]) -> np.ndarray:
    """(N, 4) array of [t, x, y, confidence], sorted by t."""
    arr = np.array([[d.t, d.x, d.y, d.confidence] for d in dets], dtype=float)
    if arr.size == 0:
        return arr.reshape(0, 4)
    return arr[np.argsort(arr[:, 0])]


def fit_arc(arr: np.ndarray, arc_id: int = -1) -> Arc:
    if len(arr) < 3:
        raise ValueError(f"fit_arc needs >= 3 points, got {len(arr)}")
    arr = arr[np.argsort(arr[:, 0])]
    t0 = arr[0, 0]
    dt = arr[:, 0] - t0
    w = arr[:, 3]
    ay, by, cy = np.polyfit(dt, arr[:, 2], 2, w=w)
    bx, cx = np.polyfit(dt, arr[:, 1], 1, w=w)
    resid = (ay * dt * dt + by * dt + cy) - arr[:, 2]
    # polyfit minimizes sum((w*resid)^2), so the consistent diagnostic
    # averages resid^2 with weights w^2 (uniform-confidence data unaffected).
    rmse = float(np.sqrt(np.average(resid**2, weights=w**2)))
    return Arc(
        id=arc_id, t_start=float(t0), t_end=float(arr[-1, 0]),
        ay=float(ay), by=float(by), cy=float(cy), bx=float(bx), cx=float(cx),
        n_points=len(arr), rmse=rmse,
    )


def y_residuals(arc: Arc, arr: np.ndarray) -> np.ndarray:
    dt = arr[:, 0] - arc.t_start
    return np.abs(arc.ay * dt * dt + arc.by * dt + arc.cy - arr[:, 2])
