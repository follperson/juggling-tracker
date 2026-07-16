"""Airborne-count periodicity: run validator + quality score (spec §4)."""
from __future__ import annotations

import numpy as np

from juggletrack.types import Arc


def airborne_count(arcs: list[Arc], t_grid: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(t_grid))
    for a in arcs:
        counts += (t_grid >= a.t_start) & (t_grid <= a.t_end)
    return counts


def periodicity_score(
    arcs: list[Arc],
    t_from: float,
    t_to: float,
    *,
    dt: float = 1.0 / 30.0,
    lag_range: tuple[float, float] = (0.2, 1.2),
) -> tuple[float, float | None]:
    if t_to - t_from < 2 * lag_range[1]:
        # window too short to see two periods at the largest lag: score what we can
        t_to = t_from + 2 * lag_range[1]
    grid = np.arange(t_from, t_to, dt)
    if len(grid) < 8:
        return 0.0, None
    s = airborne_count(arcs, grid)
    s = s - s.mean()
    var = float(np.dot(s, s))
    if var < 1e-9:
        return 0.0, None
    n = len(s)
    full = np.correlate(s, s, mode="full")[n - 1 :]  # raw lag-k sum, k = 0..n-1
    # Normalize each lag by its own overlap count (n - k), not by the fixed
    # zero-lag sum: dividing every lag by `var` (the "biased" ACF estimator)
    # systematically shrinks longer lags, since they average fewer terms yet
    # get compared against the full-window energy. That shrinkage is enough
    # to push a real periodic signal's peak below threshold near the edges of
    # a short window (e.g. a run spanning only ~2-3 periods at the lag of
    # interest). Dividing by the per-lag overlap first (the "unbiased"
    # estimator) removes that bias while leaving lag 0 at exactly 1.0 and the
    # location of the peak unchanged.
    overlap = n - np.arange(n)
    normalized = full * n / (overlap * var)
    lo, hi = int(lag_range[0] / dt), min(int(lag_range[1] / dt) + 1, len(normalized))
    if hi <= lo:
        return 0.0, None
    window = normalized[lo:hi]
    k = int(np.argmax(window))
    return float(window[k]), float((lo + k) * dt)
