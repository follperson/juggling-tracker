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
) -> tuple[float | None, float | None]:
    """Airborne-count periodicity of ``arcs`` over ``[t_from, t_to]``.

    Returns ``(None, None)`` when the window can't judge periodicity at all:
    either it can't hold two periods at any lag the caller asked about, or it
    has too few samples. This is NOT the same as a real score of 0.0 (which
    means "judged, and no periodicity found" -- e.g. a flat/empty signal).
    Callers that need a single float (e.g. ``Run.quality``) must decide how
    to collapse "can't judge" themselves; see ``events.runs.segment_runs``.

    The window is NEVER extended/zero-padded past ``t_to`` to manufacture a
    view of two periods: arcs don't exist beyond the real window, so padding
    the signal with implied zeros there biases the autocorrelation. Instead
    the searchable lag band is clipped to what the window can actually hold:
    ``[lag_range[0], min(lag_range[1], span / 2)]``. If that clipped band is
    empty (the window is too short even for the smallest requested lag), the
    result is "can't judge", not a number computed over a window that isn't
    the one asked for.
    """
    span = t_to - t_from
    lag_lo = lag_range[0]
    lag_hi = min(lag_range[1], span / 2.0)
    if lag_hi < lag_lo:
        return None, None
    grid = np.arange(t_from, t_to, dt)
    if len(grid) < 8:
        return None, None
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
    lo, hi = int(lag_lo / dt), min(int(lag_hi / dt) + 1, len(normalized))
    if hi <= lo:
        return None, None
    window = normalized[lo:hi]
    k = int(np.argmax(window))
    return float(window[k]), float((lo + k) * dt)
