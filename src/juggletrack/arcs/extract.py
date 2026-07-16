"""Global arc extraction: detections -> ballistic arcs (hard-assignment EM).

Identity-free by design: works on the unordered detection cloud, so track-ID
switches upstream cannot corrupt results (spec section 4).
"""
from __future__ import annotations

import math

import numpy as np

from juggletrack.arcs.fit import fit_arc, points_array, y_residuals
from juggletrack.types import Arc, Detection

_EM_TIME_MARGIN = 0.15  # arcs may claim points this far beyond their current span


def extract_arcs(
    dets: list[Detection],
    *,
    g_range: tuple[float, float] = (0.5, 8.0),
    link_max_dt: float = 0.12,
    link_max_dist: float = 0.08,
    resid_tol: float = 0.02,
    min_points: int = 6,
    min_duration: float = 0.15,
    em_iters: int = 5,
) -> list[Arc]:
    arr = points_array(dets)
    if len(arr) < min_points:
        return []
    # points_array only sorts by t; break ties deterministically on content
    # (x, y, confidence) so results never depend on input ordering/identity.
    order = np.lexsort((arr[:, 3], arr[:, 2], arr[:, 1], arr[:, 0]))
    arr = arr[order]

    fragments = _link_fragments(arr, link_max_dt, link_max_dist)
    seeds: list[list[int]] = []
    for frag in fragments:
        seeds.extend(_split_ballistic(arr, frag, resid_tol))

    arcs = [fit_arc(arr[idxs]) for idxs in seeds if len(idxs) >= 4]

    for _ in range(em_iters):
        arcs = _em_assign_refit(arr, arcs, resid_tol)
        arcs = _merge_pass(arr, arcs, resid_tol)
        arcs = _prune(arcs, g_range, resid_tol, min_points, min_duration)

    arcs.sort(key=lambda a: a.t_start)
    return [a.model_copy(update={"id": i}) for i, a in enumerate(arcs)]


def _link_fragments(arr: np.ndarray, max_dt: float, max_dist: float) -> list[list[int]]:
    """Greedy constant-velocity linker: detection cloud -> candidate fragments."""
    open_frags: list[dict] = []
    done: list[list[int]] = []
    for i in range(len(arr)):
        t, x, y = arr[i, 0], arr[i, 1], arr[i, 2]
        still = []
        for fr in open_frags:
            if t - fr["t"] > max_dt:
                done.append(fr["idxs"])
            else:
                still.append(fr)
        open_frags = still
        best, best_d = None, max_dist
        for fr in open_frags:
            dt = t - fr["t"]
            if dt <= 0:
                continue
            d = math.hypot(x - (fr["x"] + fr["vx"] * dt), y - (fr["y"] + fr["vy"] * dt))
            if d < best_d:
                best, best_d = fr, d
        if best is None:
            open_frags.append({"idxs": [i], "t": t, "x": x, "y": y, "vx": 0.0, "vy": 0.0})
        else:
            dt = t - best["t"]
            best["vx"], best["vy"] = (x - best["x"]) / dt, (y - best["y"]) / dt
            best["t"], best["x"], best["y"] = t, x, y
            best["idxs"].append(i)
    done.extend(fr["idxs"] for fr in open_frags)
    return [f for f in done if len(f) >= 4]


def _split_ballistic(arr: np.ndarray, idxs: list[int], resid_tol: float) -> list[list[int]]:
    """Split a fragment wherever one parabola stops explaining it."""
    pieces: list[list[int]] = []
    cur: list[int] = []
    for i in idxs:
        cur.append(i)
        if len(cur) >= 4 and fit_arc(arr[cur]).rmse > resid_tol:
            pieces.append(cur[:-1])
            cur = [i]
    if len(cur) >= 4:
        pieces.append(cur)
    return [p for p in pieces if len(p) >= 4]


def _em_assign_refit(arr: np.ndarray, arcs: list[Arc], resid_tol: float) -> list[Arc]:
    if not arcs:
        return []
    t = arr[:, 0]
    best_res = np.full(len(arr), np.inf)
    best_arc = np.full(len(arr), -1, dtype=int)
    for k, arc in enumerate(arcs):
        in_span = (t >= arc.t_start - _EM_TIME_MARGIN) & (t <= arc.t_end + _EM_TIME_MARGIN)
        res = np.where(in_span, y_residuals(arc, arr), np.inf)
        better = res < best_res
        best_res[better] = res[better]
        best_arc[better] = k
    best_arc[best_res > 2 * resid_tol] = -1

    out: list[Arc] = []
    for k in range(len(arcs)):
        member = np.where(best_arc == k)[0]
        if len(member) >= 3:
            out.append(fit_arc(arr[member]))
    return out


def _merge_pass(arr: np.ndarray, arcs: list[Arc], resid_tol: float) -> list[Arc]:
    arcs = sorted(arcs, key=lambda a: a.t_start)
    t = arr[:, 0]
    merged: list[Arc] = []
    i = 0
    while i < len(arcs):
        a = arcs[i]
        if i + 1 < len(arcs):
            b = arcs[i + 1]
            if b.t_start - a.t_end < 0.2:
                sel = ((t >= a.t_start) & (t <= a.t_end)) | ((t >= b.t_start) & (t <= b.t_end))
                pts = arr[sel]
                keep_a = y_residuals(a, pts) < 2 * resid_tol
                keep_b = y_residuals(b, pts) < 2 * resid_tol
                pts = pts[keep_a | keep_b]
                if len(pts) >= 3:
                    union = fit_arc(pts)
                    if union.rmse <= resid_tol:
                        merged.append(union)
                        i += 2
                        continue
        merged.append(a)
        i += 1
    return merged


def _prune(
    arcs: list[Arc], g_range: tuple[float, float], resid_tol: float,
    min_points: int, min_duration: float,
) -> list[Arc]:
    lo, hi = g_range[0] / 2.0, g_range[1] / 2.0
    return [
        a for a in arcs
        if a.n_points >= min_points
        and a.duration() >= min_duration
        and lo <= a.ay <= hi
        and a.rmse <= resid_tol
    ]
