"""Global arc extraction: detections -> ballistic arcs (hard-assignment EM).

Identity-free by design: works on the unordered detection cloud, so track-ID
switches upstream cannot corrupt results (spec section 4).
"""
from __future__ import annotations

import math

import numpy as np

from juggletrack.arcs.fit import fit_arc, points_array, x_residuals, y_residuals
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
    # boundary recovery is bounded by _EM_TIME_MARGIN per iteration; 5
    # empirically closes crossing-swap deficits
    em_iters: int = 5,
    max_abs_bx: float = 0.6,
) -> list[Arc]:
    arr = points_array(dets)
    if len(arr) < min_points:
        return []
    # points_array only sorts by t; break ties deterministically on content
    # (x, y, confidence) so results never depend on input ordering/identity.
    # This only guarantees order-independence for rows that differ in
    # (x, y, confidence); fully-identical rows are interchangeable anyway,
    # so their relative order can't affect the result.
    order = np.lexsort((arr[:, 3], arr[:, 2], arr[:, 1], arr[:, 0]))
    arr = arr[order]

    fragments = _link_fragments(arr, link_max_dt, link_max_dist)
    seeds: list[list[int]] = []
    for frag in fragments:
        seeds.extend(_split_ballistic(arr, frag, resid_tol))

    arcs: list[Arc] = []
    for idxs in seeds:
        if len(idxs) < 4:
            continue
        try:
            arcs.append(fit_arc(arr[idxs]))
        except ValueError:
            continue  # degenerate seed (e.g. all-same-timestamp cluster)

    for _ in range(em_iters):
        arcs = _em_assign_refit(arr, arcs, resid_tol)
        arcs = _merge_pass(arr, arcs, resid_tol)
        arcs = _prune(arcs, g_range, resid_tol, min_points, min_duration, max_abs_bx)
        arcs = _gravity_prune(arcs)

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
    """Split a fragment wherever one parabola -- or one x(t) line -- stops explaining it.

    y-rmse alone misses a crossing: two balls sharing the same gravity can
    briefly trace a near-identical y-parabola while their x(t) lines diverge
    (one ball's x rising as the other's falls through the same y-corridor),
    so `_link_fragments` links points from both into one fragment and the
    y-only check here used to let it ride as one seed all the way through
    (mirrors the x-gate `_merge_pass` already applies when unioning two
    already-separate arcs -- this closes the same hole one stage earlier,
    before a fragment is even split into candidate seeds).
    Use `max`, not an rmse, on x: a crossing shows up as the tail few points
    of `cur` suddenly landing off the fitted x-line right as the window
    should split, and averaging that into an rmse-style statistic dilutes
    the signal across the (still mostly x-consistent) rest of the window.
    The threshold constant `2*resid_tol` is reused by `_merge_pass`, but this
    check gates on the MAX x-residual (stricter), while `_merge_pass` gates on
    x-RMSE (looser) — deliberate, because rmse dilutes the tail-point signal.
    """
    pieces: list[list[int]] = []
    cur: list[int] = []
    for i in idxs:
        cur.append(i)
        if len(cur) >= 4:
            arc = fit_arc(arr[cur])
            x_bad = np.max(x_residuals(arc, arr[cur])) > 2 * resid_tol
            # Known gap: some crossing-ball fusions are invisible to both curvature
            # and the incremental x-line check here; next locus would be _link_fragments.
            if arc.rmse > resid_tol or x_bad:
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
        res_both = np.maximum(y_residuals(arc, arr), x_residuals(arc, arr))
        res = np.where(in_span, res_both, np.inf)
        better = res < best_res
        best_res[better] = res[better]
        best_arc[better] = k
    best_arc[best_res > 2 * resid_tol] = -1

    out: list[Arc] = []
    for k in range(len(arcs)):
        member = np.where(best_arc == k)[0]
        if len(member) >= 3:
            try:
                out.append(fit_arc(arr[member]))
            except ValueError:
                continue  # degenerate reassignment (e.g. all-same-timestamp cluster)
    return out


def _merge_pass(arr: np.ndarray, arcs: list[Arc], resid_tol: float) -> list[Arc]:
    """Merge same-ball fragments split by a crossing with another ball's arc.

    Candidates are scanned by ascending ``t_start``, not just the immediate
    list neighbor: a ball crossing another mid-flight can leave its own
    fragments with a different ball's arc time-interleaved between them, so
    the two pieces that truly belong together are not adjacent in this sort
    order. Scanning ahead until the time gap exceeds the threshold finds them
    while keeping the same acceptance test (``union.rmse <= resid_tol``) that
    guards against merging genuinely different balls.

    y-rmse alone is not enough, though: two fragments from *different* balls
    that cross paths through the same y-corridor (e.g. one ball rising in x
    while another falls in x, briefly overlapping in y around the same time)
    can union into a single low-y-rmse arc, because y alone doesn't carry
    ball identity there. A real single ball has an (approximately) constant
    x-velocity across a short merge window, so we additionally require the
    union's own linear x-fit to explain the merged points, and reject
    outright when the two fragments' fitted x-velocities point in opposite
    directions with meaningful magnitude -- the structural signature of a
    crossing rather than one continuous flight.
    """
    arcs = sorted(arcs, key=lambda a: a.t_start)
    t = arr[:, 0]
    n = len(arcs)
    used = [False] * n
    merged: list[Arc] = []
    for i in range(n):
        if used[i]:
            continue
        a = arcs[i]
        best_j, best_union = None, None
        for j in range(i + 1, n):
            if used[j]:
                continue
            b = arcs[j]
            if b.t_start - a.t_end >= 0.2:
                break  # candidates sorted by t_start: gap only grows from here
            # Crossing-balls signature: opposite-signed, non-negligible
            # x-velocities. A single ball's x-velocity can't flip sign like
            # this over such a short window, so reject before even fitting
            # the union.
            if a.bx * b.bx < 0 and abs(a.bx) > 0.02 and abs(b.bx) > 0.02:
                continue
            sel = ((t >= a.t_start) & (t <= a.t_end)) | ((t >= b.t_start) & (t <= b.t_end))
            pts = arr[sel]
            keep_a = y_residuals(a, pts) < 2 * resid_tol
            keep_b = y_residuals(b, pts) < 2 * resid_tol
            pts = pts[keep_a | keep_b]
            if len(pts) < 3:
                continue
            try:
                union = fit_arc(pts)
            except ValueError:
                continue  # degenerate union (e.g. all-same-timestamp cluster)
            if union.rmse > resid_tol:
                continue
            # x-gate: even when the union's y-fit looks fine, a fused pair of
            # crossing balls will not lie on one consistent x(t) line. Reject
            # unions whose x-residuals against their own linear x-fit are too
            # large to be one ball.
            dt = pts[:, 0] - union.t_start
            x_pred = union.bx * dt + union.cx
            x_rmse = float(np.sqrt(np.mean((x_pred - pts[:, 1]) ** 2)))
            if x_rmse > 2 * resid_tol:
                continue
            if best_union is None or union.rmse < best_union.rmse:
                best_j, best_union = j, union
        if best_j is not None:
            used[i] = used[best_j] = True
            merged.append(best_union)
        else:
            merged.append(a)
    return merged


def _prune(
    arcs: list[Arc], g_range: tuple[float, float], resid_tol: float,
    min_points: int, min_duration: float, max_abs_bx: float,
) -> list[Arc]:
    """Drop arcs that aren't physically plausible ball flights.

    ``lo <= ay <= hi`` bounds vertical motion to a plausible gravity range
    (the existing check). ``abs(bx) <= max_abs_bx`` is the same idea applied
    to horizontal motion: with the x-consistency gate now guarding merges
    (see ``_merge_pass``), a handful of unlinked false-positive detections
    can still coincidentally chain into a low-point, low-y-rmse "arc" with
    an implausibly large horizontal velocity -- previously such a fragment
    would usually get silently absorbed into a real neighboring arc by the
    old (too-permissive) merge gate. Rejecting implausible bx here catches
    it directly instead of relying on that absorption as accidental cleanup.
    """
    lo, hi = g_range[0] / 2.0, g_range[1] / 2.0
    return [
        a for a in arcs
        if a.n_points >= min_points
        and a.duration() >= min_duration
        and lo <= a.ay <= hi
        and a.rmse <= resid_tol
        and abs(a.bx) <= max_abs_bx
    ]


def _gravity_prune(arcs: list[Arc], band: float = 0.3) -> list[Arc]:
    """Self-consistency: all real arcs share one gravity, so curvature outliers
    (cross-ball chimeras fit ay far above the cohort) are spurious. Needs a
    quorum of >=4 arcs so the median is trustworthy."""
    if len(arcs) < 4:
        return arcs
    med = float(np.median([a.ay for a in arcs]))
    lo, hi = (1.0 - band) * med, (1.0 + band) * med
    return [a for a in arcs if lo <= a.ay <= hi]
