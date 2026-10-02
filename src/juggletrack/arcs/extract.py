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

# A bin's detections only chain into one "occupancy run" while consecutive
# timestamps are within this many seconds of each other (see
# filter_static_detections). Comfortably above a normal per-frame/per-stride
# interval (a real object present in a cell keeps re-appearing every frame,
# with gaps this small) and comfortably below a real juggling cascade's
# throw-to-throw period (>=0.45s in every existing sim config): a juggled
# ball passing back through the same cell on a *later* throw always leaves a
# gap well past this, so it starts a new run instead of extending one that
# spans the whole video.
_STATIC_RUN_GAP_S = 0.15


def filter_static_detections(
    dets: list[Detection], *, cell: float = 0.03, max_span_s: float = 1.5
) -> list[Detection]:
    """Drop detections parked in one spatial cell for too long: static clutter,
    not a ball.

    Field motivation: a fine-tuned detector can emit persistent high-confidence
    false positives at fixed image locations (background objects) on new-
    environment footage. The greedy nearest-neighbor linker in
    ``_link_fragments`` locks onto these static clusters and absorbs real ball
    detections into junk fragments, so filtering them out *before* linking is
    the fix (see docs/superpowers/plans/2026-07-16-plan3-flywheel-turn2-findings.md).

    Physics rationale: a juggled ball never *dwells* in one small region for
    long -- hand dwell is ~0.2-0.5s, and flights sweep across the frame. A
    detection cluster occupying one tiny spatial cell continuously across many
    seconds is therefore a static object, not a ball: bin detections into
    ``cell``-sized (x, y) grid cells (``floor(x/cell), floor(y/cell)``), and
    within each cell, chain consecutive (sorted-by-time) detections into
    "occupancy runs" -- a gap of more than ``_STATIC_RUN_GAP_S`` between
    neighbors starts a new run. Any cell with a run spanning more than
    ``max_span_s`` is judged static, and every detection in that cell is
    dropped (not just the offending run's points -- see the module-level
    comment above for why the gap threshold is chosen so this only ever
    matches genuine continuous presence). Survivors are returned in input
    order; bin membership and run spans are computed from each cell's own
    sorted timestamps regardless of input order, so this is order-invariant
    -- shuffling the input can't change which detections get kept.

    Why runs, not the raw (max t - min t) envelope: juggling is periodic --
    every same-hand throw starts and ends at the exact same hand position, so
    a real ball's own cell near a hand gets hit again on every later throw of
    that hand. The raw envelope across *all* those separate, brief visits
    spans nearly the whole video, indistinguishable by that measure alone
    from one continuously-present static object (confirmed empirically: on
    every multi-throw ``simulate_cascade`` scenario in this test suite, the
    raw-envelope version discards real detections, worst case 100% of them).
    A static object, unlike a revisited-but-otherwise-empty cell, is detected
    on essentially every frame throughout its whole span -- that continuity,
    not just the time envelope, is what "dwell" in the physics rationale
    above actually means, and is what the run-chaining picks out.

    v1 known gap: a static object that jitters across a cell boundary splits
    across two adjacent bins instead of landing in one. In the common case
    each split bin is *still* continuously occupied throughout the window
    (the object hasn't moved, so both bins keep getting hit on every frame)
    and both get dropped correctly anyway; only an unlucky short-looking
    timing pattern within one split bin could let a few of its points slip
    through. Not handled here.
    """
    bins: dict[tuple[int, int], list[float]] = {}
    keys: list[tuple[int, int]] = []
    for d in dets:
        k = (math.floor(d.x / cell), math.floor(d.y / cell))
        keys.append(k)
        bins.setdefault(k, []).append(d.t)

    static_bin: dict[tuple[int, int], bool] = {}
    for k, ts in bins.items():
        ts = sorted(ts)
        run_start = ts[0]
        prev = ts[0]
        is_static = False
        for t in ts[1:]:
            if t - prev > _STATIC_RUN_GAP_S:
                run_start = t  # gap breaks the run: start a fresh one
            elif t - run_start > max_span_s:
                is_static = True
                break
            prev = t
        static_bin[k] = is_static

    return [d for d, k in zip(dets, keys) if not static_bin[k]]


def extract_arcs(
    dets: list[Detection],
    *,
    # Floor lowered 0.5 -> 0.1: normalized gravity scales with framing (a tight
    # crop shrinks ay below the old 0.25 ay-floor and zeroed out extraction on
    # ground-truth footage). KNOWN GAP accepted with this change: a cohort of
    # SELF-consistent slow-drift junk (ay in [0.05, 0.25)) now survives — the
    # median-relative _gravity_prune only rejects outliers within a cohort, it
    # cannot invalidate a junk-consistent cohort, and the old absolute floor
    # was the only arc-level defense. Pinned by
    # test_slow_drift_junk_cohort_known_gap.
    g_range: tuple[float, float] = (0.1, 8.0),
    # 0.12 -> 0.18: 0.12s tolerated zero consecutive missed-detection frames
    # at 24fps stride-2 (and only ~2 at 30fps stride-1) -- fine for in-domain
    # footage, but turn-4's outdoor/domain-shifted holdout showed 3-4
    # consecutive-frame detection gaps (0.125s-0.166s) clustering around
    # single flights, exceeding the old budget and costing 60% of that
    # video's real misses (see docs/superpowers/sdd/turn4-diagnosis.md).
    # 0.18s tolerates 3 missed frames at 24fps.
    link_max_dt: float = 0.18,
    link_max_dist: float = 0.08,
    resid_tol: float = 0.02,
    min_points: int = 6,
    min_duration: float = 0.15,
    # boundary recovery is bounded by _EM_TIME_MARGIN per iteration; 5
    # empirically closes crossing-swap deficits
    em_iters: int = 5,
    max_abs_bx: float = 0.6,
    static_cell: float = 0.03,
    static_max_span_s: float = 1.5,
) -> list[Arc]:
    dets = filter_static_detections(dets, cell=static_cell, max_span_s=static_max_span_s)
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

    # Real, distinct ids are required before the split-stitch pass: every
    # arc up to this point still carries fit_arc's default id=-1, which
    # collides with assign_detections's own "no arc fits" sentinel.
    arcs.sort(key=lambda a: a.t_start)
    arcs = [a.model_copy(update={"id": i}) for i, a in enumerate(arcs)]

    arcs = _stitch_splits(dets, arcs, resid_tol)

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


# A closed-form fit statistic must clear its threshold by this much before it
# may stand in for fit_arc; anything closer is re-decided by fit_arc itself.
_EXACT_MARGIN = 1e-7
# Normal equations whose determinant is below this fraction of the product of
# their diagonal (det of the column-normalized Gram matrix, in (0, 1]) are too
# ill-conditioned to trust the closed-form solve. Measured: the closed-form
# statistics stay within 3e-14 of fit_arc's above 1e-6 but drift to 4e-7 by
# 1e-10, and real-session fits sit at 3e-6 and up.
_MIN_REL_DET = 1e-6


class _Moments:
    """Weighted power sums of a point set, with dt measured from ``t0``.

    The weight w is the detection confidence, fit_arc's polyfit weight, so
    polyfit minimizes sum((w*resid)^2) and every sum carries w^2:
    ``s[k] = sum(w^2 dt^k)`` for k = 0..4, ``sy[k] = sum(w^2 dt^k y)`` for
    k = 0..2 and ``sx[k] = sum(w^2 dt^k x)`` for k = 0..1.
    """

    __slots__ = ("t0", "s", "sy", "sx")

    def __init__(self, t0: float) -> None:
        self.t0 = t0
        self.s = [0.0] * 5
        self.sy = [0.0] * 3
        self.sx = [0.0] * 2

    @classmethod
    def of(cls, pts: np.ndarray) -> _Moments:
        """Moments of ``pts`` sorted by t, so ``t0`` is fit_arc's ``t_start``."""
        m = cls(float(pts[0, 0]))
        pw = np.vander(pts[:, 0] - m.t0, 5, increasing=True) * (pts[:, 3] ** 2)[:, None]
        m.s = pw.sum(axis=0).tolist()
        m.sy = (pts[:, 2] @ pw[:, :3]).tolist()
        m.sx = (pts[:, 1] @ pw[:, :2]).tolist()
        return m

    def add(self, t: float, x: float, y: float, w: float) -> None:
        dt = t - self.t0
        s, sy, sx = self.s, self.sy, self.sx
        p = w * w
        s[0] += p
        sy[0] += p * y
        sx[0] += p * x
        p *= dt
        s[1] += p
        sy[1] += p * y
        sx[1] += p * x
        p *= dt
        s[2] += p
        sy[2] += p * y
        p *= dt
        s[3] += p
        s[4] += p * dt

    def fit_residuals(self, pts: np.ndarray) -> tuple[float, np.ndarray] | None:
        """Solve the weighted least-squares parabola y(dt) and line x(dt) from
        the sums, then evaluate them on ``pts`` (the summed points): returns
        the weighted y-rmse exactly as fit_arc defines it and the signed x
        residuals, or None when the y system is too ill-conditioned. The x
        system is its leading 2x2 block, so by Fischer's inequality the x
        system's relative determinant is at least the y system's."""
        s0, s1, s2, s3, s4 = self.s
        c11 = s2 * s0 - s1 * s1  # also the determinant of the x system
        c12 = s2 * s1 - s3 * s0
        c13 = s3 * s1 - s2 * s2
        det = s4 * c11 + s3 * c12 + s2 * c13
        if not (math.isfinite(det) and det > _MIN_REL_DET * s4 * s2 * s0):
            return None
        c22 = s4 * s0 - s2 * s2
        c23 = s3 * s2 - s4 * s1
        c33 = s4 * s2 - s3 * s3
        r0, r1, r2 = self.sy
        ay = (c11 * r2 + c12 * r1 + c13 * r0) / det
        by = (c12 * r2 + c22 * r1 + c23 * r0) / det
        cy = (c13 * r2 + c23 * r1 + c33 * r0) / det
        bx = (s0 * self.sx[1] - s1 * self.sx[0]) / c11
        cx = (s2 * self.sx[0] - s1 * self.sx[1]) / c11

        dt = pts[:, 0] - self.t0
        ry = ay * dt * dt + by * dt + cy - pts[:, 2]
        w2 = pts[:, 3] * pts[:, 3]
        y_rmse = math.sqrt(float(w2 @ (ry * ry)) / s0)
        return y_rmse, bx * dt + cx - pts[:, 1]


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

    Fast path: running weighted moments of the current piece give its
    least-squares fit in closed form, so most steps skip fit_arc. That fit
    only ever decides "keep extending", and only when its y-rmse and max
    x-residual both clear their thresholds by ``_EXACT_MARGIN``; every other
    step, including every split, is decided by fit_arc as before. The y side
    is safe even for an imperfect solve, because any parabola's rmse is at
    least the least-squares minimum fit_arc finds; the x side relies on the
    well-conditioned 2x2 solve plus the margin. Requires ``idxs`` in strictly
    increasing t (``_link_fragments`` links only forward in time), so the
    piece's first point is fit_arc's ``t_start``.
    """
    pieces: list[list[int]] = []
    cur: list[int] = []
    rows = arr[idxs]
    for k, (i, row) in enumerate(zip(idxs, rows.tolist())):
        if not cur:
            mom = _Moments(row[0])
        cur.append(i)
        mom.add(*row)
        if len(cur) >= 4:
            # cur is always the run of idxs ending at k, so its rows are a slice
            fast = mom.fit_residuals(rows[k + 1 - len(cur) : k + 1])
            if (
                fast is not None
                and fast[0] <= resid_tol - _EXACT_MARGIN
                and np.max(np.abs(fast[1])) <= 2 * resid_tol - _EXACT_MARGIN
            ):
                continue
            arc = fit_arc(arr[cur])
            x_bad = np.max(x_residuals(arc, arr[cur])) > 2 * resid_tol
            # Known gap: some crossing-ball fusions are invisible to both curvature
            # and the incremental x-line check here; next locus would be _link_fragments.
            if arc.rmse > resid_tol or x_bad:
                pieces.append(cur[:-1])
                cur = [i]
                mom = _Moments(row[0])
                mom.add(*row)
    if len(cur) >= 4:
        pieces.append(cur)
    return [p for p in pieces if len(p) >= 4]


def _em_assign_refit(arr: np.ndarray, arcs: list[Arc], resid_tol: float) -> list[Arc]:
    """Hand each point to its best-fitting arc, then refit every arc from its points.

    Requires ``arr`` sorted by t (extract_arcs sorts it), so each arc's claim
    window ``[t_start - _EM_TIME_MARGIN, t_end + _EM_TIME_MARGIN]`` is one
    contiguous index range and only that slice is scored.
    """
    if not arcs:
        return []
    t = arr[:, 0]
    best_res = np.full(len(arr), np.inf)
    best_arc = np.full(len(arr), -1, dtype=int)
    for k, arc in enumerate(arcs):
        lo = np.searchsorted(t, arc.t_start - _EM_TIME_MARGIN, side="left")
        hi = np.searchsorted(t, arc.t_end + _EM_TIME_MARGIN, side="right")
        win = arr[lo:hi]
        res = np.maximum(y_residuals(arc, win), x_residuals(arc, win))
        better = res < best_res[lo:hi]
        best_res[lo:hi][better] = res[better]
        best_arc[lo:hi][better] = k
    best_arc[best_res > 2 * resid_tol] = -1

    # stable, so each arc's members stay in ascending row order
    order = np.argsort(best_arc, kind="stable")
    bounds = np.searchsorted(best_arc[order], np.arange(len(arcs) + 1), side="left")
    out: list[Arc] = []
    for k in range(len(arcs)):
        member = order[bounds[k] : bounds[k + 1]]
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

    Requires ``arr`` sorted by t (extract_arcs sorts it), so each arc's
    ``[t_start, t_end]`` span is one contiguous index range and a union's
    rows come from two slices instead of a mask over every point.

    A union whose closed-form least-squares fit (``_Moments``) misses either
    acceptance bound by more than ``_EXACT_MARGIN`` is rejected without
    running fit_arc; every other union goes through fit_arc's acceptance test
    unchanged. Rejecting on y-rmse rests on the solve being accurate, which
    ``_MIN_REL_DET`` guards: duplicate timestamps leaving fewer than three
    distinct times fall below it and keep polyfit's own handling.
    """
    arcs = sorted(arcs, key=lambda a: a.t_start)
    t = arr[:, 0]
    lo = np.searchsorted(t, [a.t_start for a in arcs], side="left").tolist()
    hi = np.searchsorted(t, [a.t_end for a in arcs], side="right").tolist()
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
            # sorted by t_start, so lo[i] <= lo[j]: overlapping or touching
            # spans are one slice, disjoint ones two
            if lo[j] <= hi[i]:
                pts = arr[lo[i] : max(hi[i], hi[j])]
            else:
                pts = np.concatenate((arr[lo[i] : hi[i]], arr[lo[j] : hi[j]]))
            keep_a = y_residuals(a, pts) < 2 * resid_tol
            keep_b = y_residuals(b, pts) < 2 * resid_tol
            pts = pts[keep_a | keep_b]
            if len(pts) < 3:
                continue
            fast = _Moments.of(pts).fit_residuals(pts)
            if fast is not None and (
                fast[0] > resid_tol + _EXACT_MARGIN
                or math.sqrt(float(np.mean(fast[1] ** 2))) > 2 * resid_tol + _EXACT_MARGIN
            ):
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


# Split-stitch tunables (turn-4 overcount-surgeon bake-off, stage B): merges
# a single flight that a brief mid-flight detection dropout split into two
# sequential arcs. Deliberately conservative -- measured on the full turn-4
# battery to fire only on genuine same-flight splits (2 true repairs on af1,
# 0 elsewhere): a gap budget, midpoint continuity in both y and x, curvature
# (ay) agreement, compatible x-velocity sign, and a strict union-refit
# acceptance test are ALL required together before two arcs are merged.
_STITCH_GAP_MAX = 0.25
_STITCH_AY_BAND = 0.30


def _stitch_splits(dets: list[Detection], arcs: list[Arc], resid_tol: float) -> list[Arc]:
    """Merge sequential same-flight arc splits (final pass in extract_arcs).

    A brief mid-flight detection gap can make ``_link_fragments`` /
    ``_split_ballistic`` mint two arcs for what was really one continuous
    ball flight (see test_split_flight_gets_stitched_back_into_one_arc).
    This scans consecutive-in-time arc pairs and, only when their fitted
    parabolas agree closely enough across the gap to plausibly be one
    flight, refits the union of their member detections and accepts the
    merge only if that union is itself a clean single-arc fit -- the same
    acceptance bar (``resid_tol`` / ``2*resid_tol`` on x) real extraction
    uses everywhere else, so a stitched arc is never worse-fit than a
    freshly-extracted one would be. Requires ``arcs`` to already carry
    real, distinct ids (assign_detections's "no arc fits" sentinel is -1,
    which collides with fit_arc's default arc_id).
    """
    arcs = sorted(arcs, key=lambda a: a.t_start)
    continuity_tol = 2.0 * resid_tol
    changed = True
    while changed:
        changed = False
        for i in range(len(arcs) - 1):
            a = arcs[i]
            for j in range(i + 1, len(arcs)):
                b = arcs[j]
                gap = b.t_start - a.t_end
                if gap > _STITCH_GAP_MAX:
                    break  # candidates sorted by t_start: gap only grows from here
                if gap < 0:
                    continue
                tm = a.t_end + 0.5 * gap
                if abs(a.y_at(tm) - b.y_at(tm)) >= continuity_tol:
                    continue
                if abs(a.x_at(tm) - b.x_at(tm)) >= continuity_tol:
                    continue
                if abs(a.ay - b.ay) > _STITCH_AY_BAND * max(a.ay, b.ay):
                    continue
                # Crossing-balls signature (same guard as _merge_pass): a
                # single ball's x-velocity can't flip sign like this across
                # such a short gap.
                if a.bx * b.bx < 0 and abs(a.bx) > 0.02 and abs(b.bx) > 0.02:
                    continue
                window = [d for d in dets if a.t_start - 0.02 <= d.t <= b.t_end + 0.02]
                owner = assign_detections(window, [a, b], resid_tol=resid_tol)
                member = [d for d, o in zip(window, owner) if o in (a.id, b.id)]
                if len(member) < 4:
                    continue
                try:
                    union = fit_arc(points_array(member))
                except ValueError:
                    continue
                if union.rmse > resid_tol:
                    continue
                pts = points_array(member)
                dt = pts[:, 0] - union.t_start
                x_rmse = float(np.sqrt(np.mean((union.bx * dt + union.cx - pts[:, 1]) ** 2)))
                if x_rmse > 2 * resid_tol:
                    continue
                union = union.model_copy(update={"id": a.id})
                arcs = arcs[:i] + [union] + [c for c in arcs[i + 1 :] if c is not b]
                arcs.sort(key=lambda arc: arc.t_start)
                changed = True
                break
            if changed:
                break
    return arcs


# Arc-level parallel-arc dedup defaults (Plan 5 task 2b; full adjudication
# and combo-sweep history recorded verbatim in AnalyzeConfig's own comment
# above the arc_dedup_* fields, and in the committed findings doc
# docs/superpowers/plans/2026-08-03-meschke-validation-findings.md.
# .superpowers/sdd/task-2b-report.md §§9-10 has additional per-round detail
# but is an untracked local file, not resolvable from a fresh clone).
# Single-sourced here as module constants so AnalyzeConfig's own fields
# reference the same values instead of duplicating the literals (previously
# both dedup_parallel_arcs's keyword defaults and AnalyzeConfig carried
# independent copies of 0.75/0.15, which could silently drift apart).
ARC_DEDUP_OVERLAP_FRAC = 0.75
ARC_DEDUP_TRAJ_TOL = 0.15


def dedup_parallel_arcs(
    arcs: list[Arc], *, overlap_frac: float = ARC_DEDUP_OVERLAP_FRAC,
    traj_tol: float = ARC_DEDUP_TRAJ_TOL, samples: int = 5,
) -> list[Arc]:
    """Collapse arcs that trace the same physical flight (Plan 5 task 2b).

    Field motivation: per-frame duplicate-box clustering (detect/cluster.py)
    turns duplicate/crossing discrimination into a box-confidence question --
    but on REAL footage two crossing balls virtually always differ in
    confidence, so the box-level strict-lower-confidence guard that protects
    genuine crossings cannot also collapse duplicate storms without a large
    enough merge_dist to start swallowing real crossings too (measured:
    ss531_id_005 40->16 catches against an oracle of 63, see
    docs/superpowers/plans/2026-08-03-meschke-validation-findings.md).
    Box-level geometry in one frame simply lacks the discriminating
    information; full arc TRAJECTORIES have it -- a duplicate-box storm
    mints two (or more) arcs that trace nearly the same parabola over their
    ENTIRE shared time window (one physical flight seen through jittered
    duplicate boxes), while two crossing balls trace different trajectories
    that merely intersect briefly (near-zero separation at one instant, but
    large separation everywhere else in the window). Run AFTER extract_arcs
    and BEFORE the event-derivation tail (analyze.py's ``_events_from_arcs``)
    so both offline and realtime callers share the fix (same integration
    point clustering already uses).

    Only pairs whose temporal overlap exceeds ``overlap_frac`` of the
    SHORTER arc's own span are even compared -- a real crossing's brief
    intersection is not "overlap" in this sense; the two arcs' full spans
    need not coincide at all for two crossing balls. For a qualifying pair,
    both trajectories are sampled at ``samples`` evenly spaced times across
    the OVERLAP window (``Arc.x_at``/``Arc.y_at``); if the mean per-sample
    ``|dx| + |dy|`` is under ``traj_tol``, they are judged the same flight
    and only the better-witnessed one survives (higher ``n_points``, ties
    broken by lower ``rmse`` -- mirrors detect/cluster.py's own tie-break
    preference for "more real evidence wins").

    Applied greedily best-witnessed-first, mirroring detect/cluster.py's
    confidence-first greedy absorption: arcs are visited in
    ``(n_points desc, rmse asc)`` order, and a visited, still-surviving arc
    drops every later (worse-witnessed), still-surviving arc that duplicates
    it. A duplicate CLUSTER of 3+ parallel arcs therefore collapses directly
    to its one best-witnessed member (every member is compared against the
    best when the best is visited), not just pairwise-adjacent ones.
    """
    if not 0.0 <= overlap_frac <= 1.0:
        raise ValueError(f"overlap_frac must be in [0, 1] (got {overlap_frac})")
    if traj_tol < 0.0:
        raise ValueError(f"traj_tol must be >= 0 (got {traj_tol})")
    if len(arcs) < 2:
        return list(arcs)

    order = sorted(range(len(arcs)), key=lambda i: (-arcs[i].n_points, arcs[i].rmse))
    dropped = [False] * len(arcs)
    for oi in range(len(order)):
        i = order[oi]
        if dropped[i]:
            continue
        a = arcs[i]
        for oj in range(oi + 1, len(order)):
            j = order[oj]
            if dropped[j]:
                continue
            if _same_flight(a, arcs[j], overlap_frac, traj_tol, samples):
                dropped[j] = True
    return [a for a, d in zip(arcs, dropped) if not d]


def _same_flight(
    a: Arc, b: Arc, overlap_frac: float, traj_tol: float, samples: int,
) -> bool:
    """True if `a` and `b` are judged the same physical flight (the
    per-pair predicate behind `dedup_parallel_arcs` -- see its own
    docstring for the acceptance rule this implements).

    Known limitation (columns patterns, not the async-siteswap set this
    stage was tuned/gated against): the premise above -- that two arcs
    which agree closely across their ENTIRE shared window must be one
    flight seen twice, because two genuinely different balls only ever
    agree briefly -- fails for synchronous columns. Two real balls thrown
    in parallel columns, less than `traj_tol` apart and moving in lockstep
    (~100% temporal overlap, near-identical trajectories throughout, not
    just a brief crossing), would read exactly like a duplicate-box storm
    to this check and incorrectly collapse to one arc. Out of scope here:
    every video in the Meschke oracle validation set (the evidence base
    for `dedup_parallel_arcs`'s `overlap_frac`/`traj_tol` defaults) is
    async, so this gap is undiagnosed by that evidence, not closed by this
    predicate.
    """
    ov_start = max(a.t_start, b.t_start)
    ov_end = min(a.t_end, b.t_end)
    ov_dur = ov_end - ov_start
    if ov_dur <= 0.0:
        return False
    shorter_span = min(a.duration(), b.duration())
    if shorter_span <= 0.0 or ov_dur <= overlap_frac * shorter_span:
        return False
    if samples <= 1:
        ts = [ov_start + 0.5 * ov_dur]
    else:
        ts = [ov_start + k * ov_dur / (samples - 1) for k in range(samples)]
    total = sum(abs(a.x_at(t) - b.x_at(t)) + abs(a.y_at(t) - b.y_at(t)) for t in ts)
    return (total / len(ts)) < traj_tol


def assign_detections(
    dets: list[Detection],
    arcs: list[Arc],
    *,
    resid_tol: float = 0.02,
    time_margin: float = 0.02,
) -> list[int]:
    """Best-fitting arc id per detection (input order), or -1 if none fits.

    Same acceptance rule as EM assignment (max of y/x residuals under
    2*resid_tol), so 'assigned' means 'would have survived extraction'.
    This is the auto-labeler's precision gate (spec §5).
    """
    out: list[int] = []
    for d in dets:
        best_id, best_res = -1, 2.0 * resid_tol
        for arc in arcs:
            if not (arc.t_start - time_margin <= d.t <= arc.t_end + time_margin):
                continue
            dt = d.t - arc.t_start
            ry = abs(arc.ay * dt * dt + arc.by * dt + arc.cy - d.y)
            rx = abs(arc.bx * dt + arc.cx - d.x)
            res = max(ry, rx)
            if res < best_res:
                best_id, best_res = arc.id, res
        out.append(best_id)
    return out
