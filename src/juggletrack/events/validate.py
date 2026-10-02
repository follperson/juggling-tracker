"""Run-level structural validators (spec §4): junk-cohort defenses that
inspect a whole run's arcs, not any single arc in isolation.
"""
from __future__ import annotations

import numpy as np

from juggletrack.types import Arc

# Below this, an arc's net x displacement is noise, not signal -- protects
# near-vertical throws (e.g. a siteswap-4 "columns" pattern, or multiplex
# throws) from having their sign flip on measurement noise alone.
DX_MIN = 0.03
DOM_T = 0.99   # dom2 >= this: every meaningful arc drifts the same direction
MONO_T = 0.9   # mono >= this: throw x-origins march monotonically with index


def is_drift_cohort(arcs: list[Arc]) -> bool:
    """True iff ``arcs`` (one run's worth) look like a single self-consistent
    slow-drift junk cohort rather than real juggling.

    Per arc i (sorted by ``t_start``): ``dx_i = bx_i * duration_i`` (net x
    displacement over the arc's flight). ``dx_i`` is *meaningful* iff
    ``abs(dx_i) >= DX_MIN``.

    - ``dom2`` = ``abs(mean(sign(dx_i)))`` over meaningful arcs only (1.0 =
      every meaningful arc drifts the same direction; 0.0 if none are
      meaningful).
    - ``mono`` = ``abs(pearson(cx_i, i))`` over ALL arcs in order (1.0 = the
      arcs' starting x-position marches monotonically across the frame with
      arc index -- the signature of one cohort drifting, not a cascade
      alternating hands). Needs >= 3 arcs and nonzero variance in cx;
      otherwise 0.0 (can't be judged monotonic).
    - ``sweeps`` = every arc after the first has an x-range
      ``[min(cx_i, cx_i + dx_i), max(...)]`` disjoint from the hull of all
      earlier arcs' x-ranges (touching counts as overlap).

    Reject iff ``dom2 >= DOM_T AND mono >= MONO_T AND sweeps``.

    No alternation term is needed. ``dom2 >= DOM_T`` allows one minority
    sign only per 200 meaningful arcs, and a sweep of 200 meaningful arcs
    spans at least ``200 * DX_MIN`` = 6 frame widths. ``_sweeps`` reads
    fitted endpoints (``cx`` and ``cx + dx``), which can sit slightly outside
    [0, 1], but nowhere near that far. So inside the frame, every meaningful
    arc of a rejected run shares a sign. The 200 and 6 follow from DOM_T and
    DX_MIN; test_a_minority_sign_cannot_fit_in_one_frame_width guards them.
    """
    ordered = sorted(arcs, key=lambda a: a.t_start)
    dx = np.array([a.bx * a.duration() for a in ordered])
    meaningful = np.abs(dx) >= DX_MIN

    mdx = dx[meaningful]
    dom2 = float(abs(np.mean(np.sign(mdx)))) if len(mdx) else 0.0

    cx = np.array([a.cx for a in ordered])
    mono = (float(abs(np.corrcoef(cx, np.arange(len(cx)))[0, 1]))
            if len(cx) >= 3 and np.std(cx) > 1e-9 else 0.0)

    return dom2 >= DOM_T and mono >= MONO_T and _sweeps(cx, dx)


def _sweeps(cx: np.ndarray, dx: np.ndarray) -> bool:
    lo, hi = np.minimum(cx, cx + dx), np.maximum(cx, cx + dx)
    for i in range(1, len(lo)):
        if not (hi[i] < lo[:i].min() or lo[i] > hi[:i].max()):
            return False
    return True
