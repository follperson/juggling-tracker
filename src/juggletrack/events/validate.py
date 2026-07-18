"""Run-level structural validators (spec §4): junk-cohort defenses that
inspect a whole run's arcs, not any single arc in isolation.

Turn-4 cascade-structure-gate bake-off: this drift-cohort clause (clause A)
was measured discriminating and safe across the full battery -- the pinned
junk-cohort fixture (tests/test_extract.py::test_slow_drift_junk_cohort_known_gap)
is the only run in the battery with dom2=1.0 AND mono=1.0 simultaneously.
The bake-off's other clause ("freg", rejecting runs with irregular throw
intervals) was tuned against a since-corrected target (the old merged-cloud
Meschke oracle undercounted ss3_id_016 at 23 catches; the per-ball oracle
now reports 52 -- see data.meschke_import.oracle_events) and is
intentionally NOT shipped here.
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
ALT_T = 0.2    # alt2 <= this (or undefined): essentially never alternates


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
    - ``alt2`` = fraction of ADJACENT arc pairs (in the full sorted
      sequence, both meaningful) whose dx sign flips; ``None`` (undefined)
      when there are no such pairs.

    Reject iff ``dom2 >= DOM_T AND mono >= MONO_T AND (alt2 is None or
    alt2 <= ALT_T)``: unidirectional, monotonically marching, and never
    alternating side to side -- a real cascade alternates hands and so
    flips dx sign often; a self-consistent slow-drift cohort, by
    construction, does not.
    """
    ordered = sorted(arcs, key=lambda a: a.t_start)
    dx = np.array([a.bx * a.duration() for a in ordered])
    meaningful = np.abs(dx) >= DX_MIN

    pairs = [(dx[i], dx[i + 1]) for i in range(len(dx) - 1)
             if meaningful[i] and meaningful[i + 1]]
    alt2 = (sum(1 for u, v in pairs if u * v < 0) / len(pairs)) if pairs else None

    mdx = dx[meaningful]
    dom2 = float(abs(np.mean(np.sign(mdx)))) if len(mdx) else 0.0

    cx = np.array([a.cx for a in ordered])
    mono = (float(abs(np.corrcoef(cx, np.arange(len(cx)))[0, 1]))
            if len(cx) >= 3 and np.std(cx) > 1e-9 else 0.0)

    return dom2 >= DOM_T and mono >= MONO_T and (alt2 is None or alt2 <= ALT_T)
