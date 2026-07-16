"""Hand-line estimation from arc statistics (pose-free fallback, spec §4)."""
from __future__ import annotations

import numpy as np

from juggletrack.types import Arc


def estimate_hand_line(arcs: list[Arc]) -> float:
    """Robust median of arc endpoint heights.

    Throws and catches happen near hand height, so arc endpoints cluster there.
    A dropped ball contributes one floor-level endpoint; the median absorbs it.
    """
    if not arcs:
        return 0.0
    ys = [a.y_at(a.t_start) for a in arcs] + [a.y_at(a.t_end) for a in arcs]
    return float(np.median(ys))
