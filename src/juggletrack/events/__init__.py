"""Shared constants for the events package.

Both the catch gate (``events/catches.py``) and the drop-candidacy gate
(``events/drops.py``) test an arc's terminal y-position against the hand
line using the *same* margin. They must stay exact complements of each
other (catch: ``<= hand_line + FLOOR_MARGIN``; drop candidacy: ``>
hand_line + FLOOR_MARGIN``) so that every arc falls into exactly one
bucket and no arc lands in a dead band that yields neither a catch nor a
drop candidate.
"""
from __future__ import annotations

FLOOR_MARGIN = 0.10

# Seconds of dropout/occlusion tolerance (~4-5 frames at 30fps) for how far
# a catch's analytically-projected hand-line crossing may extend past an
# arc's last real detection before it stops counting as witnessed.
CATCH_EXTRAPOLATION_MARGIN = 0.15
# The same tolerance as a share of the flight, hand line to hand line. An arc
# that ends a fixed height above the hand line reaches it later the slower
# the footage plays, so a margin in seconds alone rejects real catches in
# slow motion. At real-time speed a flight lasts about 0.6 s and the seconds
# above still decide.
CATCH_EXTRAPOLATION_FRACTION = 0.15


def catch_witness_margin(flight_s: float, floor: float = CATCH_EXTRAPOLATION_MARGIN) -> float:
    """Seconds a hand-line crossing may lie past the last real detection of a
    flight lasting ``flight_s`` and still count as witnessed, never less than
    ``floor``."""
    return max(floor, CATCH_EXTRAPOLATION_FRACTION * flight_s)
