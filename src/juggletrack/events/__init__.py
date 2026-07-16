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
