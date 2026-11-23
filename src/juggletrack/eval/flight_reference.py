"""Reference flights read off labelled ball tracks, and scores against them.

``oracle_events`` builds its reference with the arc extractor it is meant to
judge, so extractor losses cancel out of every comparison. This module finds
flights with no parabola fit and no code from ``juggletrack.arcs``,
``juggletrack.events`` or ``juggletrack.analyze``.

It reads one ball's labelled track at a time. A flight is a rise from a low
point of the track to an apex and back down, where both swings exceed a
threshold. The threshold is a fraction of the clip's pattern height, so the
same track gives the same flights at any zoom and any playback speed. Running
several thresholds gives the trust signal. A count that moves with the
threshold means the clip mixes real flights with low passes or hand movement
that this method cannot tell apart, and the clip is flagged.

``score_catches`` matches a pipeline's catches to the reference landings.
``score_arcs`` counts the flights that an extracted arc explains.

Coordinates are normalized with y growing downward, so an apex is a minimum
of y and "below" means a larger y.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel
from scipy.optimize import linear_sum_assignment

from juggletrack.types import Arc, CatchEvent, Detection

# Smallest swing that counts as a flight, as fractions of the pattern height.
# The count must hold across all of them for a clip to be trusted. The band
# runs from just above hand wobble to just below the lowest throw of a
# single-height pattern.
SWEEP = (0.05, 0.10, 0.15, 0.20, 0.30)
# The fraction whose flights are the reference.
REFERENCE_SWING = 0.10
# Largest change in the flight count across SWEEP, relative to the reference
# count, that still counts as stable.
STABLE_WITHIN = 0.03
# Hand level sits this fraction of a flight's rise above its low point.
HAND_BAND = 0.25
# A flight with a hole in its samples longer than this fraction of its own
# duration was not seen whole.
MAX_GAP = 0.25
# The floor line sits this many typical rises below the clip's hand level. A
# ball that passes hand level and reaches the floor line within the time it
# took to fall from its apex was not caught. A caught ball that the juggler
# lowers afterwards takes longer.
DROP_BELOW = 0.5
# A catch can match a flight when it comes within this fraction of the
# flight's own duration of the landing. A pipeline may put its hand line
# anywhere between the low point of the track and the middle of the rise,
# which moves its catch times by up to about 0.15 of a flight.
MATCH_TOLERANCE = 0.2
# The catch must also be within this fraction of the pattern width of where
# the ball landed. That leaves room for the ball's sideways travel during the
# time tolerance and still excludes the other hand.
MATCH_WIDTH = 0.3
# The width bound never drops below the 10 px of a 480 px frame that the
# detector benchmark allows a ball centre, so a pattern thrown straight up
# and down can still match.
MATCH_WIDTH_FLOOR = 0.02
# An arc explains a flight when it passes within this fraction of the pattern
# height of the apex.
APEX_NEAR = 0.1


class Flight(BaseModel):
    """One ball's trip from hand level to an apex and back.

    ``launch_t`` is None when the track begins after the ball left hand
    level, and ``landing_t`` and ``landing_x`` are None when the track ends
    before it came back. The landing position is (``landing_x``, ``hand_y``).
    ``whole`` says the flight was seen from launch to landing with no long
    hole in its samples. ``caught`` is False when the ball went on down to
    the floor.
    """

    ball: int
    launch_t: float | None
    apex_t: float
    apex_x: float
    apex_y: float
    landing_t: float | None
    landing_x: float | None
    hand_y: float
    whole: bool
    caught: bool

    @property
    def counts(self) -> bool:
        """True for a flight seen whole and caught, which a correct counter reports once."""
        return self.whole and self.caught


class FlightReference(BaseModel):
    """Every flight found in one clip at the reference threshold.

    ``sweep`` lists the count of flights at each threshold fraction.
    ``trusted`` is False when those counts disagree by more than the stable
    margin. ``span`` is the first and last labelled time.
    """

    flights: list[Flight]
    pattern_height: float
    pattern_width: float
    sweep: list[tuple[float, int]]
    trusted: bool
    span: tuple[float, float]

    @property
    def counted(self) -> list[Flight]:
        return [f for f in self.flights if f.counts]

    @property
    def median_flight_s(self) -> float | None:
        """Median launch-to-landing time of the counted flights."""
        counted = self.counted
        if not counted:
            return None
        return float(np.median([f.landing_t - f.launch_t for f in counted]))


def _turns(y: np.ndarray, min_swing: float) -> list[tuple[int, bool]]:
    """Turning points of one track as (sample index, is_apex), alternating.

    A point is a turn once the track has moved ``min_swing`` away from it, so
    wobble smaller than that leaves none. The last entry is the extreme the
    track was heading away from when it ended. A track that never swings by
    ``min_swing`` has no turns.
    """
    turns: list[tuple[int, bool]] = []
    top = low = 0  # highest and lowest sample since the last turn
    rising: bool | None = None
    for i in range(1, len(y)):
        if y[i] < y[top]:
            top = i
        if y[i] > y[low]:
            low = i
        if rising is not False and y[i] - y[top] >= min_swing:
            turns.append((top, True))
            rising, low = False, i
        elif rising is not True and y[low] - y[i] >= min_swing:
            turns.append((low, False))
            rising, top = True, i
    if rising is True:
        turns.append((top, True))
    elif rising is False:
        turns.append((low, False))
    return turns


def _ball_flights(
    ball: int, t: np.ndarray, x: np.ndarray, y: np.ndarray,
    turns: list[tuple[int, bool]], floor_y: float,
) -> list[Flight]:
    flights: list[Flight] = []
    for k, (apex, is_apex) in enumerate(turns):
        if not is_apex:
            continue
        before = turns[k - 1][0] if k > 0 else None
        after = turns[k + 1][0] if k + 1 < len(turns) else None
        # A low point counts as the hand only if the ball was seen to fall
        # into it and climb out again. The first and last turns of a track
        # may be a ball that was already, or still, in the air.
        seen = [
            y[i] for i, has_far_side in ((before, k > 1), (after, k + 2 < len(turns)))
            if i is not None and has_far_side
        ]
        if seen:
            # Of two real low points take the higher, so a ball that fell on
            # to the floor still lands where it passed the hand.
            low_y = min(seen)
        else:
            low_y = max(y[i] for i in (before, after) if i is not None)
        hand_y = low_y - HAND_BAND * (low_y - y[apex])

        first = before if before is not None else 0
        last = after if after is not None else len(y) - 1
        # Whole-pixel labels put a run of equal samples at the top of a
        # flight. Take the middle one.
        tied = first + np.flatnonzero(y[first:last + 1] == y[apex])
        apex = int(tied[len(tied) // 2])

        launch_t = landing_t = landing_x = None
        caught = True
        in_hand = np.flatnonzero(y[first:apex] >= hand_y)
        if in_hand.size:
            i = first + int(in_hand[-1])
            f = (y[i] - hand_y) / (y[i] - y[i + 1])
            launch_t = float(t[i] + f * (t[i + 1] - t[i]))
        in_hand = np.flatnonzero(y[apex + 1:last + 1] >= hand_y)
        if in_hand.size:
            j = apex + 1 + int(in_hand[0])
            f = (hand_y - y[j - 1]) / (y[j] - y[j - 1])
            landing_t = float(t[j - 1] + f * (t[j] - t[j - 1]))
            landing_x = float(x[j - 1] + f * (x[j] - x[j - 1]))
            one_fall_later = np.searchsorted(t, 2 * landing_t - t[apex], side="right")
            caught = not (y[j:one_fall_later] > floor_y).any()

        whole = launch_t is not None and landing_t is not None
        if whole:
            biggest_gap = float(np.diff(t[i:j + 1]).max())
            whole = biggest_gap <= MAX_GAP * (landing_t - launch_t)
        flights.append(Flight(
            ball=ball, launch_t=launch_t,
            apex_t=float(t[apex]), apex_x=float(x[apex]), apex_y=float(y[apex]),
            landing_t=landing_t, landing_x=landing_x, hand_y=float(hand_y),
            whole=whole, caught=caught,
        ))
    return flights


def _find_flights(
    trajectories: Sequence[Sequence[Detection]], min_swing: float,
) -> list[Flight]:
    tracks = []
    for ball, track in enumerate(trajectories):
        track = sorted(track, key=lambda d: d.t)
        t, x, y = (np.array([getattr(d, k) for d in track]) for k in "txy")
        tracks.append((ball, t, x, y, _turns(y, min_swing)))

    # Where the hands are and how high a typical throw goes, from the turns
    # that have a turn on either side. The medians ignore a few drops.
    lows = [y[i] for _, _, _, y, turns in tracks for i, is_apex in turns[1:-1] if not is_apex]
    tops = [y[i] for _, _, _, y, turns in tracks for i, is_apex in turns[1:-1] if is_apex]
    floor_y = float("inf")
    if lows and tops:
        hands = float(np.median(lows))
        floor_y = hands + DROP_BELOW * (hands - float(np.median(tops)))

    flights = [
        f for ball, t, x, y, turns in tracks
        for f in _ball_flights(ball, t, x, y, turns, floor_y)
    ]
    return sorted(flights, key=lambda f: (f.apex_t, f.ball))


def flight_reference(trajectories: Sequence[Sequence[Detection]]) -> FlightReference:
    """Find the flights in per-ball labelled tracks, one list per ball.

    The input is what ``load_meschke_trajectories`` returns. The pattern
    height and width are the 1st to 99th percentile ranges of y and x over
    every ball, which a held ball or a few stray labels do not move.
    """
    labels = [d for track in trajectories for d in track]
    if not labels:
        return FlightReference(
            flights=[], pattern_height=0.0, pattern_width=0.0, sweep=[], trusted=False,
            span=(0.0, 0.0),
        )
    high, low = np.percentile([d.y for d in labels], [1, 99])
    left, right = np.percentile([d.x for d in labels], [1, 99])
    height = float(low - high)

    found = {
        frac: _find_flights(trajectories, frac * height) if height > 0 else []
        for frac in SWEEP
    }
    sweep = [(frac, sum(f.counts for f in flights)) for frac, flights in found.items()]
    counts = [n for _, n in sweep]
    n = dict(sweep)[REFERENCE_SWING]
    return FlightReference(
        flights=found[REFERENCE_SWING], pattern_height=height,
        pattern_width=float(right - left), sweep=sweep,
        trusted=n > 0 and max(counts) - min(counts) <= STABLE_WITHIN * n,
        span=(min(d.t for d in labels), max(d.t for d in labels)),
    )


Verdict = Literal["matched", "duplicate", "unmatched", "ignored"]


class CatchScore(BaseModel):
    """How one clip's catches line up with its reference flights.

    ``flights`` counts the reference flights seen whole and caught, and
    ``catches`` counts the catches given. ``matched`` is how many of those
    flights got a catch. ``duplicates`` are catches on a flight that already
    has one, and ``unmatched`` are catches near no flight. ``ignored``
    catches fall outside the labelled span, or are the one catch on a flight
    that was not seen whole, and count neither for nor against. Such a flight
    still lands once, so a second catch on it is a duplicate. ``count_error``
    is the catches that were not ignored, minus ``flights``. ``offset_s`` is
    the median of catch time minus landing time over the matched pairs,
    positive when the catches come late, and None when nothing matched. The
    time tolerance is loose, so a shift of the catch clock shows here long
    before it turns matched catches into unmatched ones.

    ``verdicts`` has one entry per catch, in the order given. ``missed``
    holds the indices into ``FlightReference.flights`` of the counted
    flights with no catch.
    """

    flights: int
    catches: int
    matched: int
    duplicates: int
    unmatched: int
    ignored: int
    count_error: int
    offset_s: float | None
    verdicts: list[Verdict]
    missed: list[int]


def score_catches(reference: FlightReference, catches: Sequence[CatchEvent]) -> CatchScore:
    """Match catches to reference landings, one catch per flight.

    A catch can match a flight when it comes within ``MATCH_TOLERANCE`` of
    the flight's own duration of the landing, and within ``MATCH_WIDTH`` of
    the pattern width of the landing position. Both bounds follow the
    footage, so slow motion and real time are held to the same standard.
    Among all one-to-one pairings the scorer takes one that matches the most
    catches, and of those the one with the smallest total time error.
    """
    flights = reference.flights
    start, end = reference.span
    scored = [i for i, c in enumerate(catches) if start <= c.t <= end]
    # A dropped ball has no catch to match, so a catch on one stays unmatched.
    landed = [k for k, f in enumerate(flights) if f.caught and f.landing_t is not None]

    verdicts: list[Verdict] = ["ignored"] * len(catches)
    for i in scored:
        verdicts[i] = "unmatched"
    caught: set[int] = set()
    offsets: list[float] = []
    if scored and landed:
        catch_t, catch_x = (np.array([getattr(catches[i], k) for i in scored]) for k in "tx")
        landing_t, landing_x, apex_t = (
            np.array([getattr(flights[k], name) for k in landed])
            for name in ("landing_t", "landing_x", "apex_t")
        )
        # Time error as a fraction of what each flight allows. A flight lasts
        # twice its fall, which holds even when its launch was not seen.
        error = np.abs(catch_t[:, None] - landing_t) / (
            MATCH_TOLERANCE * 2 * (landing_t - apex_t)
        )
        reach_x = max(MATCH_WIDTH * reference.pattern_width, MATCH_WIDTH_FLOOR)
        near = (error <= 1) & (np.abs(catch_x[:, None] - landing_x) <= reach_x)
        # One pair out of reach costs more than every pair in reach together,
        # so the cheapest assignment also matches the most catches.
        out_of_reach = min(error.shape) + 1.0
        rows, cols = linear_sum_assignment(np.where(near, error, out_of_reach))
        paired = {int(r): landed[c] for r, c in zip(rows, cols) if near[r, c]}
        for row, i in enumerate(scored):
            k = paired.get(row)
            if k is None:
                verdicts[i] = "duplicate" if near[row].any() else "unmatched"
            elif flights[k].counts:
                verdicts[i] = "matched"
                caught.add(k)
                offsets.append(catches[i].t - flights[k].landing_t)
            else:
                verdicts[i] = "ignored"

    counted = [k for k, f in enumerate(flights) if f.counts]
    matched, duplicates, unmatched = (
        verdicts.count(v) for v in ("matched", "duplicate", "unmatched")
    )
    return CatchScore(
        flights=len(counted), catches=len(catches),
        matched=matched, duplicates=duplicates, unmatched=unmatched,
        ignored=verdicts.count("ignored"),
        count_error=matched + duplicates + unmatched - len(counted),
        offset_s=float(np.median(offsets)) if offsets else None,
        verdicts=verdicts, missed=[k for k in counted if k not in caught],
    )


class ArcScore(BaseModel):
    """How many reference flights the extracted arcs account for.

    ``flights`` counts the reference flights seen whole and caught.
    ``explained`` counts those with at least one arc in flight at the apex
    time and near the apex. ``repeated`` counts those with more than one
    such arc. ``unexplained`` holds the indices into
    ``FlightReference.flights`` of the rest.
    """

    flights: int
    arcs: int
    explained: int
    repeated: int
    unexplained: list[int]


def score_arcs(reference: FlightReference, arcs: Sequence[Arc]) -> ArcScore:
    """Count the reference flights that at least one extracted arc explains.

    This view judges the extractor alone. A flight can be explained and
    still get no catch, when the loss is in event derivation.
    """
    near = APEX_NEAR * reference.pattern_height
    counted = [k for k, f in enumerate(reference.flights) if f.counts]
    arcs_at: dict[int, int] = {}
    for k in counted:
        f = reference.flights[k]
        arcs_at[k] = sum(
            1 for a in arcs
            if a.t_start <= f.apex_t <= a.t_end
            and np.hypot(a.x_at(f.apex_t) - f.apex_x, a.y_at(f.apex_t) - f.apex_y) <= near
        )
    return ArcScore(
        flights=len(counted), arcs=len(arcs),
        explained=sum(n >= 1 for n in arcs_at.values()),
        repeated=sum(n > 1 for n in arcs_at.values()),
        unexplained=[k for k, n in arcs_at.items() if n == 0],
    )
