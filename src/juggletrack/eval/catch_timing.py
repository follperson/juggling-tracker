"""Time-only scoring restricted to exhaustively reviewed contact coverage."""
from __future__ import annotations

import heapq
from bisect import bisect_left

from juggletrack.eval.labels import ReviewedCatchWindow, validate_catch_coverage
from juggletrack.events.catches import derive_events
from juggletrack.types import CatchEvent, SessionResult


def pool_catch_timing(reports: list[dict]) -> dict:
    """Pool counts before computing rates; undefined denominators stay unknown."""
    tp = sum(r["tp"] for r in reports)
    fp = sum(r["fp"] for r in reports)
    fn = sum(r["fn"] for r in reports)
    return {
        "reviewed_windows": sum(r["reviewed_windows"] for r in reports),
        "reviewed_seconds": sum(r["reviewed_seconds"] for r in reports),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }


def score_catch_windows(
    predictions: list[CatchEvent], windows: list[ReviewedCatchWindow],
) -> dict:
    """Maximum-cardinality point/interval matching, without a timing tolerance.

    Process predictions chronologically and match the eligible bracket that
    expires first. Swapping that choice with any later-expiring bracket cannot
    reduce future matches, so this interval-specific greedy matching is maximum.
    Original bracket indices disambiguate simultaneous or identical contacts.
    """
    validate_catch_coverage(windows)
    ordered = sorted(predictions, key=lambda event: (event.t, event.arc_id))
    times = [event.t for event in ordered]
    results = []
    for window in windows:
        pending = sorted(enumerate(window.catches), key=lambda pair: pair[1].earliest_t)
        active: list[tuple[float, int]] = []
        cursor = 0
        matched: set[int] = set()
        matches, extras = [], []
        start = bisect_left(times, window.start_t)
        end = bisect_left(times, window.end_t)
        for event in ordered[start:end]:
            while cursor < len(pending) and pending[cursor][1].earliest_t <= event.t:
                index, bracket = pending[cursor]
                heapq.heappush(active, (bracket.latest_t, index))
                cursor += 1
            while active and active[0][0] < event.t:
                heapq.heappop(active)
            if not active:
                extras.append({"t": event.t, "arc_id": event.arc_id})
                continue
            _, index = heapq.heappop(active)
            matched.add(index)
            matches.append({"reference_index": index, **window.catches[index].model_dump(),
                            "predicted_t": event.t, "arc_id": event.arc_id})
        misses = [{"reference_index": i, **bracket.model_dump()}
                  for i, bracket in enumerate(window.catches) if i not in matched]
        results.append({"start_t": window.start_t, "end_t": window.end_t,
                        "matches": matches, "extras": extras, "misses": misses})
    counts = {
        "reviewed_windows": len(windows),
        "reviewed_seconds": sum(w.end_t - w.start_t for w in windows),
        "tp": sum(len(w["matches"]) for w in results),
        "fp": sum(len(w["extras"]) for w in results),
        "fn": sum(len(w["misses"]) for w in results),
    }
    return {**pool_catch_timing([counts]), "windows": results}


def evaluate_catch_timing(session: SessionResult, windows: list[ReviewedCatchWindow]) -> dict:
    """Rebuild shipped catches only from arcs belonging to surviving runs."""
    accepted = {arc_id for run in session.runs for arc_id in run.arc_ids}
    _, catches = derive_events(session.arcs, session.hand_line_y)
    counted = [c for c in catches if c.arc_id in accepted]
    if len(counted) != sum(run.catches for run in session.runs):
        raise ValueError("derived catch events disagree with the session's run counts")
    return score_catch_windows(counted, windows)
