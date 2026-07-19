"""Debug overlay: draw what the event core concluded on top of the source video."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2

from juggletrack.arcs.extract import assign_detections
from juggletrack.events.catches import derive_events
from juggletrack.pipeline.draw import (
    draw_arc_tails,
    draw_detections,
    draw_drop_marker,
    draw_hand_line,
    draw_hud,
)
from juggletrack.types import Detection, SessionResult
from juggletrack.video.reader import VideoReader


def render_overlay(
    video_path: str | Path,
    session: SessionResult,
    out_path: str | Path,
    *,
    detections: list[Detection] | None = None,
    tail_s: float = 0.4,
    dots: str = "verified",
) -> None:
    """Render the debug overlay.

    ``dots`` controls which raw detections get drawn as small circles:
    "verified" (default) draws only detections the arc gate assigned to a
    fitted arc -- junk detections (e.g. eye pupils) are rejected from counts
    by that same gate, so they're hidden here too, by default, to avoid
    alarming the user with dots that don't affect the result. "all" draws
    every detection (the old, unfiltered behavior). "none" draws no dots.
    """
    if dots not in ("verified", "all", "none"):
        raise ValueError(f"dots must be 'verified', 'all', or 'none' (got {dots!r})")

    dets = list(detections or [])
    if dots == "none":
        dets = []
    elif dots == "verified" and dets:
        assigned = assign_detections(dets, session.arcs)
        dets = [d for d, arc_id in zip(dets, assigned) if arc_id != -1]

    dets_by_frame: dict[int, list[Detection]] = defaultdict(list)
    for d in dets:
        dets_by_frame[d.frame_idx].append(d)

    _, catches = derive_events(session.arcs, session.hand_line_y)
    catch_times_by_run = [
        sorted(c.t for c in catches if c.arc_id in set(run.arc_ids))
        for run in session.runs
    ]
    arcs_by_id = {a.id: a for a in session.arcs}

    with VideoReader(video_path) as reader:
        w, h = reader.info.width, reader.info.height
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), reader.info.fps, (w, h)
        )
        if not writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter failed to open: {out_path}")

        for idx, t, frame in reader.frames():
            draw_hand_line(frame, session.hand_line_y)
            draw_detections(frame, dets_by_frame.get(idx, []))
            draw_arc_tails(frame, session.arcs, t, tail_s=tail_s)

            active = next(
                (i for i, r in enumerate(session.runs) if r.start_t - 0.5 <= t <= r.end_t + 0.5),
                None,
            )
            drops_so_far = sum(1 for d in session.drops if d.t <= t)
            if active is not None:
                caught = sum(1 for ct in catch_times_by_run[active] if ct <= t)
                hud = f"run {active + 1}/{len(session.runs)}  catches {caught}  drops {drops_so_far}"
            else:
                hud = f"runs {len(session.runs)}  drops {drops_so_far}"
            draw_hud(frame, hud)

            for drop in session.drops:
                if drop.t <= t <= drop.t + 0.5:
                    # DropEvent carries no y field (see types.py); recover the
                    # drop's vertical position from its source arc at drop.t,
                    # falling back to the hand line if the arc is unknown.
                    drop_arc = arcs_by_id.get(drop.arc_id)
                    y_frac = drop_arc.y_at(drop.t) if drop_arc is not None else session.hand_line_y
                    draw_drop_marker(frame, drop.x, y_frac)

            writer.write(frame)
        writer.release()
