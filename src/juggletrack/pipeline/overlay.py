"""Debug overlay: draw what the event core concluded on top of the source video."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from juggletrack.events.catches import derive_events
from juggletrack.types import Detection, SessionResult
from juggletrack.video.reader import VideoReader

_GREEN = (0, 200, 0)
_YELLOW = (0, 220, 220)
_RED = (0, 0, 255)
_WHITE = (240, 240, 240)


def render_overlay(
    video_path: str | Path,
    session: SessionResult,
    out_path: str | Path,
    *,
    detections: list[Detection] | None = None,
    tail_s: float = 0.4,
) -> None:
    dets_by_frame: dict[int, list[Detection]] = defaultdict(list)
    for d in detections or []:
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

        hand_y_px = int(session.hand_line_y * h)
        for idx, t, frame in reader.frames():
            cv2.line(frame, (0, hand_y_px), (w, hand_y_px), _YELLOW, 1)

            for d in dets_by_frame.get(idx, []):
                cv2.circle(frame, (int(d.x * w), int(d.y * h)), 4, _WHITE, 1)

            for arc in session.arcs:
                if not (arc.t_start <= t <= arc.t_end + 0.1):
                    continue
                t0 = max(arc.t_start, t - tail_s)
                ts = np.arange(t0, min(t, arc.t_end) + 1e-9, 1 / 60)
                pts = np.array(
                    [[int(arc.x_at(tt) * w), int(arc.y_at(tt) * h)] for tt in ts]
                )
                if len(pts) >= 2:
                    cv2.polylines(frame, [pts], False, _GREEN, 2)

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
            cv2.putText(frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _WHITE, 2)

            for drop in session.drops:
                if drop.t <= t <= drop.t + 0.5:
                    # DropEvent carries no y field (see types.py); recover the
                    # drop's vertical position from its source arc at drop.t,
                    # falling back to the hand line if the arc is unknown.
                    drop_arc = arcs_by_id.get(drop.arc_id)
                    y_frac = drop_arc.y_at(drop.t) if drop_arc is not None else session.hand_line_y
                    x, y = int(drop.x * w), int(y_frac * h)
                    cv2.drawMarker(frame, (x, y), _RED, cv2.MARKER_TILTED_CROSS, 24, 3)

            writer.write(frame)
        writer.release()
