"""Live capture loop: webcam or file -> detector -> RealtimeAnalyzer -> HUD.

cv2 is allowed at module scope here (unlike arcs/extract.py, realtime.py):
this is the capture/display layer, mirroring video/reader.py's carve-out.
"""
from __future__ import annotations

import time
from collections.abc import Callable

import cv2
import numpy as np

from juggletrack.arcs.extract import assign_detections
from juggletrack.detect import BallDetector
from juggletrack.pipeline.draw import (
    draw_arc_tails,
    draw_detections,
    draw_hand_line,
    draw_hud,
)
from juggletrack.pipeline.realtime import RealtimeAnalyzer, RealtimeConfig, RealtimeState


def run_live(
    source: int | str,
    detector: BallDetector,
    *,
    config: RealtimeConfig | None = None,
    display: bool = True,
    max_frames: int | None = None,
    on_state: Callable[[RealtimeState, np.ndarray], None] | None = None,
) -> tuple[RealtimeState, float]:
    """Run the realtime engine over a webcam (``source: int``) or file
    (``source: str``) until the stream ends, ``max_frames`` is reached, or
    (when ``display``) the user presses ``q``.

    Returns ``(final_state, mean_processing_fps)`` -- ``final_state`` comes
    from ``RealtimeAnalyzer.finalize()`` so a run still open at the last
    frame is force-closed exactly like the offline analyzer would close it.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"could not open source: {source!r}")
    fps_meta = cap.get(cv2.CAP_PROP_FPS) or 30.0
    is_file = isinstance(source, str)

    analyzer = RealtimeAnalyzer(config)
    idx = 0
    t_wall0 = time.monotonic()
    proc_fps = 0.0
    try:
        while True:
            if max_frames is not None and idx >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if is_file:
                # Simplified vs VideoReader.frames(): no seam-rebase, no
                # first-frame epoch subtraction. Verified on the synthetic
                # test video (mp4v/avfoundation backend, constant 30fps):
                # CAP_PROP_POS_MSEC after read() is 0.0 at idx==0 (the same
                # "healthy sentinel" VideoReader treats specially) and a
                # clean idx/fps-aligned value at idx>0, so `idx / fps_meta`
                # at idx==0 and `msec / 1000.0` thereafter agree with
                # VideoReader's own timestamps on this backend/content. A
                # VFR file with a mid-stream unsupported/non-monotonic PTS
                # run would still fall through to idx/fps here with no
                # rebase, unlike VideoReader -- acceptable for the live loop
                # (webcams never take this branch at all; they use the wall
                # clock below), but worth knowing if file-source timestamps
                # ever look jumpy on real footage.
                msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                t = msec / 1000.0 if msec > 0 else idx / fps_meta
            else:
                t = time.monotonic() - t_wall0

            dets = detector.detect(frame, idx, t)
            state = analyzer.feed(dets, t)

            draw_hand_line(frame, state.hand_line_y)
            if dets:
                verified = assign_detections(dets, state.window_arcs)
                draw_detections(frame, [d for d, a in zip(dets, verified) if a != -1])
            draw_arc_tails(frame, state.window_arcs, t)
            elapsed = time.monotonic() - t_wall0
            proc_fps = (idx + 1) / elapsed if elapsed > 0 else 0.0
            run_txt = f"run {state.runs_completed + (1 if state.run_active else 0)}"
            draw_hud(
                frame,
                f"{run_txt}  catches {state.catches_current_run}"
                f"  drops {state.drops_total}  fps {proc_fps:.0f}",
            )

            if on_state is not None:
                on_state(state, frame)
            if display:
                cv2.imshow("juggletrack live", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            idx += 1
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()
    return analyzer.finalize(), proc_fps
