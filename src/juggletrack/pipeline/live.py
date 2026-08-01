"""Live capture loop: webcam or file -> detector -> RealtimeAnalyzer -> HUD.

cv2 is allowed at module scope here (unlike arcs/extract.py, realtime.py):
this is the capture/display layer, mirroring video/reader.py's carve-out.
"""
from __future__ import annotations

import sys
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


def _hud_lines(state: RealtimeState, run_history: list[int], fps: float) -> list[str]:
    """The live HUD's text, as plain strings (no cv2 -- easy to test in
    isolation). Branches on run_active like the offline overlay's
    idle/active HUD (pipeline/overlay.py) instead of always showing
    "run N" with a stale catches_current_run of 0 between runs. A second
    line (spec §8's "run history") lists the last few completed runs'
    catch counts, when any exist yet."""
    if state.run_active:
        main = (
            f"run {state.runs_completed + 1}  catches {state.catches_current_run}"
            f"  drops {state.drops_total}  fps {fps:.0f}"
        )
    else:
        main = (
            f"runs {state.runs_completed}  catches {state.catches_total}"
            f"  drops {state.drops_total}  fps {fps:.0f}"
        )
    lines = [main]
    if run_history:
        lines.append("history: " + ", ".join(str(c) for c in run_history[-5:]))
    return lines


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
    ``mean_processing_fps`` is warmup-corrected: the first couple of
    frame-to-frame intervals can carry residual cap-open/backend-init
    latency and would otherwise drag a from-the-start mean down.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"could not open source: {source!r}")
    fps_meta = cap.get(cv2.CAP_PROP_FPS) or 30.0
    is_file = isinstance(source, str)
    # A file source's own frame count, known up front -- lets the loop tell
    # a clean end-of-stream apart from stopping well short of it (see the
    # short-read warning in `finally` below).
    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_file else 0

    analyzer = RealtimeAnalyzer(config)
    idx = 0
    t_wall0: float | None = None
    frame_ts: list[float] = []
    fps_ema = 0.0
    run_history: list[int] = []
    prev_runs_completed = 0
    prev_catches_current_run = 0
    stopped_by_max_frames = False
    # File-source PTS health rule -- same algorithm as VideoReader.frames()
    # (video/reader.py): trust CAP_PROP_POS_MSEC until a frame with idx>0
    # reports a non-monotonic or zero value (a common "unsupported"
    # sentinel on some backends), then permanently rebase off the last
    # known-good (t, idx) pair instead of jumping to the absolute idx/fps
    # clock (which can go backward relative to where PTS had already
    # drifted). Duplicated here rather than shared: VideoReader exposes a
    # generator over a whole read pass, while this loop interleaves
    # detection/HUD work per frame and also drives webcam sources through
    # an entirely different (wall-clock) branch -- the two call shapes
    # don't share a natural extraction point without a larger refactor.
    pts_ok = True
    first_msec = 0.0
    prev_msec = 0.0
    last_good_t = 0.0
    last_good_idx = 0
    # Bounded retry for a transient webcam read failure (a dropped frame
    # shouldn't immediately conclude the camera is gone); file sources
    # never retry -- a file read failure is end-of-stream or corruption,
    # not a transient hiccup.
    read_retries_left = 3
    try:
        while True:
            if max_frames is not None and idx >= max_frames:
                stopped_by_max_frames = True
                break
            ok, frame = cap.read()
            if not ok:
                if not is_file and read_retries_left > 0:
                    read_retries_left -= 1
                    continue
                break
            read_retries_left = 3
            now_wall = time.monotonic()
            if t_wall0 is None:
                # Anchor AFTER the first successful read, not before
                # cap.read() -- otherwise frame 0's timestamp (webcam
                # branch) and every fps figure already carries whatever
                # cap-open/detector-construction latency preceded the loop.
                t_wall0 = now_wall

            if is_file:
                if pts_ok:
                    msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                    if idx == 0:
                        if msec < 0:
                            pts_ok = False
                            t = last_good_t + (idx - last_good_idx) / fps_meta
                        else:
                            first_msec = msec
                            prev_msec = msec
                            t = 0.0
                            last_good_t, last_good_idx = t, idx
                    elif msec <= prev_msec or msec == 0.0:
                        pts_ok = False
                        t = last_good_t + (idx - last_good_idx) / fps_meta
                    else:
                        t = (msec - first_msec) / 1000.0
                        prev_msec = msec
                        last_good_t, last_good_idx = t, idx
                else:
                    t = last_good_t + (idx - last_good_idx) / fps_meta
            else:
                t = now_wall - t_wall0

            dets = detector.detect(frame, idx, t)
            state = analyzer.feed(dets, t)

            # Track completed-run catch counts for the HUD's run-history
            # line. RealtimeState only exposes the CURRENT run's catch
            # count (catches_current_run resets to 0 the instant
            # runs_completed increments), so the last value observed
            # before a close IS that run's final count. If more than one
            # run closes within a single cycle (only possible across a
            # very long freeze/debounce gap), the same last-seen count is
            # recorded for each -- an accepted approximation for a
            # display-only history.
            if state.runs_completed > prev_runs_completed:
                run_history.extend(
                    [prev_catches_current_run] * (state.runs_completed - prev_runs_completed)
                )
            prev_runs_completed = state.runs_completed
            prev_catches_current_run = state.catches_current_run

            draw_hand_line(frame, state.hand_line_y)
            if dets:
                verified = assign_detections(dets, state.window_arcs)
                draw_detections(frame, [d for d, a in zip(dets, verified) if a != -1])
            draw_arc_tails(frame, state.window_arcs, t)

            # HUD fps is an EMA of INSTANTANEOUS per-frame fps (reflects a
            # recent slowdown/speedup quickly); the warmup-corrected mean
            # returned to the caller is computed separately, below.
            if frame_ts:
                dt = now_wall - frame_ts[-1]
                if dt > 0:
                    inst_fps = 1.0 / dt
                    fps_ema = inst_fps if fps_ema == 0.0 else 0.2 * inst_fps + 0.8 * fps_ema
            frame_ts.append(now_wall)

            for i, line in enumerate(_hud_lines(state, run_history, fps_ema)):
                draw_hud(frame, line, y=24 + 24 * i)

            if on_state is not None:
                on_state(state, frame)
            if display:
                # A display backend can fail to open a window (headless
                # CI, no X/Wayland session, a permission-denied
                # framebuffer) well after `display=True` was requested
                # legitimately -- fall back to headless rather than
                # crashing the whole session; finalize()/--out still run
                # normally either way.
                try:
                    cv2.imshow("juggletrack live", frame)
                    quit_pressed = cv2.waitKey(1) & 0xFF == ord("q")
                except cv2.error as exc:
                    print(
                        f"warning: cv2.imshow failed ({exc}); continuing headless",
                        file=sys.stderr,
                    )
                    display = False
                    quit_pressed = False
                if quit_pressed:
                    break
            idx += 1
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()
        # A file source that stopped well short of its own reported frame
        # count (and wasn't just cut off by --max-frames) likely hit a
        # decode problem partway through, not a clean end-of-stream. 90%:
        # tolerates a handful of trailing frames a container's metadata
        # over-counts without flagging a real, much-larger truncation.
        if (
            is_file and not stopped_by_max_frames
            and expected_frames > 0 and idx < 0.9 * expected_frames
        ):
            print(
                f"warning: only read {idx}/{expected_frames} frames from "
                f"{source!r}; stream may have ended early",
                file=sys.stderr,
            )

    if len(frame_ts) >= 3:
        # Warmup-corrected mean: drop the frame-0-to-1 interval too
        # (residual lazy-init latency can still bleed into it on some
        # backends), averaging from frame 1 onward instead.
        mean_fps = (len(frame_ts) - 2) / (frame_ts[-1] - frame_ts[1])
    elif len(frame_ts) >= 2:
        mean_fps = (len(frame_ts) - 1) / (frame_ts[-1] - frame_ts[0])
    else:
        mean_fps = 0.0
    return analyzer.finalize(), mean_fps
