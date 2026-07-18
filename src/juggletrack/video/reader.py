"""cv2-backed video reading. The only module (with pipeline/) allowed to import cv2.

Phone videos carry rotation metadata; OpenCV >= 4.5 applies it automatically
(CAP_PROP_ORIENTATION_AUTO defaults on), so width/height are post-rotation.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel


class VideoInfo(BaseModel):
    path: str
    fps: float
    width: int
    height: int
    frame_count: int
    duration: float


class VideoReader:
    def __init__(self, path: str | Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"video not found: {path}")
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise ValueError(f"cv2 could not open video: {path}")
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
            raise ValueError(f"video has no readable frames/fps: {path}")
        self.info = VideoInfo(
            path=str(path),
            fps=fps,
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=frame_count,
            duration=frame_count / fps,
        )

    def frames(self) -> Iterator[tuple[int, float, np.ndarray]]:
        """Yield (frame_idx, timestamp_seconds, BGR frame) from the start.

        Phone (VFR) footage has a container fps that is only an *average*;
        idx/fps drifts from the actual decode time and degrades downstream
        parabola fits. Prefer cv2's per-frame presentation timestamp
        (CAP_PROP_POS_MSEC, read AFTER a successful ``read()``) and fall back
        to idx/fps only when PTS looks unusable.

        Measured on a synthetic constant-fps clip (macOS, mp4v/avfoundation
        backend): POS_MSEC after reading frame idx==0 is 0.0, and after frame
        idx==1 is ~33.33ms (=1000/30) -- i.e. this backend reports
        *start-of-frame* PTS, not end-of-frame, so idx 0's timestamp is
        already 0.0 with no offset needed. Other backends are known to report
        POS_MSEC using an end-of-frame convention (first frame ~1000/fps
        instead of 0); to stay correct there too, we treat whatever PTS is
        observed at idx==0 as the epoch and subtract it from every
        subsequent PTS, so t always starts at 0.0 and stays aligned to the
        same convention the backend uses.

        Sanity/fallback rule: PTS is trusted (``self._pts_ok``) until a frame
        with idx > 0 reports ``msec <= previous_msec`` (non-monotonic) or
        ``msec == 0`` (a common "unsupported" sentinel on some backends).
        idx == 0 reporting msec == 0 is normal/expected and never breaks
        trust. Once broken, every remaining frame in this iteration falls
        back to idx/fps -- mixing time bases within one clip would be worse
        than using either consistently.
        """
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        idx = 0
        pts_ok = True
        first_msec = 0.0
        prev_msec = 0.0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            if pts_ok:
                msec = self._cap.get(cv2.CAP_PROP_POS_MSEC)
                if idx == 0:
                    if msec < 0:
                        pts_ok = False
                        t = idx / self.info.fps
                    else:
                        first_msec = msec
                        prev_msec = msec
                        t = 0.0
                elif msec <= prev_msec or msec == 0.0:
                    pts_ok = False
                    t = idx / self.info.fps
                else:
                    t = (msec - first_msec) / 1000.0
                    prev_msec = msec
            else:
                t = idx / self.info.fps
            yield idx, t, frame
            idx += 1

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
