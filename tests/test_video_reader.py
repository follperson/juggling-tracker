import numpy as np
import pytest

from juggletrack.video.reader import VideoReader
from tests.helpers import write_test_video


@pytest.fixture()
def video_path(tmp_path):
    path = tmp_path / "orbit.mp4"
    write_test_video(path, n_frames=60, fps=30.0, size=(320, 240))
    return path


def test_info_fields(video_path):
    with VideoReader(video_path) as reader:
        info = reader.info
        assert info.fps == pytest.approx(30.0, rel=0.01)
        assert (info.width, info.height) == (320, 240)
        assert info.frame_count == 60
        assert info.duration == pytest.approx(2.0, rel=0.02)


def test_frames_iteration(video_path):
    with VideoReader(video_path) as reader:
        frames = list(reader.frames())
    assert len(frames) == 60
    idx0, t0, frame0 = frames[0]
    assert idx0 == 0 and t0 == 0.0
    assert frame0.shape == (240, 320, 3) and frame0.dtype == np.uint8
    idx9, t9, _ = frames[9]
    assert idx9 == 9 and t9 == pytest.approx(9 / 30.0)


def test_frame_timestamps_track_decode_pts_within_2ms(video_path):
    """VFR fix: timestamps should come from cv2's per-frame PTS (POS_MSEC),
    not idx/fps. On a constant-fps synthetic clip the two agree almost
    exactly, so this also pins backward compatibility."""
    with VideoReader(video_path) as reader:
        frames = list(reader.frames())
    for idx, t, _ in frames:
        assert t == pytest.approx(idx / 30.0, abs=0.002)


def test_frame_timestamps_strictly_increasing(video_path):
    with VideoReader(video_path) as reader:
        frames = list(reader.frames())
    times = [t for _, t, _ in frames]
    assert all(b > a for a, b in zip(times, times[1:]))


def test_frame_timestamps_prefer_decode_pts_over_container_fps(
    video_path, monkeypatch
):
    """The container reports 30fps, but decode PTS drifting to a real 24fps
    (simulating a VFR phone capture where the header fps is only an average)
    must win over idx/container_fps."""
    import cv2

    class _DriftingPtsCap:
        def __init__(self, real):
            self._real = real
            self._idx = 0

        def read(self):
            ok, frame = self._real.read()
            if ok:
                self._idx += 1
            return ok, frame

        def get(self, prop_id):
            if prop_id == cv2.CAP_PROP_POS_MSEC:
                return (self._idx - 1) * (1000.0 / 24.0)
            return self._real.get(prop_id)

        def __getattr__(self, name):
            return getattr(self._real, name)

    with VideoReader(video_path) as reader:
        monkeypatch.setattr(reader, "_cap", _DriftingPtsCap(reader._cap))
        frames = list(reader.frames())

    for idx, t, _ in frames:
        assert t == pytest.approx(idx / 24.0, abs=1e-9)
        assert t != pytest.approx(idx / 30.0, abs=1e-9) or idx == 0


def test_frame_timestamps_fall_back_to_idx_over_fps_when_pts_unavailable(
    video_path, monkeypatch
):
    """Some backends report all-zero POS_MSEC; the reader must detect that
    and fall back to idx/fps for every frame rather than mixing time bases."""
    import cv2

    class _ZeroMsecCap:
        """Proxy that zeroes POS_MSEC and forwards everything else to the
        real cv2.VideoCapture (whose attributes are read-only, so it can't
        be monkeypatched directly)."""

        def __init__(self, real):
            self._real = real

        def get(self, prop_id):
            if prop_id == cv2.CAP_PROP_POS_MSEC:
                return 0.0
            return self._real.get(prop_id)

        def __getattr__(self, name):
            return getattr(self._real, name)

    with VideoReader(video_path) as reader:
        monkeypatch.setattr(reader, "_cap", _ZeroMsecCap(reader._cap))
        frames = list(reader.frames())

    for idx, t, _ in frames:
        assert t == idx / 30.0


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        VideoReader("/nonexistent/nope.mp4")


def test_unreadable_file_raises(tmp_path):
    bogus = tmp_path / "not_a_video.mp4"
    bogus.write_bytes(b"this is not an mp4")
    with pytest.raises(ValueError):
        VideoReader(bogus)
