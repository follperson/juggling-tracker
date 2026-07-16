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


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        VideoReader("/nonexistent/nope.mp4")


def test_unreadable_file_raises(tmp_path):
    bogus = tmp_path / "not_a_video.mp4"
    bogus.write_bytes(b"this is not an mp4")
    with pytest.raises(ValueError):
        VideoReader(bogus)
