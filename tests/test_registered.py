"""Generated scenes test proposal guarantees without private footage or weights."""
import json
import subprocess
import sys

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from juggletrack.detect.registered import BallProfile, register_ball, registered_proposals


def image(x=80, y=100, radius=12, color=(60, 200, 120)):
    frame = np.zeros((240, 360, 3), dtype=np.uint8)
    cv2.circle(frame, (x, y), radius, color, -1)
    return frame


def enrollment():
    return register_ball(image(), 80, 100, 12)


def stream(xs, times=None):
    if times is None:
        times = [i/30 for i in range(len(xs))]
    return [(i, t, image(x=x)) for i, (x, t) in enumerate(zip(xs, times))]


def test_profile_round_trip_is_portable_and_full_radius():
    profile = enrollment()
    assert profile.radius_fraction == 12/240
    data = json.loads(profile.model_dump_json())
    assert set(data) == {'schema_version', 'hue', 'hue_halfwidth', 'saturation_min',
                         'value_min', 'radius_fraction'}
    assert BallProfile.model_validate_json(profile.model_dump_json()) == profile
    with pytest.raises(ValidationError):
        BallProfile.model_validate({**data, 'path': 'frame.jpg'})
    with pytest.raises(ValidationError):
        BallProfile.model_validate({**data, 'hue': float('nan')})
    with pytest.raises(ValidationError):
        BallProfile.model_validate({**data, 'schema_version': 2})
    with pytest.raises(ValidationError):
        BallProfile.model_validate({**data, 'schema_version': True})


@pytest.mark.parametrize('color', [(120, 120, 120), (0, 0, 0), (5, 10, 8)])
def test_achromatic_or_dark_enrollment_is_explicit(color):
    with pytest.raises(ValueError, match='achromatic|dark'):
        register_ball(image(color=color), 80, 100, 12)


def test_noisy_and_mixed_hue_enrollment_rejected():
    frame = np.random.default_rng(1).integers(0, 256, (240, 360, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match='enrollment'):
        register_ball(frame, 80, 100, 12)
    frame = image(color=(0, 0, 255))
    frame[:, 80:] = image(color=(0, 255, 0))[:, 80:]
    with pytest.raises(ValueError, match='mixed|ambiguous'):
        register_ball(frame, 80, 100, 12)


def test_red_hue_wrap_is_supported():
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    hsv = np.zeros_like(frame)
    hsv[:, :, 1:] = (220, 180)
    hsv[:, :60, 0], hsv[:, 60:, 0] = 179, 1
    frame[:] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    profile = register_ball(frame, 60, 60, 15)
    assert min(profile.hue, 180-profile.hue) < 2


@pytest.mark.parametrize('selection', [(0, 0, 12), (80, 100, 3), (80., 100, 12),
                                       (80, True, 12), (350, 100, 12)])
def test_invalid_selection(selection):
    with pytest.raises(ValueError, match='circle'):
        register_ball(image(), *selection)


@pytest.mark.parametrize('bad', [np.zeros((20, 20), np.uint8), np.zeros((20, 20, 3)),
                                 np.zeros((0, 20, 3), np.uint8)])
def test_invalid_frames(bad):
    with pytest.raises(ValueError, match='BGR'):
        register_ball(bad, 10, 10, 4)


def test_moving_ball_outputs_chronological_once_with_float_pts():
    times = [.01, .044, .08, .111, .15, .18]
    frames = stream([60, 68, 76, 84, 92, 100], times)
    proposals = list(registered_proposals(frames, [enrollment()]))
    assert [d.frame_idx for d in proposals] == list(range(6))
    assert [d.t for d in proposals] == times
    assert all(isinstance(d.t, float) for d in proposals)
    assert np.allclose([d.x*360 for d in proposals], [60, 68, 76, 84, 92, 100], atol=.5)
    assert all(d.confidence == 1 for d in proposals)


def test_static_ball_and_repeated_frames_emit_nothing():
    assert list(registered_proposals(stream([80]*8), [enrollment()])) == []
    frame = image()
    repeated = [(i, i/30, frame) for i in range(6)]
    assert list(registered_proposals(repeated, [enrollment()])) == []


def test_duplicate_and_overlapping_profiles_do_not_duplicate_ball():
    profile = enrollment()
    overlapping = profile.model_copy(update={'hue': profile.hue+.5})
    frames = stream([60, 68, 76, 84, 92, 100])
    expected = list(registered_proposals(frames, [profile]))
    assert list(registered_proposals(frames, [profile, profile, overlapping])) == expected


def test_separate_balls_are_not_deduplicated():
    frames = []
    for i in range(6):
        frame = image(x=40+i*8, y=60)
        cv2.circle(frame, (220+i*8, 180), 12, (60, 200, 120), -1)
        frames.append((i, i/30, frame))
    out = list(registered_proposals(frames, [enrollment()]))
    assert len(out) == 12
    assert all(sum(d.frame_idx == i for d in out) == 2 for i in range(6))


def test_ambiguous_crossing_is_not_joined_into_a_confirmed_tracklet():
    frames = []
    for i, (x1, x2) in enumerate([(70, 130), (82, 118), (94, 106), (106, 94), (118, 82)]):
        frame = image(x=x1, radius=8)
        cv2.circle(frame, (x2, 100), 8, (60, 200, 120), -1)
        frames.append((i, i/30, frame))
    profile = register_ball(image(radius=8), 80, 100, 8)
    assert list(registered_proposals(frames, [profile])) == []


def test_same_color_background_is_not_a_prop():
    frame = np.full((240, 360, 3), (60, 200, 120), dtype=np.uint8)
    assert list(registered_proposals([(i, i/30, frame) for i in range(5)], [enrollment()])) == []


def test_two_frame_lookahead_and_generator_laziness():
    consumed = []
    def frames():
        for row in stream([60, 68, 76, 84, 92, 100]):
            consumed.append(row[0])
            yield row
    proposals = registered_proposals(frames(), [enrollment()])
    assert consumed == []
    assert next(proposals).frame_idx == 0
    assert consumed == [0, 1, 2]


@pytest.mark.parametrize('frames', [
    [(0, 0., image()), (0, .1, image())],
    [(0, 0., image()), (1, 0., image())],
    [(0, float('nan'), image())], [(0, True, image())],
    [(True, 0., image())], [(-1, 0., image())],
    [(0, 0., image()), (1, .1, image()[:120])],
])
def test_invalid_recording_inputs_are_explicit(frames):
    with pytest.raises(ValueError, match='indices|geometry'):
        list(registered_proposals(frames, [enrollment()]))


def test_gaps_cannot_supply_three_consecutive_frames():
    frames = [(i*2, i/30, image(x=60+i*8)) for i in range(5)]
    assert list(registered_proposals(frames, [enrollment()])) == []


def test_empty_and_short_recordings():
    for frames in ([], stream([60]), stream([60, 68])):
        assert list(registered_proposals(frames, [enrollment()])) == []
    with pytest.raises(ValueError, match='profiles'):
        list(registered_proposals([], []))


def test_module_import_does_not_load_heavy_detector_dependencies():
    code = ('import sys; import juggletrack.detect.registered; '
            'assert not any(x in sys.modules for x in ("cv2", "torch", "scipy"))')
    subprocess.run([sys.executable, '-c', code], check=True)


def test_motion_witness_requires_the_same_registered_color_across_three_frames():
    green, red = (0, 255, 0), (0, 0, 255)
    colors = [green, red, green, red]
    frames = [(i, i/30, image(x=80+8*i, color=color)) for i, color in enumerate(colors)]
    profiles = [register_ball(frame, 80+8*i, 100, 12) for i, _, frame in frames[:2]]
    assert list(registered_proposals(frames, profiles)) == []



def test_overlapping_sizes_each_verify_motion_before_deduplication():
    small = enrollment()
    large = register_ball(image(radius=20), 80, 100, 20)
    frames = stream([60, 63, 66, 69, 72, 75])
    small_only = list(registered_proposals(frames, [small]))
    assert len(small_only) == len(frames)
    assert list(registered_proposals(frames, [large])) == []
    assert list(registered_proposals(frames, [large, small])) == small_only
    assert list(registered_proposals(frames, [small, large])) == small_only
