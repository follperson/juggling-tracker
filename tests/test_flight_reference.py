import math
import subprocess
import sys

import numpy as np
import pytest

from juggletrack.eval.flight_reference import flight_reference, score_arcs, score_catches
from juggletrack.types import Arc, CatchEvent, Detection
from tests.helpers import Throw, cascade_throws, juggle_tracks

HAND_Y = 0.7
FPS = 30.0
# The reference puts hand level a quarter of the rise above the low point. A
# parabola is at that height this fraction of its duration after it leaves
# the hand, and the same fraction before it returns.
EDGE = (1 - math.sqrt(0.75)) / 2


def _stretch(tracks, factor):
    return [[d.model_copy(update={"t": d.t * factor}) for d in track] for track in tracks]


def _reshape(track, from_t, y_at):
    """Replace the track's height from from_t on with y_at(seconds since from_t)."""
    return [
        d.model_copy(update={"y": y_at(d.t - from_t)}) if d.t > from_t else d
        for d in track
    ]


def _assert_flights_match_throws(flights, throws, *, abs_t=0.01, abs_apex_t=1 / FPS):
    assert [f.ball for f in flights] == [th.ball for th in throws]
    for f, th in zip(flights, throws):
        assert f.launch_t == pytest.approx(th.t0 + EDGE * th.dur, abs=abs_t)
        assert f.landing_t == pytest.approx(th.t0 + (1 - EDGE) * th.dur, abs=abs_t)
        assert f.apex_t == pytest.approx(th.t0 + th.dur / 2, abs=abs_apex_t)
        assert f.apex_y == pytest.approx(HAND_Y - th.rise, abs=0.01)
        assert f.apex_x == pytest.approx((th.x0 + th.x1) / 2, abs=0.02)
        assert f.landing_x == pytest.approx(th.x1 - EDGE * (th.x1 - th.x0), abs=0.01)
        assert f.hand_y == pytest.approx(HAND_Y - 0.25 * th.rise, abs=0.01)


def test_finds_every_flight_of_a_cascade_with_its_times():
    throws = cascade_throws(n_balls=3, per_ball=6)

    ref = flight_reference(juggle_tracks(throws))

    assert all(f.whole and f.caught for f in ref.flights)
    _assert_flights_match_throws(ref.flights, throws)
    assert ref.trusted
    assert [n for _, n in ref.sweep] == [18] * 5
    assert ref.median_flight_s == pytest.approx((1 - 2 * EDGE) * 0.6, abs=0.01)


def test_slow_motion_gives_the_same_flights_five_times_slower():
    tracks = juggle_tracks(cascade_throws())

    real, slow = flight_reference(tracks), flight_reference(_stretch(tracks, 5))

    assert len(slow.flights) == len(real.flights) == 18
    assert slow.trusted and slow.sweep == real.sweep
    for a, b in zip(real.flights, slow.flights):
        assert (b.ball, b.whole, b.caught) == (a.ball, a.whole, a.caught)
        assert b.launch_t == pytest.approx(5 * a.launch_t)
        assert b.apex_t == pytest.approx(5 * a.apex_t)
        assert b.landing_t == pytest.approx(5 * a.landing_t)
        assert (b.apex_x, b.apex_y, b.landing_x) == pytest.approx(
            (a.apex_x, a.apex_y, a.landing_x))


def test_label_noise_keeps_the_count_and_the_landing_times():
    throws = cascade_throws()
    rng = np.random.default_rng(3)
    noisy = [
        [d.model_copy(update={"x": d.x + rng.normal(0, 0.002), "y": d.y + rng.normal(0, 0.002)})
         for d in track]
        for track in juggle_tracks(throws)
    ]

    ref = flight_reference(noisy)

    assert ref.trusted and len(ref.counted) == 18
    # Noise moves the highest sample along the flat top of the parabola.
    _assert_flights_match_throws(ref.flights, throws, abs_t=0.02, abs_apex_t=0.1)


def test_whole_pixel_labels_in_slow_motion_keep_the_apex_centred():
    # Meschke labels are whole pixels, so the top of a slow flight is a run
    # of equal samples. Here it is about five samples long.
    throws = cascade_throws(per_ball=3, dur=3.0, dwell=1.5)
    pixels = [
        [d.model_copy(update={"x": round(d.x * 480) / 480, "y": round(d.y * 848) / 848})
         for d in track]
        for track in juggle_tracks(throws)
    ]

    ref = flight_reference(pixels)

    assert ref.trusted
    _assert_flights_match_throws(ref.flights, throws, abs_t=0.02, abs_apex_t=1 / FPS)


@pytest.mark.xfail(strict=True, reason="known gap: a three-sample median over the heights "
                   "removes these glitches, but on ss441 it also removes real 1-zips whose "
                   "swing sits at the threshold")
def test_single_frame_label_glitches_add_no_flights():
    throws = cascade_throws()
    tracks = juggle_tracks(throws)
    first, second = [th for th in throws if th.ball == 0][1:3]
    # One frame mislabelled far below the ball on its way down, and one far
    # above it on its way up, as hand-labelled CSVs sometimes are.
    for t, jump in ((first.t0 + 0.75 * first.dur, 0.3), (second.t0 + 0.2 * second.dur, -0.2)):
        i = round(t * FPS)
        tracks[0][i] = tracks[0][i].model_copy(update={"y": tracks[0][i].y + jump})

    ref = flight_reference(tracks)

    assert len(ref.flights) == 18
    _assert_flights_match_throws(ref.flights, throws)
    assert ref.trusted


def test_scattered_missing_samples_change_nothing():
    throws = cascade_throws()
    thinned = [[d for d in track if d.frame_idx % 4] for track in juggle_tracks(throws)]

    ref = flight_reference(thinned)

    assert all(f.whole for f in ref.flights)
    _assert_flights_match_throws(ref.flights, throws, abs_t=0.02, abs_apex_t=2 / FPS)


def test_a_flight_with_a_long_hole_is_found_but_not_counted():
    throws = cascade_throws()
    holed = throws[7]
    tracks = juggle_tracks(throws)
    tracks[holed.ball] = [
        d for d in tracks[holed.ball]
        if not holed.t0 + 0.3 * holed.dur < d.t < holed.t0 + 0.7 * holed.dur
    ]

    ref = flight_reference(tracks)

    assert len(ref.flights) == 18
    assert [f.whole for f in ref.flights] == [th != holed for th in throws]
    assert len(ref.counted) == 17


def test_a_dropped_ball_is_a_flight_nobody_caught():
    throws = [th for th in cascade_throws() if th.ball != 0 or th.t0 < 3.0]
    dropped = [th for th in throws if th.ball == 0][-1]
    depth = 0.25
    floor = HAND_Y + depth
    # The same parabola, followed past the hand, meets the floor here.
    impact = dropped.dur * (1 + math.sqrt(1 + depth / dropped.rise)) / 2
    bounce = Throw(ball=0, t0=dropped.t0 + impact, dur=0.35, rise=0.12, x0=0.7, x1=0.72)

    def after_the_hand(dt):
        t = dropped.t0 + dropped.dur + dt
        if t < bounce.t0:
            return dropped.position(t, HAND_Y)[1]
        if t < bounce.t0 + bounce.dur:
            return bounce.position(t, floor)[1]
        return floor

    tracks = juggle_tracks(throws, tail=2.0)
    tracks[0] = _reshape(tracks[0], dropped.t0 + dropped.dur, after_the_hand)

    ref = flight_reference(tracks)

    fell, bounced = [f for f in ref.flights if not f.caught]
    assert fell.ball == 0 and fell.whole
    assert fell.apex_t == pytest.approx(dropped.t0 + dropped.dur / 2, abs=1 / FPS)
    assert fell.landing_t == pytest.approx(
        dropped.t0 + (1 - EDGE) * dropped.dur, abs=0.01)
    # The bounce off the floor is a flight too, and nobody caught it either.
    assert bounced.ball == 0
    assert bounced.apex_t == pytest.approx(bounce.t0 + bounce.dur / 2, abs=1 / FPS)
    _assert_flights_match_throws(ref.counted, [th for th in throws if th != dropped])
    assert ref.trusted


def test_a_ball_lowered_after_its_catch_was_still_caught():
    throws = cascade_throws()
    tracks = juggle_tracks(throws, tail=4.0)
    for ball, track in enumerate(tracks):
        last = max(th.t0 + th.dur for th in throws if th.ball == ball)
        # The juggler holds for a moment, then lowers the ball well past the
        # floor line over two seconds.
        tracks[ball] = _reshape(
            track, last + 0.3, lambda dt: HAND_Y + 0.25 * min(1.0, dt / 2.0))

    ref = flight_reference(tracks)

    assert all(f.caught for f in ref.flights)
    _assert_flights_match_throws(ref.counted, throws)


def test_a_held_ball_adds_no_flights():
    throws = cascade_throws()
    tracks = juggle_tracks(throws)
    held = [
        d.model_copy(update={"x": 0.5, "y": HAND_Y + 0.005 * math.sin(9 * d.t)})
        for d in tracks[0]
    ]

    ref = flight_reference([*tracks, held])

    assert all(f.ball != 3 for f in ref.flights)
    _assert_flights_match_throws(ref.flights, throws)
    assert ref.trusted


def test_flights_cut_by_the_ends_of_the_track_are_not_counted():
    throws = cascade_throws()
    # The labels start while the first ball is on its way up and stop while
    # the last two throws are in the air.
    start = throws[0].t0 + 0.3 * throws[0].dur
    end = throws[-2].t0 + 0.75 * throws[-2].dur
    tracks = [[d for d in track if start <= d.t <= end] for track in juggle_tracks(throws)]

    ref = flight_reference(tracks)

    assert len(ref.flights) == 18
    first, *middle, second_last, last = ref.flights
    assert first.launch_t is None and first.landing_t is not None and not first.whole
    for cut in (second_last, last):
        assert cut.launch_t is not None and cut.landing_t is None and not cut.whole
    _assert_flights_match_throws(ref.counted, throws[1:-2])
    assert ref.counted == middle
    assert ref.span == pytest.approx((start, end), abs=1 / FPS)


def test_a_pattern_that_mixes_high_throws_with_low_passes_is_flagged():
    throws = []
    for k in range(8):
        for ball in range(3):
            t0 = 0.5 + 1.2 * k + 0.4 * ball
            throws.append(Throw(ball=ball, t0=t0, dur=0.6, rise=0.3, x0=0.35, x1=0.65))
            throws.append(Throw(ball=ball, t0=t0 + 0.8, dur=0.2, rise=0.035, x0=0.65, x1=0.35))

    ref = flight_reference(juggle_tracks(throws))

    counts = [n for _, n in ref.sweep]
    assert counts[0] == 48 and counts[-1] == 24
    assert not ref.trusted


def test_no_labels_gives_an_untrusted_empty_reference():
    for tracks in ([], [[], []], [[Detection(frame_idx=0, t=0.0, x=0.5, y=0.5)]]):
        ref = flight_reference(tracks)
        assert ref.flights == [] and not ref.trusted


def test_the_module_stays_independent_of_the_extractor():
    loaded = subprocess.run(
        [sys.executable, "-c",
         "import sys, juggletrack.eval.flight_reference;"
         "print(*sorted(m for m in sys.modules if m.startswith('juggletrack')))"],
        capture_output=True, text=True, check=True,
    ).stdout.split()

    assert "juggletrack.eval.flight_reference" in loaded
    assert not [
        m for m in loaded
        if m.startswith(("juggletrack.arcs", "juggletrack.events", "juggletrack.analyze"))
    ]


def _catch(throw, late=0.0, x=None):
    """A catch where the pipeline puts it: back at hand height, in the catching hand."""
    return CatchEvent(t=throw.t0 + throw.dur + late, x=throw.x1 if x is None else x, arc_id=0)


@pytest.fixture
def cascade():
    throws = cascade_throws()
    return throws, flight_reference(juggle_tracks(throws))


def test_every_flight_caught_once_scores_clean(cascade):
    throws, ref = cascade

    score = score_catches(ref, [_catch(th) for th in throws])

    assert (score.flights, score.catches, score.matched) == (18, 18, 18)
    assert (score.duplicates, score.unmatched, score.ignored, score.count_error) == (0, 0, 0, 0)
    assert score.verdicts == ["matched"] * 18 and score.missed == []


def test_a_missing_catch_is_a_missed_flight(cascade):
    throws, ref = cascade

    score = score_catches(ref, [_catch(th) for k, th in enumerate(throws) if k != 5])

    assert (score.matched, score.duplicates, score.unmatched) == (17, 0, 0)
    assert score.missed == [5] and score.count_error == -1


def test_a_second_catch_on_one_flight_is_a_duplicate(cascade):
    throws, ref = cascade
    catches = [_catch(th) for th in throws] + [_catch(throws[5], late=0.02)]

    score = score_catches(ref, catches)

    assert (score.matched, score.duplicates, score.unmatched) == (18, 1, 0)
    assert sorted([score.verdicts[5], score.verdicts[-1]]) == ["duplicate", "matched"]
    assert score.missed == [] and score.count_error == 1


def test_a_duplicate_does_not_fill_in_for_a_missed_neighbour(cascade):
    throws, ref = cascade
    catches = [_catch(th) for k, th in enumerate(throws) if k != 6]
    catches.append(_catch(throws[5], late=0.02))

    score = score_catches(ref, catches)

    assert (score.matched, score.duplicates, score.unmatched) == (17, 1, 0)
    assert score.missed == [6] and score.count_error == 0


def test_a_catch_near_no_landing_matches_no_flight(cascade):
    throws, ref = cascade
    phantom = CatchEvent(t=throws[5].t0 + throws[5].dur + 0.15, x=0.5, arc_id=0)

    score = score_catches(ref, [_catch(th) for th in throws] + [phantom])

    assert (score.matched, score.duplicates, score.unmatched) == (18, 0, 1)
    assert score.verdicts[-1] == "unmatched" and score.count_error == 1


def test_a_catch_in_the_wrong_hand_matches_no_flight(cascade):
    throws, ref = cascade
    catches = [_catch(th, x=th.x0 if k == 5 else None) for k, th in enumerate(throws)]

    score = score_catches(ref, catches)

    assert score.verdicts[5] == "unmatched"
    assert (score.matched, score.unmatched) == (17, 1) and score.missed == [5]


def test_catches_shifted_by_half_a_beat_all_match_no_flight(cascade):
    throws, ref = cascade

    score = score_catches(ref, [_catch(th, late=0.15) for th in throws])

    assert (score.matched, score.duplicates, score.unmatched) == (0, 0, 18)
    assert score.count_error == 0 and len(score.missed) == 18


def test_the_offset_is_the_median_of_catch_minus_landing(cascade):
    throws, ref = cascade

    def at_landings(*shifts):
        return [CatchEvent(t=f.landing_t + shifts[k % len(shifts)], x=f.landing_x, arc_id=0)
                for k, f in enumerate(ref.flights)]

    late, early = score_catches(ref, at_landings(0.05)), score_catches(ref, at_landings(-0.05))
    mostly_late = score_catches(ref, at_landings(0.05, 0.05, -0.08))
    nothing = score_catches(ref, [_catch(th, late=0.15) for th in throws])

    assert late.matched == early.matched == mostly_late.matched == 18
    assert late.offset_s == pytest.approx(0.05)
    assert early.offset_s == pytest.approx(-0.05)
    assert mostly_late.offset_s == pytest.approx(0.05)
    assert nothing.matched == 0 and nothing.offset_s is None


def test_two_landings_close_together_each_keep_their_own_catch():
    first = Throw(ball=0, t0=1.0, dur=0.6, rise=0.3, x0=0.35, x1=0.5)
    second = Throw(ball=1, t0=1.14, dur=0.6, rise=0.3, x0=0.65, x1=0.5)
    ref = flight_reference(juggle_tracks([first, second]))
    assert [f.landing_t for f in ref.counted] == pytest.approx([1.56, 1.70], abs=0.01)
    # The later catch is the nearest catch to the first landing, and it is
    # the only one in reach of the second. Giving each landing its nearest
    # catch would leave the second flight out and call the early catch a
    # duplicate.
    catches = [CatchEvent(t=1.49, x=0.5, arc_id=0), CatchEvent(t=1.61, x=0.5, arc_id=1)]

    score = score_catches(ref, catches)

    assert (score.matched, score.duplicates, score.unmatched) == (2, 0, 0)


def test_the_time_tolerance_follows_the_speed_of_the_footage():
    throws = cascade_throws()
    tracks = juggle_tracks(throws)
    late = 0.2

    real = score_catches(flight_reference(tracks), [_catch(th, late=late) for th in throws])
    slow = score_catches(
        flight_reference(_stretch(tracks, 5)),
        [CatchEvent(t=5 * (th.t0 + th.dur) + late, x=th.x1, arc_id=0) for th in throws],
    )

    # A fifth of a second is a third of a real-time flight, and a fifteenth
    # of the same flight in slow motion.
    assert real.matched == 0
    assert slow.matched == 18 and slow.unmatched == 0


def test_catches_outside_the_labels_or_on_cut_flights_are_ignored():
    throws = cascade_throws()
    start = throws[0].t0 + 0.3 * throws[0].dur
    end = throws[-2].t0 + 0.75 * throws[-2].dur
    ref = flight_reference(
        [[d for d in track if start <= d.t <= end] for track in juggle_tracks(throws)])
    catches = [_catch(th) for th in throws[:-2]]
    catches += [CatchEvent(t=start - 0.2, x=0.5, arc_id=0), CatchEvent(t=end + 0.2, x=0.5, arc_id=0)]

    score = score_catches(ref, catches)

    assert score.verdicts == ["ignored"] + ["matched"] * 15 + ["ignored"] * 2
    assert (score.flights, score.catches, score.matched, score.ignored) == (15, 18, 15, 3)
    assert score.count_error == 0 and score.missed == []


def test_a_catch_on_a_dropped_ball_matches_no_flight():
    throws = [th for th in cascade_throws() if th.ball != 0 or th.t0 < 3.0]
    dropped = [th for th in throws if th.ball == 0][-1]
    tracks = juggle_tracks(throws, tail=2.0)
    tracks[0] = _reshape(
        tracks[0], dropped.t0 + dropped.dur,
        lambda dt: min(HAND_Y + 0.25, dropped.position(dropped.t0 + dropped.dur + dt, HAND_Y)[1]),
    )
    ref = flight_reference(tracks)
    caught = [th for th in throws if th != dropped]

    honest = score_catches(ref, [_catch(th) for th in caught])
    fooled = score_catches(ref, [_catch(th) for th in throws])

    assert (honest.flights, honest.matched, honest.count_error) == (len(caught), len(caught), 0)
    assert honest.missed == []
    assert (fooled.matched, fooled.unmatched, fooled.count_error) == (len(caught), 1, 1)


def test_scoring_against_no_flights_calls_every_catch_unmatched():
    held = [Detection(frame_idx=i, t=i / FPS, x=0.5, y=HAND_Y) for i in range(60)]

    score = score_catches(flight_reference([held]), [CatchEvent(t=1.0, x=0.5, arc_id=0)])

    assert (score.flights, score.matched, score.unmatched, score.count_error) == (0, 0, 1, 1)


def _arc(throw, arc_id=0, *, dx=0.0, until=1.0):
    """The parabola of a throw as the extractor would report it."""
    return Arc(
        id=arc_id, t_start=throw.t0, t_end=throw.t0 + until * throw.dur,
        ay=4 * throw.rise / throw.dur**2, by=-4 * throw.rise / throw.dur, cy=HAND_Y,
        bx=(throw.x1 - throw.x0) / throw.dur, cx=throw.x0 + dx, n_points=18, rmse=0.0,
    )


def test_arcs_that_follow_the_flights_explain_them_all(cascade):
    throws, ref = cascade

    score = score_arcs(ref, [_arc(th, i) for i, th in enumerate(throws)])

    assert (score.flights, score.arcs, score.explained, score.repeated) == (18, 18, 18, 0)
    assert score.unexplained == []


def test_a_flight_without_an_arc_through_its_apex_is_unexplained(cascade):
    throws, ref = cascade
    arcs = [_arc(th, i) for i, th in enumerate(throws) if i not in (3, 8, 12)]
    arcs.append(_arc(throws[3], 3, until=0.4))  # stops before the apex
    arcs.append(_arc(throws[8], 8, dx=0.1))  # in flight at the apex time, but elsewhere

    score = score_arcs(ref, arcs)

    assert score.explained == 15 and score.unexplained == [3, 8, 12]


def test_two_arcs_on_one_flight_are_reported_as_repeated(cascade):
    throws, ref = cascade
    arcs = [_arc(th, i) for i, th in enumerate(throws)] + [_arc(throws[4], 18, dx=0.005)]

    score = score_arcs(ref, arcs)

    assert (score.explained, score.repeated, score.unexplained) == (18, 1, [])
