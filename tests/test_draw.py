import numpy as np

from juggletrack.pipeline.draw import (
    draw_arc_tails,
    draw_detections,
    draw_drop_marker,
    draw_hand_line,
    draw_hud,
)
from juggletrack.types import Arc, Detection


def blank():
    return np.zeros((240, 320, 3), dtype=np.uint8)


def test_each_helper_draws_something():
    arc = Arc(id=0, t_start=0.0, t_end=1.1, ay=1.0, by=-1.1, cy=0.65,
              bx=0.1, cx=0.4, n_points=30, rmse=0.005)
    det = Detection(frame_idx=0, t=0.5, x=0.5, y=0.5)
    for fn, args in [
        (draw_hand_line, (0.65,)),
        (draw_detections, ([det],)),
        (draw_arc_tails, ([arc], 0.6)),
        (draw_hud, ("run 1  catches 5",)),
        (draw_drop_marker, (0.5, 0.8)),
    ]:
        frame = blank()
        fn(frame, *args)
        assert frame.sum() > 0, f"{fn.__name__} drew nothing"


def test_arc_tails_skip_inactive_arcs():
    arc = Arc(id=0, t_start=5.0, t_end=6.1, ay=1.0, by=-1.1, cy=0.65,
              bx=0.1, cx=0.4, n_points=30, rmse=0.005)
    frame = blank()
    draw_arc_tails(frame, [arc], t=1.0)  # long before the arc
    assert frame.sum() == 0


def test_draw_hud_y_offset_stacks_lines_without_overlap():
    """S2: draw_hud's y parameter lets live.py stack a second HUD line
    (run history) below the first without the two overlapping."""
    frame = blank()
    draw_hud(frame, "line one", y=24)
    frame_two_lines = frame.copy()
    draw_hud(frame_two_lines, "line two", y=48)

    assert frame_two_lines.sum() > frame.sum(), "the second line must add pixels"
    # Pixels well above line two's row (y=48's text extends up from its
    # baseline, but not past ~y=30) are identical between the two frames
    # -- adding the second line must not have touched the first.
    assert np.array_equal(frame[:28], frame_two_lines[:28])


DRAWN = [
    (draw_hand_line, (0.65,)),
    (draw_detections, ([Detection(frame_idx=0, t=0.5, x=0.5, y=0.5)],)),
    (draw_arc_tails, ([Arc(id=0, t_start=0.0, t_end=1.0, ay=0.0, by=0.0, cy=0.5,
                           bx=0.3, cx=0.2, n_points=30, rmse=0.0)], 0.5)),
    (draw_hud, ("run 1/2  catches 12  drops 0",)),
    (draw_drop_marker, (0.5, 0.5)),
]


def _drawn_height(fn, args, shape):
    frame = np.zeros((*shape, 3), dtype=np.uint8)
    fn(frame, *args)
    rows = np.nonzero(frame.any(axis=(1, 2)))[0]
    return rows.max() - rows.min() + 1


def test_480_high_frame_draws_exactly_as_before():
    import cv2

    h, w = 480, 640
    expected = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.line(expected, (0, int(0.65 * h)), (w, int(0.65 * h)), (0, 220, 220), 1)
    cv2.circle(expected, (w // 2, h // 2), 4, (240, 240, 240), 1)
    ts = np.arange(0.1, 0.5 + 1e-9, 1 / 60)
    pts = np.array([[int((0.3 * t + 0.2) * w), int(0.5 * h)] for t in ts])
    cv2.polylines(expected, [pts], False, (0, 200, 0), 2)
    cv2.putText(expected, "run 1/2  catches 12  drops 0", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
    cv2.drawMarker(expected, (w // 2, h // 2), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 24, 3)

    frame = np.zeros((h, w, 3), dtype=np.uint8)
    for fn, args in DRAWN:
        fn(frame, *args)
    assert np.array_equal(frame, expected)


def test_phone_frame_draws_proportionally_larger():
    sizes = {fn.__name__: (_drawn_height(fn, args, (480, 640)),
                           _drawn_height(fn, args, (1920, 1080)))
             for fn, args in DRAWN}
    for name in ("draw_hud", "draw_detections", "draw_drop_marker"):
        small, phone = sizes[name]
        assert 2.0 * small <= phone <= 2.5 * small, (name, sizes[name])
    for name in ("draw_hand_line", "draw_arc_tails"):
        small, phone = sizes[name]
        assert phone > small, (name, sizes[name])
