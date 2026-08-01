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
