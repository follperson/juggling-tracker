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
