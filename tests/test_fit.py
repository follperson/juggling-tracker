import numpy as np
import pytest

from juggletrack.arcs.fit import fit_arc, points_array, y_residuals
from juggletrack.sim import CascadeParams, simulate_cascade


def flight_points(noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """Sample one clean flight from the simulator's first throw."""
    r = simulate_cascade(n_throws=1, fps=60.0, noise=noise, seed=seed)
    return points_array(r.detections)


def test_fit_recovers_gravity():
    arr = flight_points()
    arc = fit_arc(arr, arc_id=7)
    p = CascadeParams()
    assert arc.id == 7
    assert arc.ay == pytest.approx(p.g / 2.0, rel=1e-3)   # ay = g/2
    assert arc.rmse < 1e-6
    assert arc.n_points == len(arr)


def test_fit_recovers_apex_and_endpoints():
    arr = flight_points()
    arc = fit_arc(arr)
    p = CascadeParams()
    assert arc.apex_y() == pytest.approx(p.hand_y - p.v0**2 / (2 * p.g), abs=1e-3)
    assert arc.y_at(arc.t_start) == pytest.approx(p.hand_y, abs=0.01)


def test_fit_with_noise_rmse_tracks_noise():
    arr = flight_points(noise=0.005, seed=2)
    arc = fit_arc(arr)
    assert 0.001 < arc.rmse < 0.015
    p = CascadeParams()
    assert arc.ay == pytest.approx(p.g / 2.0, rel=0.15)


def test_confidence_weighting_downweights_outlier():
    arr = flight_points()
    outlier = arr[len(arr) // 2].copy()
    outlier[2] += 0.3          # push y far off the parabola
    outlier[3] = 0.01          # ...but with near-zero confidence
    arr_w = np.vstack([arr, outlier])
    arc = fit_arc(arr_w)
    assert arc.ay == pytest.approx(CascadeParams().g / 2.0, rel=0.02)


def test_residuals_flag_the_outlier():
    arr = flight_points()
    outlier = arr[10].copy()
    outlier[2] += 0.2
    arr2 = np.vstack([arr, outlier])
    arc = fit_arc(arr)  # fit without the outlier
    res = y_residuals(arc, arr2)
    assert res[-1] > 0.15 and res[:-1].max() < 0.01


def test_fit_requires_three_points():
    with pytest.raises(ValueError):
        fit_arc(np.array([[0.0, 0.5, 0.5, 1.0], [0.1, 0.5, 0.5, 1.0]]))
