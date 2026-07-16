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


def test_fit_rejects_all_same_timestamp():
    """Regression: real dense detections can put >=3 candidate boxes in a
    single video frame. Shrunk from the array actually captured off
    af2.mp4 at imgsz=960 (4 points, all t=28.85389312...), which crashed
    `numpy.polyfit`'s SVD-based solver with `LinAlgError: SVD did not
    converge` instead of the documented ValueError contract.

    Cause: with every point at the same t, the dt-dependent Vandermonde
    columns (dt^2, dt) are identically zero even after weighting, so
    numpy.polyfit's internal column-scale normalization computes 0/0 and
    feeds a NaN-containing matrix to LAPACK's SVD, which cannot converge.
    A parabola is genuinely unidentifiable from points at a single instant,
    so this must raise the same ValueError as too-few-points, not crash.
    """
    t = 28.85389312
    arr = np.array([
        [t, 0.63215172, 0.59684718, 0.28999031],
        [t, 0.63299447, 0.59711826, 0.06328251],
        [t, 0.63366514, 0.60147285, 0.09570954],
        [t, 0.6382072, 0.59683901, 0.3012276],
    ])
    # Note: np.linalg.LinAlgError is itself a ValueError subclass, so a bare
    # `pytest.raises(ValueError)` would not distinguish the documented,
    # controlled contract from the leaked numpy crash this test guards
    # against -- assert the exact type to make that distinction explicit.
    with pytest.raises(ValueError) as exc_info:
        fit_arc(arr)
    assert not isinstance(exc_info.value, np.linalg.LinAlgError), (
        "fit_arc leaked numpy's LinAlgError instead of raising its own "
        "controlled ValueError before ever calling polyfit"
    )


def test_rmse_downweights_outlier_consistently_with_fit():
    """rmse must reflect the fit's own objective: weights enter squared.

    A near-zero-confidence outlier barely moves the fit (already tested);
    it must also barely move rmse. With linear weights the outlier's
    contribution is ~w*res^2 (=> rmse ~5.5e-3 here); with the correct w^2
    weighting it is ~w^2*res^2 (=> rmse ~5.5e-4).
    """
    arr = flight_points()
    outlier = arr[len(arr) // 2].copy()
    outlier[2] += 0.3
    outlier[3] = 0.01
    arc = fit_arc(np.vstack([arr, outlier]))
    assert arc.rmse < 1e-3
