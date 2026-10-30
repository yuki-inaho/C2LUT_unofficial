# SPDX-License-Identifier: MIT
"""Independent numerical references for colour conversion and metrics."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

from c2lut.color import angular_error_degrees, delta_e76, delta_e2000, xyz_to_lab

FloatArray = NDArray[np.float64]
Metric = Callable[[ArrayLike, ArrayLike], FloatArray]

# Independent published reference values, rather than outputs from this package.
# Sharma, Wu and Dalal (2005), DOI: 10.1002/col.20070.
# Source: https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/
# dataNprograms/ciede2000testdata.txt (34 pairs; expected results rounded to 4 dp).
_SHARMA_DATA = np.array(
    [
        [50.0000, 2.6772, -79.7751, 50.0000, 0.0000, -82.7485, 2.0425],
        [50.0000, 3.1571, -77.2803, 50.0000, 0.0000, -82.7485, 2.8615],
        [50.0000, 2.8361, -74.0200, 50.0000, 0.0000, -82.7485, 3.4412],
        [50.0000, -1.3802, -84.2814, 50.0000, 0.0000, -82.7485, 1.0000],
        [50.0000, -1.1848, -84.8006, 50.0000, 0.0000, -82.7485, 1.0000],
        [50.0000, -0.9009, -85.5211, 50.0000, 0.0000, -82.7485, 1.0000],
        [50.0000, 0.0000, 0.0000, 50.0000, -1.0000, 2.0000, 2.3669],
        [50.0000, -1.0000, 2.0000, 50.0000, 0.0000, 0.0000, 2.3669],
        [50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0009, 7.1792],
        [50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0010, 7.1792],
        [50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0011, 7.2195],
        [50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0012, 7.2195],
        [50.0000, -0.0010, 2.4900, 50.0000, 0.0009, -2.4900, 4.8045],
        [50.0000, -0.0010, 2.4900, 50.0000, 0.0010, -2.4900, 4.8045],
        [50.0000, -0.0010, 2.4900, 50.0000, 0.0011, -2.4900, 4.7461],
        [50.0000, 2.5000, 0.0000, 50.0000, 0.0000, -2.5000, 4.3065],
        [50.0000, 2.5000, 0.0000, 73.0000, 25.0000, -18.0000, 27.1492],
        [50.0000, 2.5000, 0.0000, 61.0000, -5.0000, 29.0000, 22.8977],
        [50.0000, 2.5000, 0.0000, 56.0000, -27.0000, -3.0000, 31.9030],
        [50.0000, 2.5000, 0.0000, 58.0000, 24.0000, 15.0000, 19.4535],
        [50.0000, 2.5000, 0.0000, 50.0000, 3.1736, 0.5854, 1.0000],
        [50.0000, 2.5000, 0.0000, 50.0000, 3.2972, 0.0000, 1.0000],
        [50.0000, 2.5000, 0.0000, 50.0000, 1.8634, 0.5757, 1.0000],
        [50.0000, 2.5000, 0.0000, 50.0000, 3.2592, 0.3350, 1.0000],
        [60.2574, -34.0099, 36.2677, 60.4626, -34.1751, 39.4387, 1.2644],
        [63.0109, -31.0961, -5.8663, 62.8187, -29.7946, -4.0864, 1.2630],
        [61.2901, 3.7196, -5.3901, 61.4292, 2.2480, -4.9620, 1.8731],
        [35.0831, -44.1164, 3.7933, 35.0232, -40.0716, 1.5901, 1.8645],
        [22.7233, 20.0904, -46.6940, 23.0331, 14.9730, -42.5619, 2.0373],
        [36.4612, 47.8580, 18.3852, 36.2715, 50.5065, 21.2231, 1.4146],
        [90.8027, -2.0831, 1.4410, 91.1528, -1.6435, 0.0447, 1.4441],
        [90.9257, -0.5406, -0.9208, 88.6381, -0.8985, -0.7239, 1.5381],
        [6.7747, -0.2908, -2.4247, 5.8714, -0.0985, -2.2286, 0.6377],
        [2.0776, 0.0795, -1.1350, 0.9033, -0.0636, -0.5514, 0.9082],
    ],
    dtype=np.float64,
)


def test_xyz_black_white_and_mid_grey() -> None:
    white = np.array([0.95047, 1.0, 1.08883])
    inputs = np.array([np.zeros(3), white, 0.125 * white])
    expected = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [42.0, 0.0, 0.0]])
    np.testing.assert_allclose(xyz_to_lab(inputs), expected, rtol=0.0, atol=1e-12)


def test_xyz_negative_values_use_linear_branch_without_clipping() -> None:
    normalized = np.array([-0.2, -0.1, -0.05])
    kappa = 24389.0 / 27.0
    expected = np.array(
        [
            kappa * normalized[1],
            (500.0 / 116.0) * kappa * (normalized[0] - normalized[1]),
            (200.0 / 116.0) * kappa * (normalized[1] - normalized[2]),
        ]
    )
    np.testing.assert_allclose(
        xyz_to_lab(normalized, white=(1.0, 1.0, 1.0)), expected, rtol=0.0, atol=1e-12
    )


def test_xyz_linear_nonlinear_boundary_and_white_scale() -> None:
    epsilon = 216.0 / 24389.0
    np.testing.assert_allclose(
        xyz_to_lab([epsilon, epsilon, epsilon], white=[1.0, 1.0, 1.0]),
        [8.0, 0.0, 0.0],
        rtol=0.0,
        atol=1e-12,
    )
    xyz = np.array([0.2, 0.3, 0.5])
    white = np.array([0.95047, 1.0, 1.08883])
    np.testing.assert_allclose(xyz_to_lab(xyz * 100.0, white * 100.0), xyz_to_lab(xyz))


@pytest.mark.parametrize(
    "white",
    [[1.0, 0.0, 1.0], [1.0, -1.0, 1.0], [1.0, np.nan, 1.0], [[1.0, 1.0, 1.0]]],
)
def test_xyz_rejects_invalid_reference_white(white: ArrayLike) -> None:
    with pytest.raises(ValueError):
        xyz_to_lab([0.1, 0.2, 0.3], white=white)


def test_delta_e76_matches_pythagoras_and_handles_large_finite_values() -> None:
    np.testing.assert_array_equal(delta_e76([50, 3, 4], [50, 0, 0]), np.array(5.0))
    np.testing.assert_allclose(delta_e76([0.0, 3e200, 4e200], [0.0, 0.0, 0.0]), 5e200, rtol=1e-15)


@pytest.mark.parametrize("row", _SHARMA_DATA, ids=[f"Sharma-{i:02d}" for i in range(1, 35)])
def test_delta_e2000_published_reference_pairs(row: FloatArray) -> None:
    actual = delta_e2000(row[:3], row[3:6])
    np.testing.assert_allclose(actual, row[6], rtol=0.0, atol=5e-5)


def test_delta_e2000_is_symmetric_and_zero_on_identical_colours() -> None:
    first = _SHARMA_DATA[:, :3]
    second = _SHARMA_DATA[:, 3:6]
    np.testing.assert_allclose(
        delta_e2000(first, second), delta_e2000(second, first), rtol=0.0, atol=1e-12
    )
    np.testing.assert_array_equal(delta_e2000(first, first), np.zeros(len(first)))


def test_angular_errors_and_explicit_zero_conventions() -> None:
    first = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0]])
    second = np.array([[1, 1, 0], [0, 1, 0], [-1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0]])
    expected = np.array([45.0, 90.0, 180.0, 0.0, 90.0, 90.0])
    np.testing.assert_allclose(angular_error_degrees(first, second), expected, atol=1e-12)


def test_angular_error_is_stable_under_extreme_rescaling() -> None:
    first = np.array([1.0, 2.0, 3.0])
    second = np.array([2.0, 3.0, 1.0])
    expected = np.degrees(np.arccos(11.0 / 14.0))
    np.testing.assert_allclose(
        angular_error_degrees(first * 1e250, second * 1e-250), expected, atol=1e-12
    )
    np.testing.assert_array_equal(angular_error_degrees(first, first), np.array(0.0))


@pytest.mark.parametrize("metric", [delta_e76, delta_e2000, angular_error_degrees])
def test_metrics_broadcast_and_preserve_inputs(metric: Metric) -> None:
    first = np.array([50.0, 3.0, 4.0])
    second = np.ones((2, 1, 3), dtype=np.float64)
    first_copy = first.copy()
    second_copy = second.copy()
    first.flags.writeable = False
    second.flags.writeable = False
    result = metric(first, second)
    assert result.shape == (2, 1)
    assert result.dtype == np.float64
    assert isinstance(metric(first, first), np.ndarray)
    assert metric(first, first).shape == ()
    np.testing.assert_array_equal(first, first_copy)
    np.testing.assert_array_equal(second, second_copy)


@pytest.mark.parametrize("metric", [delta_e76, delta_e2000, angular_error_degrees])
def test_metrics_accept_empty_batches(metric: Metric) -> None:
    assert metric(np.empty((0, 3)), [1.0, 2.0, 3.0]).shape == (0,)
    assert xyz_to_lab(np.empty((2, 0, 3))).shape == (2, 0, 3)


@pytest.mark.parametrize("metric", [delta_e76, delta_e2000, angular_error_degrees])
@pytest.mark.parametrize(
    "invalid",
    [
        0.0,
        [1.0, 2.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.0, np.nan, 2.0],
        [1.0, np.inf, 2.0],
    ],
)
def test_metrics_reject_bad_shapes_and_nonfinite_values(metric: Metric, invalid: ArrayLike) -> None:
    with pytest.raises(ValueError):
        metric(invalid, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        metric([1.0, 2.0, 3.0], invalid)
    with pytest.raises(ValueError):
        xyz_to_lab(invalid)


@pytest.mark.parametrize("metric", [delta_e76, delta_e2000, angular_error_degrees])
def test_metrics_reject_incompatible_batches(metric: Metric) -> None:
    with pytest.raises(ValueError, match="incompatible shapes"):
        metric(np.ones((2, 3)), np.ones((4, 3)))


@pytest.mark.parametrize("invalid", [[1.0j, 2.0, 3.0], ["1", "2", "3"]])
def test_reject_nonreal_or_nonnumeric_input(invalid: ArrayLike) -> None:
    with pytest.raises(ValueError):
        xyz_to_lab(invalid)
    with pytest.raises(ValueError):
        delta_e2000(invalid, [1.0, 2.0, 3.0])


def test_unrepresentable_lab_differences_raise_clear_error() -> None:
    with pytest.raises(ValueError, match="float64"):
        delta_e76([1.7e308, 0.0, 0.0], [-1.7e308, 0.0, 0.0])
