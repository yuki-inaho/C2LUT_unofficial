"""Independent analytic tests of the actual nanobind extension."""

from __future__ import annotations

import gc
from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from c2lut import _native


def grid(resolution: int) -> NDArray[np.float64]:
    axis = np.linspace(0.0, 1.0, resolution)
    return np.ascontiguousarray(np.stack(np.meshgrid(axis, axis, axis, indexing="ij")))


@pytest.mark.parametrize("resolution", [2, 3, 9, 65])
def test_identity_and_all_corners(resolution: int) -> None:
    points = np.concatenate(
        [
            np.asarray(list(product([0.0, 1.0], repeat=3))),
            np.random.default_rng(47).random((300, 3)),
        ]
    )
    assert_allclose(_native.trilinear(grid(resolution), points), points, atol=4e-16, rtol=1e-15)


def test_affine_transform_with_negative_outputs() -> None:
    matrix = np.array([[0.9, 0.2, -0.3], [-0.2, 1.1, 0.1], [0.4, -0.1, 0.7]])
    offset = np.array([0.05, -0.3, 0.1])
    coordinates = grid(5)
    lut = np.einsum("ij,jabc->iabc", matrix, coordinates) + offset[:, None, None, None]
    points = np.random.default_rng(16).random((83, 3))
    assert_allclose(
        _native.trilinear(np.ascontiguousarray(lut), points),
        points @ matrix.T + offset,
        atol=6e-16,
        rtol=2e-15,
    )


def test_multilinear_analytic_function() -> None:
    # A trilinear interpolant reproduces these mixed terms exactly, on any grid.
    def transform(values: NDArray[np.float64]) -> NDArray[np.float64]:
        red, green, blue = values
        return np.stack(
            [red * green + 0.2 * blue, green * blue - red * blue, red * green * blue + 0.3]
        )

    lut = np.ascontiguousarray(transform(grid(7)))
    points = np.random.default_rng(23).random((113, 3))
    assert_allclose(_native.trilinear(lut, points), transform(points.T).T, atol=4e-16, rtol=2e-15)


def test_random_lut_against_independent_vertex_sum() -> None:
    rng = np.random.default_rng(107)
    lut = rng.normal(size=(3, 6, 6, 6))
    points = rng.random((19, 3))
    expected = np.zeros_like(points)
    for index, point in enumerate(points):
        coordinate = 5 * point
        low = np.floor(coordinate).astype(np.int64)
        fraction = coordinate - low
        for corner in product((0, 1), repeat=3):
            step = np.array(corner)
            weight = np.prod(np.where(step == 1, fraction, 1 - fraction))
            red, green, blue = low + step
            expected[index] += weight * lut[:, red, green, blue]
    assert_allclose(_native.trilinear(lut, points), expected, atol=3e-16, rtol=3e-15)


def test_grid_nodes_and_adjacent_values() -> None:
    rng = np.random.default_rng(64)
    lut = rng.normal(size=(3, 5, 5, 5))
    nodes = grid(5).reshape(3, -1).T.copy()
    assert_allclose(_native.trilinear(lut, nodes), lut.reshape(3, -1).T, atol=0, rtol=0)
    points = np.array([[np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0), 1.0]])
    assert_allclose(_native.trilinear(grid(5), points), points, atol=1e-15, rtol=1e-15)


def test_empty_input() -> None:
    result = _native.trilinear(grid(3), np.empty((0, 3)))
    assert result.shape == (0, 3)
    assert result.dtype == np.float64


def test_result_lifetime_is_independent_of_inputs() -> None:
    points = np.array([[0.4, 0.3, 0.2]])
    lut = grid(3)
    output = _native.trilinear(lut, points)
    lut.fill(-90)
    points.fill(-90)
    del lut, points
    gc.collect()
    assert_allclose(output, [[0.4, 0.3, 0.2]], atol=2e-16, rtol=1e-15)
    # The capsule-backed array must also remain writable after the inputs die.
    output[0, 0] = 1.5
    assert output[0, 0] == 1.5


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -0.001, 1.001])
def test_invalid_point_values(value: float) -> None:
    with pytest.raises(ValueError, match="points"):
        _native.trilinear(grid(3), np.array([[value, 0.2, 0.5]]))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_lut_values(value: float) -> None:
    lut = grid(3)
    lut[0, 1, 1, 1] = value
    with pytest.raises(ValueError, match="LUT"):
        _native.trilinear(lut, np.zeros((1, 3)))


@pytest.mark.parametrize("shape", [(3, 2, 2), (2, 2, 2, 3), (3, 2, 3, 2), (3, 1, 1, 1)])
def test_invalid_lut_shapes(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        _native.trilinear(np.zeros(shape), np.zeros((1, 3)))


@pytest.mark.parametrize("shape", [(3,), (1, 2), (1, 1, 3)])
def test_invalid_points_shapes(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="points"):
        _native.trilinear(grid(3), np.zeros(shape))


def test_dtype_and_layout_are_not_silently_converted() -> None:
    with pytest.raises(TypeError):
        _native.trilinear(grid(3).astype(np.float32), np.zeros((1, 3)))
    with pytest.raises(TypeError):
        _native.trilinear(grid(3), np.zeros((1, 3), dtype=np.float32))
    with pytest.raises(TypeError):
        _native.trilinear(grid(3), np.zeros((6, 3))[::2])
    with pytest.raises(TypeError):
        _native.trilinear(grid(3)[:, ::-1], np.zeros((1, 3)))


def test_unaligned_buffers_are_rejected() -> None:
    buffer = np.zeros(3 * np.dtype(np.float64).itemsize + 1, dtype=np.uint8)
    points = np.ndarray((1, 3), dtype=np.float64, buffer=buffer, offset=1)
    with pytest.raises(ValueError, match="aligned"):
        _native.trilinear(grid(3), points)


def test_readonly_inputs_are_supported() -> None:
    lut = grid(3)
    points = np.array([[0.3, 0.5, 0.7]])
    lut.flags.writeable = False
    points.flags.writeable = False
    assert_allclose(_native.trilinear(lut, points), points, atol=2e-16, rtol=1e-15)
