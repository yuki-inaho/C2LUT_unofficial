"""Replaceable interpolation implementations with one public array contract."""

from __future__ import annotations

from itertools import product
from typing import Protocol

import numpy as np

from ._validation import FloatArray, finite_array, triples


class Interpolator(Protocol):
    """A LUT evaluator, independent of how the LUT was learned or stored."""

    def __call__(self, lut: FloatArray, points: FloatArray) -> FloatArray: ...


def validate_lut(lut: FloatArray) -> FloatArray:
    array = finite_array(lut, "lut")
    if (
        array.ndim != 4
        or array.shape[0] != 3
        or array.shape[1] < 2
        or len(set(array.shape[1:])) != 1
    ):
        raise ValueError("lut must have shape (3, N, N, N) with N >= 2")
    return array


def native_interpolate(lut: FloatArray, points: FloatArray) -> FloatArray:
    """Evaluate the mandatory native backend; never silently substitute Python."""
    from . import _native

    return _native.trilinear(lut, points)


def numpy_interpolate(lut: FloatArray, points: FloatArray) -> FloatArray:
    """Readable NumPy reference backend for auditing or explicit fallback use."""
    table = validate_lut(lut)
    pixels = triples(points, "points")
    if pixels.ndim != 2:
        raise ValueError("points must have shape (M, 3)")
    if np.any(pixels < 0.0) or np.any(pixels > 1.0):
        raise ValueError("points must lie in [0, 1]")
    n = table.shape[1]
    coordinates = pixels * (n - 1)
    lower = np.minimum(np.floor(coordinates).astype(np.int64), n - 2)
    fraction = coordinates - lower
    result = np.zeros_like(pixels)
    for corner in product((0, 1), repeat=3):
        offsets = np.asarray(corner)
        indices = lower + offsets
        weights = np.prod(np.where(offsets, fraction, 1.0 - fraction), axis=1)
        result += weights[:, None] * table[:, indices[:, 0], indices[:, 1], indices[:, 2]].T
    return result
