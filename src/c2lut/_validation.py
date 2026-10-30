"""Shared numerical contracts at Python API boundaries."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def finite_array(value: ArrayLike, name: str) -> FloatArray:
    """Convert real numeric inputs to finite, C-contiguous double precision."""
    source = np.asarray(value)
    if source.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numbers")
    array = np.asarray(source, dtype=np.float64, order="C")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def triples(value: ArrayLike, name: str) -> FloatArray:
    """Validate an array whose last dimension contains three color channels."""
    array = finite_array(value, name)
    if array.ndim < 1 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (..., 3)")
    return array


def white_point(value: ArrayLike) -> FloatArray:
    """Validate one strictly positive camera white point, with no normalization."""
    array = triples(value, "white_point")
    if array.shape != (3,) or np.any(array <= 0.0):
        raise ValueError("white_point must have shape (3,) with positive components")
    return array


def chromaticity(value: ArrayLike) -> FloatArray:
    """Return camera rg chromaticities without overflowing large white points."""
    array = triples(value, "white_points")
    if np.any(array <= 0.0):
        raise ValueError("white_points must have positive components")
    scaled = array / np.max(array, axis=-1, keepdims=True)
    return np.asarray(scaled[..., :2] / scaled.sum(axis=-1, keepdims=True))


def positive_int(value: int, name: str, minimum: int = 1) -> None:
    """Reject booleans and non-integral configuration values early."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
