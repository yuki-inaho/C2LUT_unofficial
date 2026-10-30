# SPDX-License-Identifier: MIT
"""Vectorized colour metrics with explicit shape and numerical conventions.

XYZ values use the scale of the supplied reference white (Y=1 by default).
Negative XYZ values follow the standard linear CIELAB branch; they are not
clipped. Pairwise metrics broadcast leading dimensions, retaining the final
three-component colour axis. Scalar metrics return zero-dimensional arrays.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

_D65_WHITE = (0.95047, 1.0, 1.08883)
_DELTA = 6.0 / 29.0


def _triplets(value: ArrayLike, name: str) -> FloatArray:
    """Validate a real, finite array ending in a three-component axis."""
    try:
        original = np.asarray(value)
        if original.dtype.kind not in "biuf":
            raise ValueError(f"{name} must contain real numeric values")
        array = np.asarray(original, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an array of real numeric values") from error
    if array.ndim < 1 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (..., 3); got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _pair(
    first: ArrayLike, second: ArrayLike, first_name: str, second_name: str
) -> tuple[FloatArray, FloatArray]:
    """Validate both colours and their leading-dimension broadcasting."""
    first_array = _triplets(first, first_name)
    second_array = _triplets(second, second_name)
    try:
        a, b = np.broadcast_arrays(first_array, second_array)
    except ValueError as error:
        raise ValueError(
            f"{first_name} and {second_name} have incompatible shapes: "
            f"{first_array.shape} and {second_array.shape}"
        ) from error
    return np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)


def xyz_to_lab(xyz: ArrayLike, white: ArrayLike = _D65_WHITE) -> FloatArray:
    """Convert CIE XYZ to CIE 1976 L*a*b* without clipping XYZ values.

    Parameters
    ----------
    xyz:
        Finite real values of shape (..., 3). With the default white, XYZ is
        relative to a reference white whose Y is 1, rather than 100.
    white:
        Strictly positive reference-white XYZ of shape (3,), in the same scale
        as ``xyz``. The default is the commonly used D65 2-degree white point.

    Returns
    -------
    numpy.ndarray
        Float64 Lab values with the same shape as ``xyz``. Negative XYZ values
        use the linear branch below (6/29)**3. No chromatic adaptation is done.

    Raises
    ------
    ValueError
        If shapes, finite values, or the reference white are invalid, or if
        the result cannot be represented in float64.
    """
    values = _triplets(xyz, "xyz")
    reference = _triplets(white, "white")
    if reference.shape != (3,) or np.any(reference <= 0.0):
        raise ValueError("white must have shape (3,) and strictly positive values")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            normalized = values / reference
            transformed = np.where(
                normalized > _DELTA**3,
                np.cbrt(normalized),
                normalized / (3.0 * _DELTA**2) + 4.0 / 29.0,
            )
            return np.stack(
                (
                    116.0 * transformed[..., 1] - 16.0,
                    500.0 * (transformed[..., 0] - transformed[..., 1]),
                    200.0 * (transformed[..., 1] - transformed[..., 2]),
                ),
                axis=-1,
            )
    except FloatingPointError as error:
        raise ValueError("XYZ to Lab conversion exceeds the float64 range") from error


def delta_e76(lab1: ArrayLike, lab2: ArrayLike) -> FloatArray:
    """Return Euclidean CIE 1976 colour differences between Lab triplets.

    Inputs have shape (..., 3); their leading dimensions must broadcast.
    The output has the broadcast leading shape and dtype float64. Values are
    Lab coordinates, not XYZ coordinates. The norm avoids squaring large
    components, while unrepresentable differences raise ``ValueError``.
    """
    first, second = _pair(lab1, lab2, "lab1", "lab2")
    try:
        with np.errstate(over="raise", invalid="raise"):
            difference = first - second
            norm = np.hypot(np.hypot(difference[..., 0], difference[..., 1]), difference[..., 2])
            return np.asarray(norm, dtype=np.float64)
    except FloatingPointError as error:
        raise ValueError("Lab difference exceeds the float64 range") from error


def _seventh_chroma_ratio(chroma: FloatArray) -> FloatArray:
    """Evaluate C**7 / (C**7 + 25**7) without high-power overflow."""
    below = np.minimum(chroma / 25.0, 1.0) ** 7
    reciprocal = np.divide(25.0, chroma, out=np.ones_like(chroma), where=chroma > 25.0) ** 7
    return np.where(chroma <= 25.0, below / (1.0 + below), 1.0 / (1.0 + reciprocal))


def _delta_e2000(first: FloatArray, second: FloatArray) -> FloatArray:
    """Evaluate the CIEDE2000 equations for validated, broadcast Lab arrays."""
    light1, a1, b1 = np.moveaxis(first, -1, 0)
    light2, a2, b2 = np.moveaxis(second, -1, 0)
    chroma1 = np.hypot(a1, b1)
    chroma2 = np.hypot(a2, b2)
    mean_chroma = chroma1 / 2.0 + chroma2 / 2.0
    adjustment = 0.5 * (1.0 - np.sqrt(_seventh_chroma_ratio(mean_chroma)))
    a1_prime = (1.0 + adjustment) * a1
    a2_prime = (1.0 + adjustment) * a2
    chroma1_prime = np.hypot(a1_prime, b1)
    chroma2_prime = np.hypot(a2_prime, b2)
    hue1 = np.mod(np.degrees(np.arctan2(b1, a1_prime)), 360.0)
    hue2 = np.mod(np.degrees(np.arctan2(b2, a2_prime)), 360.0)
    hue1 = np.where(chroma1_prime == 0.0, 0.0, hue1)
    hue2 = np.where(chroma2_prime == 0.0, 0.0, hue2)
    achromatic = (chroma1_prime == 0.0) | (chroma2_prime == 0.0)

    delta_light = light2 - light1
    delta_chroma = chroma2_prime - chroma1_prime
    hue_difference = hue2 - hue1
    # Exactly opposite hues are a branch boundary in the published formula.
    # Resolve only floating-point roundoff at 180 degrees, without smoothing
    # the genuine CIEDE2000 discontinuity on either side of that boundary.
    opposite = np.isclose(np.abs(hue_difference), 180.0, rtol=0.0, atol=1e-12)
    hue_difference = np.where(opposite, np.copysign(180.0, hue_difference), hue_difference)
    delta_hue = np.where(
        hue_difference > 180.0,
        hue_difference - 360.0,
        np.where(hue_difference < -180.0, hue_difference + 360.0, hue_difference),
    )
    delta_hue = np.where(achromatic, 0.0, delta_hue)
    delta_hue_term = (
        2.0 * np.sqrt(chroma1_prime) * np.sqrt(chroma2_prime) * np.sin(np.radians(delta_hue / 2.0))
    )

    mean_light = light1 / 2.0 + light2 / 2.0
    mean_chroma_prime = chroma1_prime / 2.0 + chroma2_prime / 2.0
    hue_sum = hue1 + hue2
    mean_hue = np.where(
        np.abs(hue_difference) <= 180.0,
        hue_sum / 2.0,
        np.where(hue_sum < 360.0, (hue_sum + 360.0) / 2.0, (hue_sum - 360.0) / 2.0),
    )
    mean_hue = np.where(achromatic, hue_sum, mean_hue)
    weighting = (
        1.0
        - 0.17 * np.cos(np.radians(mean_hue - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * mean_hue))
        + 0.32 * np.cos(np.radians(3.0 * mean_hue + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * mean_hue - 63.0))
    )
    light_offset = mean_light - 50.0
    light_scale = 1.0 + 0.015 * light_offset**2 / np.hypot(np.sqrt(20.0), light_offset)
    chroma_scale = 1.0 + 0.045 * mean_chroma_prime
    hue_scale = 1.0 + 0.015 * mean_chroma_prime * weighting
    theta = 30.0 * np.exp(-(((mean_hue - 275.0) / 25.0) ** 2))
    rotation = (
        -2.0 * np.sqrt(_seventh_chroma_ratio(mean_chroma_prime)) * np.sin(np.radians(2.0 * theta))
    )
    light_term = delta_light / light_scale
    chroma_term = delta_chroma / chroma_scale
    hue_term = delta_hue_term / hue_scale
    squared = light_term**2 + chroma_term**2 + hue_term**2 + rotation * chroma_term * hue_term
    return np.asarray(np.sqrt(np.maximum(squared, 0.0)), dtype=np.float64)


def delta_e2000(lab1: ArrayLike, lab2: ArrayLike) -> FloatArray:
    """Return CIEDE2000 colour differences with kL=kC=kH=1.

    Inputs are finite Lab values of shape (..., 3), with broadcast-compatible
    leading dimensions. The equations follow Sharma, Wu and Dalal (2005),
    DOI: 10.1002/col.20070. Neither input is clipped. The 180-degree hue branch
    resolves floating-point roundoff within 1e-12 degrees; the formula's
    genuine discontinuity remains intact. Invalid or numerically
    unrepresentable inputs raise ``ValueError``.
    """
    first, second = _pair(lab1, lab2, "lab1", "lab2")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _delta_e2000(first, second)
    except FloatingPointError as error:
        raise ValueError("CIEDE2000 calculation exceeds the float64 range") from error


def angular_error_degrees(xyz1: ArrayLike, xyz2: ArrayLike) -> FloatArray:
    """Return the angle in degrees between corresponding XYZ vectors.

    Leading dimensions broadcast. Both-zero vectors have error 0; exactly
    one zero vector has error 90. These are explicit API conventions for an
    otherwise undefined angle. Nonzero vectors use an atan2 formulation and
    independent scaling to avoid overflow and loss of accuracy near zero
    degrees. Negative coordinates are accepted, so the full range is [0, 180].
    """
    first, second = _pair(xyz1, xyz2, "xyz1", "xyz2")
    first_scale = np.max(np.abs(first), axis=-1, keepdims=True)
    second_scale = np.max(np.abs(second), axis=-1, keepdims=True)
    first_scaled = np.divide(first, first_scale, out=np.zeros_like(first), where=first_scale != 0.0)
    second_scaled = np.divide(
        second, second_scale, out=np.zeros_like(second), where=second_scale != 0.0
    )
    cross_norm = np.linalg.norm(np.cross(first_scaled, second_scaled), axis=-1)
    dot = np.sum(first_scaled * second_scaled, axis=-1)
    angle = np.degrees(np.arctan2(cross_norm, dot))
    first_zero = first_scale[..., 0] == 0.0
    second_zero = second_scale[..., 0] == 0.0
    return np.asarray(np.where(first_zero ^ second_zero, 90.0, angle), dtype=np.float64)
