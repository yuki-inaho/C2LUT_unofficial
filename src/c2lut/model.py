"""NumPy inference orchestration; PyTorch is intentionally not imported."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, chromaticity, triples, white_point
from .interpolation import Interpolator, native_interpolate, validate_lut
from .parameters import ModelParameters
from .serialization import load_parameters, save_parameters

RangePolicy = Literal["clip", "raise"]


@dataclass(frozen=True)
class PreparedTransform:
    """One illuminant's LUT, reusable over multiple frames without regeneration."""

    lut: FloatArray
    white: FloatArray
    interpolator: Interpolator = field(default=native_interpolate, repr=False)

    def __post_init__(self) -> None:
        table = validate_lut(self.lut).copy()
        white = white_point(self.white).copy()
        table.flags.writeable = False
        white.flags.writeable = False
        object.__setattr__(self, "lut", table)
        object.__setattr__(self, "white", white)

    def __call__(self, raw: ArrayLike, *, out_of_range: RangePolicy = "clip") -> FloatArray:
        """Convert linear camera RGB (..., 3) into reference XYZ (..., 3).

        The white point must share the raw values' radiometric scale. Values
        after white balancing are clipped by default, or rejected explicitly.
        Outputs are never clipped or gamma encoded.
        """
        if out_of_range not in ("clip", "raise"):
            raise ValueError("out_of_range must be 'clip' or 'raise'")
        pixels = triples(raw, "raw")
        with np.errstate(over="ignore", invalid="ignore"):
            balanced = pixels.reshape(-1, 3) / self.white
        if not np.isfinite(balanced).all():
            raise ValueError("white balancing produced nonfinite values")
        if out_of_range == "raise" and (np.any(balanced < 0) or np.any(balanced > 1)):
            raise ValueError("white-balanced RGB lies outside [0, 1]")
        points = np.ascontiguousarray(np.clip(balanced, 0.0, 1.0))
        result = self.interpolator(self.lut, points)
        return result.reshape(pixels.shape)


class C2LUT:
    """Illuminant-adaptive color transform with an interchangeable evaluator."""

    def __init__(
        self,
        parameters: ModelParameters,
        *,
        interpolator: Interpolator = native_interpolate,
    ) -> None:
        self.parameters = parameters
        self._interpolator = interpolator

    def weights(self, white_points: ArrayLike) -> FloatArray:
        """Compute softmax basis weights from camera rg chromaticity (Eq. 5)."""
        features = chromaticity(white_points)
        parameters = self.parameters
        hidden = np.maximum(features @ parameters.mlp_weight1.T + parameters.mlp_bias1, 0.0)
        logits = hidden @ parameters.mlp_weight2.T + parameters.mlp_bias2
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        exponential = np.exp(logits)
        return np.asarray(exponential / exponential.sum(axis=-1, keepdims=True))

    def make_lut(self, white: ArrayLike) -> FloatArray:
        """Combine cores and reconstruct one (3, N, N, N) LUT (Eqs. 6-7)."""
        parameters = self.parameters
        weights = self.weights(white_point(white))
        core = np.tensordot(weights, parameters.cores, axes=(0, 0))
        # Sequential mode products bound intermediate storage and expose axes.
        table = np.einsum("oi,iabc->oabc", parameters.color_factor, core)
        table = np.einsum("ra,oabc->orbc", parameters.red_factor, table)
        table = np.einsum("gb,orbc->orgc", parameters.green_factor, table)
        table = np.einsum("kc,orgc->orgk", parameters.blue_factor, table)
        return np.ascontiguousarray(table)

    def prepare(self, white: ArrayLike) -> PreparedTransform:
        """Cache one global illuminant's transform explicitly."""
        point = white_point(white)
        return PreparedTransform(self.make_lut(point), point, self._interpolator)

    def predict(
        self,
        raw: ArrayLike,
        white: ArrayLike,
        *,
        out_of_range: RangePolicy = "clip",
    ) -> FloatArray:
        return self.prepare(white)(raw, out_of_range=out_of_range)

    def save(self, path: str | Path) -> None:
        save_parameters(self.parameters, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        interpolator: Interpolator = native_interpolate,
    ) -> C2LUT:
        return cls(load_parameters(path), interpolator=interpolator)
