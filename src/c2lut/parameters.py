"""Portable parameter values shared by training and inference."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from ._validation import FloatArray, finite_array, positive_int


@dataclass(frozen=True)
class ModelConfig:
    """Paper-scale defaults; smaller grids can be used for CPU experiments."""

    grid_size: int = 65
    rank: int = 17
    num_bases: int = 12
    hidden_features: int = 32

    def __post_init__(self) -> None:
        positive_int(self.grid_size, "grid_size", 2)
        positive_int(self.rank, "rank", 2)
        positive_int(self.num_bases, "num_bases")
        positive_int(self.hidden_features, "hidden_features")
        if self.rank > self.grid_size:
            raise ValueError("rank must not exceed grid_size")


DEFAULT_CONFIG = ModelConfig()


@dataclass(frozen=True)
class ModelParameters:
    """Validated, owned parameter arrays, in (output, R, G, B) tensor order.

    Copies are read-only so callers cannot accidentally change an inference model
    by mutating the arrays originally supplied to its constructor.
    """

    config: ModelConfig
    cores: FloatArray
    color_factor: FloatArray
    red_factor: FloatArray
    green_factor: FloatArray
    blue_factor: FloatArray
    mlp_weight1: FloatArray
    mlp_bias1: FloatArray
    mlp_weight2: FloatArray
    mlp_bias2: FloatArray

    def __post_init__(self) -> None:
        if not isinstance(self.config, ModelConfig):
            raise ValueError("config must be ModelConfig")
        n, r = self.config.grid_size, self.config.rank
        b, h = self.config.num_bases, self.config.hidden_features
        shapes = {
            "cores": (b, 3, r, r, r),
            "color_factor": (3, 3),
            "red_factor": (n, r),
            "green_factor": (n, r),
            "blue_factor": (n, r),
            "mlp_weight1": (h, 2),
            "mlp_bias1": (h,),
            "mlp_weight2": (b, h),
            "mlp_bias2": (b,),
        }
        for name, shape in shapes.items():
            array = finite_array(getattr(self, name), name)
            if array.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
            owned = array.copy(order="C")
            owned.flags.writeable = False
            object.__setattr__(self, name, owned)

    def arrays(self) -> dict[str, FloatArray]:
        """Return named arrays for portable serialization and training copies."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "config"}

    @property
    def parameter_count(self) -> int:
        return sum(array.size for array in self.arrays().values())

    @classmethod
    def identity(
        cls,
        config: ModelConfig = DEFAULT_CONFIG,
        *,
        seed: int = 0,
        perturbation: float = 1e-3,
    ) -> ModelParameters:
        """Construct an exact identity Tucker model with broken core symmetry.

        An orthonormal basis containing the constant and linear functions gives
        the exact Tucker decomposition without materializing a 65^3 identity
        tensor. Zero-mean basis perturbations cancel under uniform softmax
        weights. Set perturbation=0 to inspect the paper's symmetric start.
        """
        if not np.isfinite(perturbation) or perturbation < 0.0:
            raise ValueError("perturbation must be finite and nonnegative")
        rng = np.random.default_rng(seed)
        n, r, b, h = (
            config.grid_size,
            config.rank,
            config.num_bases,
            config.hidden_features,
        )
        x = np.linspace(0.0, 1.0, n)
        # Random completion is independent of the exact constant/linear span.
        basis = np.column_stack((np.ones(n), x, rng.normal(size=(n, r - 2))))
        factor = np.asarray(np.linalg.qr(basis, mode="reduced")[0])
        constant = factor.T @ np.ones(n)
        linear = factor.T @ x
        core = np.stack(
            [
                np.einsum("i,j,k->ijk", linear, constant, constant),
                np.einsum("i,j,k->ijk", constant, linear, constant),
                np.einsum("i,j,k->ijk", constant, constant, linear),
            ]
        )
        noise = rng.normal(scale=perturbation, size=(b, 3, r, r, r))
        noise -= noise.mean(axis=0, keepdims=True)
        return cls(
            config=config,
            cores=np.asarray(core[None, ...] + noise),
            color_factor=np.eye(3),
            red_factor=factor,
            green_factor=factor,
            blue_factor=factor,
            mlp_weight1=rng.normal(scale=np.sqrt(2.0 / (2 + h)), size=(h, 2)),
            mlp_bias1=np.zeros(h),
            mlp_weight2=np.zeros((b, h)),
            mlp_bias2=np.zeros(b),
        )
