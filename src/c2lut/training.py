"""Optional PyTorch training for the C2LUT model in arXiv:2607.11681.

Inference does not import this module and does not require PyTorch. This module
keeps the computation differentiable through the illuminant MLP, basis cores,
factor matrices, and trilinear interpolation.

The paper does not specify the reduction used in its smoothness objective. The
default here is the sum of mean squared adjacent differences, averaged over the
batch, channels, and grid. ``frobenius`` implements the literal per-axis L2 norm
instead. Their weights are not interchangeable with the paper's reported 100.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as functional

from .parameters import ModelParameters

if TYPE_CHECKING:
    from .data import PairedDataset

SmoothnessMode = Literal["mean_squared", "frobenius"]


def _numpy_copy(tensor: Tensor) -> NDArray[np.float64]:
    return tensor.detach().cpu().numpy().astype(np.float64, copy=True)


def _require_floating(tensor: Tensor, name: str) -> None:
    if tensor.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must have dtype float32 or float64")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")


def _check_white_points(white_points: Tensor) -> None:
    _require_floating(white_points, "white_points")
    if white_points.ndim != 2 or white_points.shape[1] != 3:
        raise ValueError("white_points must have shape (illuminants, 3)")
    if white_points.shape[0] == 0:
        raise ValueError("at least one illuminant is required")
    if not bool((white_points > 0).all()):
        raise ValueError("all white-point channels must be positive")


def xyz_to_lab_torch(xyz: Tensor) -> Tensor:
    """Convert relative XYZ (D65 white Y=1) to CIE L*a*b*.

    Negative XYZ uses the standard linear branch. The inactive cube-root branch
    is also well-defined, preventing NaN derivatives near black or below zero.
    """
    if xyz.ndim < 1 or xyz.shape[-1] != 3:
        raise ValueError("xyz must have trailing dimension 3")
    _require_floating(xyz, "xyz")
    reference_white = xyz.new_tensor((0.95047, 1.0, 1.08883))
    relative = xyz / reference_white
    delta = 6.0 / 29.0
    threshold = delta**3
    transformed = torch.where(
        relative > threshold,
        relative.clamp_min(threshold).pow(1.0 / 3.0),
        relative / (3.0 * delta**2) + 4.0 / 29.0,
    )
    return torch.stack(
        (
            116.0 * transformed[..., 1] - 16.0,
            500.0 * (transformed[..., 0] - transformed[..., 1]),
            200.0 * (transformed[..., 1] - transformed[..., 2]),
        ),
        dim=-1,
    )


def delta_e76_loss(predicted_xyz: Tensor, target_xyz: Tensor) -> Tensor:
    """Mean CIE Delta E 1976 with a finite zero derivative at exact matches."""
    if predicted_xyz.shape != target_xyz.shape:
        raise ValueError("predicted and target XYZ must have identical shapes")
    if predicted_xyz.numel() == 0:
        raise ValueError("a color loss requires at least one sample")
    difference = xyz_to_lab_torch(predicted_xyz) - xyz_to_lab_torch(target_xyz)
    return cast(Tensor, torch.linalg.vector_norm(difference, dim=-1).mean())


def trilinear_torch(luts: Tensor, points: Tensor) -> Tensor:
    """Differentiably sample batched channel-first LUTs.

    ``luts`` has shape (L, 3, N, N, N), with axes R, G, B, and ``points``
    has shape (L, P, 3). Coordinates outside [0, 1] are clipped. The result
    has shape (L, P, 3). This implementation is independent of the C++ kernel
    and is checked against analytic mappings and finite-difference gradients.
    """
    _require_floating(luts, "luts")
    _require_floating(points, "points")
    if luts.ndim != 5 or luts.shape[1] != 3:
        raise ValueError("luts must have shape (illuminants, 3, N, N, N)")
    resolution = luts.shape[2]
    if resolution < 2 or luts.shape[3:] != (resolution, resolution):
        raise ValueError("LUT grid dimensions must be equal and at least 2")
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape (illuminants, samples, 3)")
    if points.shape[0] != luts.shape[0]:
        raise ValueError("points and luts must have the same illuminant count")
    if points.device != luts.device or points.dtype != luts.dtype:
        raise ValueError("points and luts must have the same device and dtype")

    coordinates = points.clamp(0.0, 1.0) * (resolution - 1)
    lower = coordinates.floor().to(torch.int64).clamp_max(resolution - 2)
    fraction = coordinates - lower.to(points.dtype)
    flat_luts = luts.flatten(start_dim=2)
    output = torch.zeros_like(points)
    for red_corner in (0, 1):
        for green_corner in (0, 1):
            for blue_corner in (0, 1):
                corners = (red_corner, green_corner, blue_corner)
                weights = torch.ones_like(fraction[..., 0])
                for axis, corner in enumerate(corners):
                    axis_fraction = fraction[..., axis]
                    weights = weights * (axis_fraction if corner else 1.0 - axis_fraction)
                index = (
                    (lower[..., 0] + red_corner) * resolution**2
                    + (lower[..., 1] + green_corner) * resolution
                    + lower[..., 2]
                    + blue_corner
                )
                gathered = torch.gather(
                    flat_luts, 2, index.unsqueeze(1).expand(-1, 3, -1)
                ).transpose(1, 2)
                output = output + weights.unsqueeze(-1) * gathered
    return output


def lut_smoothness(luts: Tensor, mode: SmoothnessMode = "mean_squared") -> Tensor:
    """Return an explicitly normalized smoothness objective.

    ``mean_squared`` sums the mean squared adjacent difference for R/G/B.
    ``frobenius`` sums each LUT's per-axis Frobenius norms and averages LUTs.
    Grid spacing is not divided out in either convention.
    """
    if luts.ndim != 5 or luts.shape[1] != 3 or min(luts.shape[2:]) < 2:
        raise ValueError("luts must have shape (L, 3, N, N, N), N >= 2")
    if luts.shape[0] == 0:
        raise ValueError("at least one LUT is required")
    if mode not in ("mean_squared", "frobenius"):
        raise ValueError("unknown smoothness mode")
    terms = []
    for axis in (2, 3, 4):
        difference = torch.diff(luts, dim=axis)
        if mode == "mean_squared":
            terms.append(difference.square().mean())
        else:
            terms.append(torch.linalg.vector_norm(difference.flatten(1), dim=1).mean())
    return torch.stack(terms).sum()


class TrainableC2LUT(nn.Module):
    """C2LUT's learnable parameters with no dependence on the native kernel."""

    def __init__(self, parameters: ModelParameters, *, dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("training requires float32 or float64")
        self.config = parameters.config
        self.cores = nn.Parameter(torch.tensor(parameters.cores.copy(), dtype=dtype))
        self.color_factor = nn.Parameter(torch.tensor(parameters.color_factor.copy(), dtype=dtype))
        self.red_factor = nn.Parameter(torch.tensor(parameters.red_factor.copy(), dtype=dtype))
        self.green_factor = nn.Parameter(torch.tensor(parameters.green_factor.copy(), dtype=dtype))
        self.blue_factor = nn.Parameter(torch.tensor(parameters.blue_factor.copy(), dtype=dtype))
        self.mlp_weight1 = nn.Parameter(torch.tensor(parameters.mlp_weight1.copy(), dtype=dtype))
        self.mlp_bias1 = nn.Parameter(torch.tensor(parameters.mlp_bias1.copy(), dtype=dtype))
        self.mlp_weight2 = nn.Parameter(torch.tensor(parameters.mlp_weight2.copy(), dtype=dtype))
        self.mlp_bias2 = nn.Parameter(torch.tensor(parameters.mlp_bias2.copy(), dtype=dtype))

    def weights(self, white_points: Tensor) -> Tensor:
        """Return the simplex of basis weights from sensor rg chromaticity."""
        _check_white_points(white_points)
        if white_points.dtype != self.cores.dtype or white_points.device != self.cores.device:
            raise ValueError("white points must match model dtype and device")
        scaled = white_points / white_points.amax(dim=1, keepdim=True)
        chromaticity = scaled[:, :2] / scaled.sum(dim=1, keepdim=True)
        hidden = functional.relu(functional.linear(chromaticity, self.mlp_weight1, self.mlp_bias1))
        return functional.softmax(
            functional.linear(hidden, self.mlp_weight2, self.mlp_bias2), dim=-1
        )

    def make_luts(self, white_points: Tensor) -> Tensor:
        """Reconstruct one full LUT per illuminant, following paper Eqs. 6-7."""
        core = torch.einsum("lb,bcrst->lcrst", self.weights(white_points), self.cores)
        color = torch.einsum("oc,lcrst->lorst", self.color_factor, core)
        red = torch.einsum("ir,lorst->loist", self.red_factor, color)
        green = torch.einsum("js,loist->loijt", self.green_factor, red)
        return torch.einsum("kt,loijt->loijk", self.blue_factor, green)

    def forward(self, raw: Tensor, white_points: Tensor) -> Tensor:
        """Map (illuminants, pixels, RGB) to relative XYZ under D65."""
        _require_floating(raw, "raw")
        if raw.ndim != 3 or raw.shape[-1] != 3:
            raise ValueError("raw must have shape (illuminants, samples, 3)")
        if raw.dtype != self.cores.dtype or raw.device != self.cores.device:
            raise ValueError("raw must match model dtype and device")
        luts = self.make_luts(white_points)
        if raw.shape[0] != white_points.shape[0]:
            raise ValueError("raw and white_points must have equal illuminant counts")
        return trilinear_torch(luts, raw / white_points[:, None, :])

    def export_parameters(self) -> ModelParameters:
        """Copy model state to the portable NumPy inference representation."""
        return ModelParameters(
            config=self.config,
            cores=_numpy_copy(self.cores),
            color_factor=_numpy_copy(self.color_factor),
            red_factor=_numpy_copy(self.red_factor),
            green_factor=_numpy_copy(self.green_factor),
            blue_factor=_numpy_copy(self.blue_factor),
            mlp_weight1=_numpy_copy(self.mlp_weight1),
            mlp_bias1=_numpy_copy(self.mlp_bias1),
            mlp_weight2=_numpy_copy(self.mlp_weight2),
            mlp_bias2=_numpy_copy(self.mlp_bias2),
        )


@dataclass(frozen=True)
class TrainingConfig:
    """Explicit training choices for reproducible CPU experiments.

    The small defaults are for synthetic verification, not the published 41M
    experiment. Adam and cosine learning-rate decay follow the paper.
    """

    steps: int = 300
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-6
    batch_illuminants: int = 16
    samples_per_illuminant: int = 256
    smoothness_weight: float = 0.1
    seed: int = 0
    smoothness_mode: SmoothnessMode = "mean_squared"
    validation_interval: int = 10

    def __post_init__(self) -> None:
        for name in (
            "steps",
            "batch_illuminants",
            "samples_per_illuminant",
            "validation_interval",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if (
            not isfinite(self.min_learning_rate)
            or self.min_learning_rate < 0
            or self.min_learning_rate > self.learning_rate
        ):
            raise ValueError("min_learning_rate must be between zero and learning_rate")
        if not isfinite(self.smoothness_weight) or self.smoothness_weight < 0:
            raise ValueError("smoothness_weight must be finite and nonnegative")
        if self.smoothness_mode not in ("mean_squared", "frobenius"):
            raise ValueError("unknown smoothness mode")


@dataclass(frozen=True)
class TrainingRecord:
    """One optimizer step; losses describe the sampled batch before its update."""

    step: int
    learning_rate: float
    training_loss: float
    color_loss: float
    smoothness_loss: float
    validation_delta_e76: float | None


@dataclass(frozen=True)
class TrainingResult:
    """Training evidence; ``fit`` restores the model selected by evaluation loss.

    Metrics use the validation set when provided, otherwise the training set.
    ``best_step=0`` means the initial state remained the best candidate.
    """

    history: tuple[TrainingRecord, ...]
    initial_delta_e76: float
    best_delta_e76: float
    final_delta_e76: float
    best_step: int


@dataclass(frozen=True)
class _TensorPairs:
    raw: Tensor
    white_points: Tensor
    xyz: Tensor


def _as_tensor_pairs(dataset: PairedDataset, model: TrainableC2LUT) -> _TensorPairs:
    def convert(array: NDArray[np.float64]) -> Tensor:
        return torch.tensor(array.copy(), dtype=model.cores.dtype, device=model.cores.device)

    pairs = _TensorPairs(
        raw=convert(dataset.raw),
        white_points=convert(dataset.white_points),
        xyz=convert(dataset.xyz),
    )
    _require_floating(pairs.raw, "raw")
    _require_floating(pairs.xyz, "xyz")
    _check_white_points(pairs.white_points)
    if pairs.raw.ndim != 3 or pairs.raw.shape[-1] != 3:
        raise ValueError("dataset raw must have shape (illuminants, samples, 3)")
    if pairs.xyz.shape != pairs.raw.shape:
        raise ValueError("dataset raw and xyz shapes must agree")
    if pairs.raw.shape[0] != pairs.white_points.shape[0] or pairs.raw.shape[1] == 0:
        raise ValueError("dataset dimensions must agree and samples must be nonempty")
    return pairs


def _evaluate(model: TrainableC2LUT, pairs: _TensorPairs, batch_size: int) -> float:
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, pairs.raw.shape[0], batch_size):
            stop = start + batch_size
            predicted = model(pairs.raw[start:stop], pairs.white_points[start:stop])
            batch_loss = delta_e76_loss(predicted, pairs.xyz[start:stop])
            batch_count = predicted.shape[0] * predicted.shape[1]
            total += float(batch_loss) * batch_count
            count += batch_count
    return total / count


def _snapshot(model: TrainableC2LUT) -> dict[str, Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def fit(
    model: TrainableC2LUT,
    train: PairedDataset,
    validation: PairedDataset | None = None,
    *,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Optimize all parameters and restore the best evaluation checkpoint.

    Sampling uses a private CPU generator; this function does not change global
    random seeds or thread counts. Validation data is never used for optimizer
    gradients. The caller supplies a third, untouched test split when assessing
    generalization. No pickle or framework checkpoint is written here.
    """
    if config is None:
        config = TrainingConfig()
    training = _as_tensor_pairs(train, model)
    evaluation = training if validation is None else _as_tensor_pairs(validation, model)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.steps, eta_min=config.min_learning_rate
    )
    initial = _evaluate(model, evaluation, config.batch_illuminants)
    best_loss = initial
    best_state = _snapshot(model)
    best_step = 0
    records: list[TrainingRecord] = []
    illuminant_count, sample_count, _ = training.raw.shape
    batch_count = min(config.batch_illuminants, illuminant_count)
    sampled_count = min(config.samples_per_illuminant, sample_count)

    for step in range(1, config.steps + 1):
        model.train()
        illuminant_indices = torch.randperm(illuminant_count, generator=generator)[:batch_count].to(
            training.raw.device
        )
        sample_indices = torch.randint(
            sample_count, (batch_count, sampled_count), generator=generator
        ).to(training.raw.device)
        raw = training.raw[illuminant_indices[:, None], sample_indices]
        target = training.xyz[illuminant_indices[:, None], sample_indices]
        white = training.white_points[illuminant_indices]

        optimizer.zero_grad(set_to_none=True)
        luts = model.make_luts(white)
        predicted = trilinear_torch(luts, raw / white[:, None, :])
        color_loss = delta_e76_loss(predicted, target)
        smoothness = lut_smoothness(luts, config.smoothness_mode)
        loss = color_loss + config.smoothness_weight * smoothness
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"nonfinite loss at step {step}")
        torch.autograd.backward(loss)
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(f"missing or nonfinite gradient for {name}")
        learning_rate = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

        validation_loss = None
        if step % config.validation_interval == 0 or step == config.steps:
            model.eval()
            validation_loss = _evaluate(model, evaluation, config.batch_illuminants)
            if not isfinite(validation_loss):
                raise FloatingPointError(f"nonfinite evaluation loss at step {step}")
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_step = step
                best_state = _snapshot(model)
        records.append(
            TrainingRecord(
                step=step,
                learning_rate=learning_rate,
                training_loss=float(loss.detach()),
                color_loss=float(color_loss.detach()),
                smoothness_loss=float(smoothness.detach()),
                validation_delta_e76=validation_loss,
            )
        )

    model.load_state_dict(best_state)
    model.eval()
    final_loss = _evaluate(model, evaluation, config.batch_illuminants)
    return TrainingResult(
        history=tuple(records),
        initial_delta_e76=initial,
        best_delta_e76=best_loss,
        final_delta_e76=final_loss,
        best_step=best_step,
    )
