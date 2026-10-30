"""Independent numerical checks for differentiable C2LUT training."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from numpy.typing import NDArray

from c2lut.data import PairedDataset
from c2lut.parameters import ModelConfig, ModelParameters

torch = pytest.importorskip("torch")

# Training is optional; skip before importing its framework dependency.
from c2lut.training import (  # noqa: E402
    TrainableC2LUT,
    TrainingConfig,
    delta_e76_loss,
    fit,
    lut_smoothness,
    trilinear_torch,
    xyz_to_lab_torch,
)


@pytest.fixture(scope="module", autouse=True)
def single_thread_torch() -> Iterator[None]:
    """Tiny numerical fixtures do not benefit from a large BLAS thread pool."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _identity_luts(resolution: int, batch: int = 1) -> torch.Tensor:
    coordinates = torch.linspace(0.0, 1.0, resolution, dtype=torch.float64)
    grid = torch.stack(torch.meshgrid(coordinates, coordinates, coordinates, indexing="ij"))
    return grid.unsqueeze(0).repeat(batch, 1, 1, 1, 1)


def _parameters(*, perturbation: float = 1e-3) -> ModelParameters:
    return ModelParameters.identity(
        ModelConfig(grid_size=5, rank=3, num_bases=3, hidden_features=12),
        seed=18,
        perturbation=perturbation,
    )


def test_trilinear_matches_known_multilinear_polynomial() -> None:
    red, green, blue = _identity_luts(5)[0]
    lut = torch.stack(
        (red * green + 0.2 * blue, 0.4 * red + green * blue, red * green * blue)
    ).unsqueeze(0)
    points = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.13, 0.61, 0.79]]],
        dtype=torch.float64,
    )
    red_points, green_points, blue_points = points.unbind(dim=-1)
    expected = torch.stack(
        (
            red_points * green_points + 0.2 * blue_points,
            0.4 * red_points + green_points * blue_points,
            red_points * green_points * blue_points,
        ),
        dim=-1,
    )
    torch.testing.assert_close(trilinear_torch(lut, points), expected, atol=2e-15, rtol=0)


def test_trilinear_gradients_match_finite_differences() -> None:
    generator = torch.Generator().manual_seed(43)
    luts = torch.rand((1, 3, 3, 3, 3), generator=generator, dtype=torch.float64)
    luts.requires_grad_()
    points = torch.tensor(
        [[[0.13, 0.27, 0.67], [0.82, 0.31, 0.91]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(trilinear_torch, (luts, points), eps=1e-6, atol=2e-5)


def test_trilinear_boundaries_clipping_and_empty_samples() -> None:
    luts = _identity_luts(4)
    points = torch.tensor([[[-2.0, 0.5, 9.0], [1.0, 0.0, 0.75]]], dtype=torch.float64)
    torch.testing.assert_close(trilinear_torch(luts, points), points.clamp(0.0, 1.0))
    empty = torch.empty((1, 0, 3), dtype=torch.float64)
    assert trilinear_torch(luts, empty).shape == (1, 0, 3)


def test_lab_values_and_gradients_at_black_and_negative_xyz() -> None:
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [0.95047, 1.0, 1.08883], [-0.01, 0.002, 0.01]],
        dtype=torch.float64,
        requires_grad=True,
    )
    lab = xyz_to_lab_torch(xyz)
    torch.testing.assert_close(lab[:2], xyz.new_tensor([[0, 0, 0], [100, 0, 0]]))
    lab.square().sum().backward()
    assert xyz.grad is not None
    assert bool(torch.isfinite(xyz.grad).all())


def test_zero_color_loss_has_finite_zero_gradient() -> None:
    xyz = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.4, 0.5]], dtype=torch.float64)
    prediction = xyz.clone().requires_grad_()
    loss = delta_e76_loss(prediction, xyz)
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert prediction.grad is not None
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))


def test_smoothness_reductions_have_analytic_scale() -> None:
    resolution = 5
    luts = _identity_luts(resolution, batch=2)
    assert float(lut_smoothness(luts)) == pytest.approx(1.0 / (resolution - 1) ** 2)
    assert float(lut_smoothness(luts, "frobenius")) == pytest.approx(
        3.0 * resolution / np.sqrt(resolution - 1)
    )
    constant = torch.ones_like(luts, requires_grad=True)
    objective = lut_smoothness(constant, "frobenius")
    objective.backward()
    assert constant.grad is not None
    assert bool(torch.isfinite(constant.grad).all())


def test_identity_weights_and_parameter_export_are_consistent() -> None:
    parameters = _parameters()
    model = TrainableC2LUT(parameters)
    white = torch.tensor([[0.5, 1.2, 0.8], [1.6, 0.7, 1.0]], dtype=torch.float64)
    points = torch.tensor(
        [[[0.1, 0.2, 0.7], [1.0, 1.0, 1.0]], [[0.3, 0.7, 0.2], [0.0, 0.0, 0.0]]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(model(points * white[:, None, :], white), points, atol=2e-14, rtol=0)
    torch.testing.assert_close(model.weights(white).sum(-1), torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(model.weights(white), model.weights(white * 4.0))
    exported = model.export_parameters()
    for name, array in parameters.arrays().items():
        np.testing.assert_array_equal(exported.arrays()[name], array)
    with torch.no_grad():
        model.cores.add_(0.1)
    np.testing.assert_array_equal(exported.cores, parameters.cores)


def test_all_parameter_groups_receive_gradients() -> None:
    model = TrainableC2LUT(_parameters(perturbation=0.1))
    generator = torch.Generator().manual_seed(90)
    with torch.no_grad():
        model.mlp_weight2.normal_(0.0, 0.2, generator=generator)
    white = torch.tensor([[0.5, 1.2, 0.8], [1.6, 0.7, 1.0]], dtype=torch.float64)
    points = torch.rand((2, 9, 3), generator=generator, dtype=torch.float64) * 0.8 + 0.1
    target = points * points.new_tensor([0.8, 1.0, 0.9])
    loss = delta_e76_loss(model(points * white[:, None, :], white), target)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name
        assert float(parameter.grad.abs().sum()) > 0.0, name


def test_symmetric_initialization_blocks_illumination_learning() -> None:
    white = torch.tensor([[0.5, 1.2, 0.8], [1.6, 0.7, 1.0]], dtype=torch.float64)
    points = torch.full((2, 5, 3), 0.3, dtype=torch.float64)
    output_gradients = []
    for perturbation in (0.0, 1e-3):
        model = TrainableC2LUT(_parameters(perturbation=perturbation))
        loss = delta_e76_loss(model(points * white[:, None, :], white), points * 0.8)
        loss.backward()
        assert model.mlp_weight2.grad is not None
        output_gradients.append(float(model.mlp_weight2.grad.abs().sum()))
    assert output_gradients[0] < 1e-11
    assert output_gradients[1] > 1e-8


def _analytic_pairs(seed: int, whites: NDArray[np.float64]) -> PairedDataset:
    """A teacher defined directly by polynomials, independent of LUT code."""
    rng = np.random.default_rng(seed)
    points = rng.uniform(0.08, 0.92, size=(48, 3))
    illuminant_term = (whites[:, 0] - whites[:, 2]) / whites.sum(axis=1)
    red, green, blue = points.T
    xyz = np.stack(
        (
            red[None, :] * (0.8 + 0.3 * illuminant_term[:, None]) + 0.1 * green * blue,
            green[None, :] * (1.0 - 0.2 * illuminant_term[:, None]) + 0.04 * red,
            blue[None, :] * (0.9 - 0.3 * illuminant_term[:, None]) + 0.05 * red * green,
        ),
        axis=-1,
    )
    return PairedDataset(
        raw=points[None, :, :] * whites[:, None, :],
        white_points=whites,
        xyz=xyz,
        illuminant_ids=np.arange(whites.shape[0], dtype=np.int64) + seed * 100,
        reflectance_ids=np.arange(points.shape[0], dtype=np.int64) + seed * 100,
    )


def test_fit_learns_held_out_illumination_dependent_transform() -> None:
    values = np.linspace(0.4, 1.6, 9)
    train_whites = np.stack((values, np.ones_like(values), 2.0 - values), axis=-1)
    values = np.linspace(0.45, 1.55, 8)
    validation_whites = np.stack((values, np.ones_like(values), 2.0 - values), axis=-1)
    train = _analytic_pairs(19, train_whites)
    validation = _analytic_pairs(45, validation_whites)
    assert not set(train.illuminant_ids) & set(validation.illuminant_ids)
    assert not set(train.reflectance_ids) & set(validation.reflectance_ids)
    model = TrainableC2LUT(_parameters())
    result = fit(
        model,
        train,
        validation,
        config=TrainingConfig(
            steps=240,
            learning_rate=0.01,
            min_learning_rate=0.0001,
            batch_illuminants=9,
            samples_per_illuminant=48,
            smoothness_weight=0.0,
            validation_interval=20,
            seed=7,
        ),
    )
    assert len(result.history) == 240
    assert result.best_step > 0
    assert result.final_delta_e76 == pytest.approx(result.best_delta_e76)
    assert result.final_delta_e76 < 0.3 * result.initial_delta_e76
    white = torch.tensor(train_whites[[0, -1]], dtype=torch.float64)
    identical_points = torch.full((2, 1, 3), 0.5, dtype=torch.float64)
    corrected = model(identical_points * white[:, None, :], white)
    assert float(torch.linalg.vector_norm(corrected[0] - corrected[1]).detach()) > 0.05


def test_fit_is_deterministic_without_changing_global_random_state() -> None:
    white = np.array([[0.6, 1.0, 1.4], [1.4, 1.0, 0.6]])
    pairs = _analytic_pairs(3, white)
    config = TrainingConfig(steps=3, validation_interval=1, seed=11)
    before = torch.get_rng_state().clone()
    first, second = TrainableC2LUT(_parameters()), TrainableC2LUT(_parameters())
    result_first = fit(first, pairs, config=config)
    result_second = fit(second, pairs, config=config)
    torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)
    assert result_first == result_second
    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters(), strict=True
    ):
        torch.testing.assert_close(first_parameter, second_parameter, atol=0, rtol=0)


@pytest.mark.parametrize(
    "argument,value",
    [
        ("steps", 0),
        ("steps", True),
        ("batch_illuminants", -1),
        ("samples_per_illuminant", 0),
        ("validation_interval", 0),
        ("learning_rate", float("nan")),
        ("learning_rate", 0.0),
        ("min_learning_rate", 1.0),
        ("smoothness_weight", -0.1),
        ("seed", -1),
    ],
)
def test_invalid_training_configuration_is_rejected(argument: str, value: object) -> None:
    with pytest.raises(ValueError):
        TrainingConfig(**{argument: value})  # type: ignore[arg-type]


def test_nonfinite_or_invalid_white_points_are_rejected() -> None:
    model = TrainableC2LUT(_parameters())
    for white in ([[0.0, 1.0, 1.0]], [[float("nan"), 1.0, 1.0]], [[-1.0, 1.0, 1.0]]):
        with pytest.raises(ValueError):
            model.weights(torch.tensor(white, dtype=torch.float64))
