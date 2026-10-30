"""Independent checks of the paper's tensor model and public inference contract."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from c2lut import C2LUT, ModelConfig, ModelParameters, numpy_interpolate


def small_parameters(*, perturbation: float = 1e-3) -> ModelParameters:
    return ModelParameters.identity(
        ModelConfig(grid_size=5, rank=3, num_bases=4, hidden_features=6),
        seed=53,
        perturbation=perturbation,
    )


@pytest.mark.parametrize("rank", [2, 3, 5])
def test_identity_tucker_initialization_and_uniform_weights(rank: int) -> None:
    parameters = ModelParameters.identity(
        ModelConfig(grid_size=5, rank=rank, num_bases=4, hidden_features=6), seed=101
    )
    model = C2LUT(parameters)
    whites = np.array([[0.8, 1.2, 0.4], [2.0, 0.5, 1.0]])
    assert_array_equal(model.weights(whites), np.full((2, 4), 0.25))
    axis = np.linspace(0.0, 1.0, 5)
    expected = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"))
    for white in whites:
        assert_allclose(model.make_lut(white), expected, atol=3e-15, rtol=4e-15)


def test_zero_mean_perturbations_break_symmetry_without_changing_initial_lut() -> None:
    symmetric = small_parameters(perturbation=0.0)
    perturbed = small_parameters(perturbation=0.01)
    assert_array_equal(symmetric.cores[0], symmetric.cores[1])
    assert np.linalg.norm(perturbed.cores[0] - perturbed.cores[1]) > 0.05
    assert_allclose(perturbed.cores.mean(axis=0), symmetric.cores[0], atol=3e-15, rtol=0)
    assert_allclose(
        C2LUT(perturbed).make_lut([1.0, 0.8, 1.2]),
        C2LUT(symmetric).make_lut([1.0, 0.8, 1.2]),
        atol=3e-15,
        rtol=0,
    )
    for name, value in perturbed.arrays().items():
        assert_array_equal(value, small_parameters(perturbation=0.01).arrays()[name])


def test_paper_configuration_parameter_count() -> None:
    assert ModelParameters.identity().parameter_count == 180_684


def test_tucker_reconstruction_matches_full_tensor_contraction() -> None:
    rng = np.random.default_rng(311)
    original = small_parameters()
    parameters = ModelParameters(
        config=original.config,
        **{
            name: rng.normal(scale=0.4, size=value.shape)
            for name, value in original.arrays().items()
        },
    )
    model = C2LUT(parameters)
    weights = model.weights([0.3, 0.4, 0.9])
    expected = np.einsum(
        "j,jiabc,oi,ra,gb,kc->orgk",
        weights,
        parameters.cores,
        parameters.color_factor,
        parameters.red_factor,
        parameters.green_factor,
        parameters.blue_factor,
        optimize=False,
    )
    assert_allclose(model.make_lut([0.3, 0.4, 0.9]), expected, atol=3e-16, rtol=2e-13)


def test_chromaticity_changes_the_transform_for_identical_balanced_rgb() -> None:
    base = ModelParameters.identity(
        ModelConfig(grid_size=4, rank=2, num_bases=2, hidden_features=2), perturbation=0
    )
    cores = base.cores.copy()
    cores[1] *= 2.0
    parameters = replace(
        base,
        cores=cores,
        mlp_weight1=np.eye(2),
        mlp_weight2=np.array([[5.0, -5.0], [-5.0, 5.0]]),
    )
    model = C2LUT(parameters)
    balanced = np.array([[0.3, 0.4, 0.6]])
    warm, green = np.array([2.0, 1.0, 1.0]), np.array([1.0, 2.0, 1.0])
    outputs = []
    for white in (warm, green):
        logits = np.array([5.0, -5.0]) * ((white[0] - white[1]) / white.sum())
        weights = np.exp(logits) / np.exp(logits).sum()
        assert_allclose(model.weights(white), weights, atol=1e-16, rtol=1e-15)
        result = model.predict(balanced * white, white)
        assert_allclose(result, balanced * (weights[0] + 2.0 * weights[1]), atol=2e-15)
        outputs.append(result)
    assert np.linalg.norm(outputs[0] - outputs[1]) > 0.1


@pytest.mark.parametrize("scale", [0.001, 7.0, 1e250])
def test_joint_raw_white_scaling_is_invariant(scale: float) -> None:
    model = C2LUT(small_parameters())
    white = np.array([0.9, 1.2, 1.8])
    raw = np.random.default_rng(17).random((19, 3)) * white
    assert_allclose(model.weights(white * scale), model.weights(white), atol=2e-16)
    assert_allclose(
        model.predict(raw * scale, white * scale), model.predict(raw, white), atol=2e-15
    )
    assert np.isfinite(model.weights([1e308, 1e308, 1e308])).all()


@pytest.mark.parametrize("shape", [(3,), (0, 3), (2, 0, 3), (7, 3), (2, 3, 4, 3)])
def test_arbitrary_leading_dimensions_and_empty_images(shape: tuple[int, ...]) -> None:
    balanced = np.random.default_rng(91).random(shape)
    white = np.array([0.8, 1.3, 0.6])
    result = C2LUT(small_parameters()).predict(balanced * white, white)
    assert result.shape == shape
    assert_allclose(result, balanced, atol=2e-15, rtol=2e-15)


def test_public_api_accepts_noncontiguous_real_inputs() -> None:
    balanced = np.random.default_rng(72).random((3, 4, 3))
    white = np.array([0.7, 1.4, 1.0])
    raw = (balanced * white)[:, ::-1]
    assert not raw.flags.c_contiguous
    assert_allclose(C2LUT(small_parameters()).predict(raw, white), balanced[:, ::-1], atol=2e-15)


def test_prepared_transform_reuses_the_lut(monkeypatch: pytest.MonkeyPatch) -> None:
    model = C2LUT(small_parameters())
    white = np.array([0.7, 1.0, 0.9])
    transform = model.prepare(white)

    def reconstruction_must_not_run(white: object) -> None:
        raise AssertionError("prepared transform regenerated its LUT")

    monkeypatch.setattr(model, "make_lut", reconstruction_must_not_run)
    for seed in range(3):
        balanced = np.random.default_rng(seed).random((4, 5, 3))
        assert_allclose(transform(balanced * white), balanced, atol=2e-15)
    assert not transform.lut.flags.writeable
    assert not transform.white.flags.writeable


def test_explicit_range_policy() -> None:
    model = C2LUT(small_parameters())
    white = np.array([0.5, 0.8, 1.0])
    raw = np.array([[-0.1, 0.5, 1.2]]) * white
    assert_allclose(model.predict(raw, white), [[0.0, 0.5, 1.0]], atol=2e-15)
    with pytest.raises(ValueError, match="outside"):
        model.predict(raw, white, out_of_range="raise")
    with pytest.raises(ValueError, match="out_of_range"):
        model.predict(raw, white, out_of_range="invalid")  # type: ignore[arg-type]


def test_native_and_numpy_backends_agree_on_nontrivial_tensor_model() -> None:
    original = small_parameters()
    rng = np.random.default_rng(29)
    parameters = replace(
        original,
        cores=original.cores + rng.normal(scale=0.5, size=original.cores.shape),
        mlp_weight2=rng.normal(scale=2.0, size=original.mlp_weight2.shape),
    )
    raw = rng.random((5, 7, 3))
    white = np.array([0.8, 0.5, 1.3])
    assert_allclose(
        C2LUT(parameters).predict(raw, white),
        C2LUT(parameters, interpolator=numpy_interpolate).predict(raw, white),
        atol=5e-16,
        rtol=2e-14,
    )


def test_parameter_arrays_are_owned_and_readonly() -> None:
    original = small_parameters()
    core = original.cores.copy()
    copied = replace(original, cores=core)
    core.fill(42)
    assert_array_equal(copied.cores, original.cores)
    with pytest.raises(ValueError):
        copied.cores[0, 0, 0, 0, 0] = 42


@pytest.mark.parametrize(
    "changes",
    [
        {"grid_size": 1},
        {"rank": 1},
        {"grid_size": 3, "rank": 4},
        {"num_bases": 0},
        {"hidden_features": 0},
        {"grid_size": True},
        {"rank": 2.5},
    ],
)
def test_invalid_configuration_is_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ModelConfig(**changes)  # type: ignore[arg-type]


@pytest.mark.parametrize("white", [[0, 1, 1], [-1, 1, 1], [np.nan, 1, 1], [[1, 1, 1]]])
def test_invalid_global_white_point_is_rejected(white: object) -> None:
    with pytest.raises(ValueError):
        C2LUT(small_parameters()).predict([[0.2, 0.4, 0.6]], white)


@pytest.mark.parametrize("raw", [[[np.nan, 0.1, 0.2]], [[1, 2]], [[1j, 0, 0]]])
def test_invalid_raw_values_are_rejected(raw: object) -> None:
    with pytest.raises(ValueError):
        C2LUT(small_parameters()).predict(raw, [1.0, 1.0, 1.0])
