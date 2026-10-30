"""Independent physical integration and artificial-dataset integrity checks."""

from __future__ import annotations

from dataclasses import fields
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray

from c2lut.data import (
    PairedDataset,
    integration_weights,
    make_synthetic_splits,
    render_spectra,
    split_dataset,
)


def spectral_inputs() -> list[NDArray[np.float64]]:
    return [
        np.array([400.0, 401.0, 404.0, 410.0]),
        np.array([[1.0, 2.0, 3.0, 2.0], [0.5, 0.8, 1.3, 1.0]]),
        np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.1, 0.7, 0.2, 0.5]]),
        np.array([[0.2, 0.6, 1.0], [0.4, 0.7, 0.8], [0.7, 1.0, 0.3], [1.0, 0.5, 0.1]]),
        np.array([[0.3, 0.2, 1.2], [0.4, 0.5, 0.9], [0.6, 0.8, 0.4], [0.9, 0.4, 0.2]]),
        np.array([0.8, 1.1, 1.0, 0.7]),
    ]


def test_irregular_wavelength_weights_integrate_linear_function_exactly() -> None:
    wavelengths = np.array([400.0, 401.0, 404.0, 410.0])
    weights = integration_weights(wavelengths)
    assert_array_equal(weights, [0.5, 2.0, 4.5, 3.0])
    # Integral_0^10 (0.4 + 0.02 x) dx = 5, independent of the intermediate knots.
    values = 0.4 + 0.02 * (wavelengths - 400.0)
    assert_allclose(weights @ values, 5.0, atol=1e-15, rtol=0)


def test_affine_spectrum_has_known_analytic_raw_and_xyz() -> None:
    wavelengths = np.array([400.0, 401.0, 404.0, 410.0])
    coordinate = wavelengths - 400.0
    lamps = np.stack([1.0 + 0.2 * coordinate, 2.0 + 0.3 * coordinate])
    albedos = np.array([0.0, 0.3, 1.0])
    surfaces = np.broadcast_to(albedos[:, None], (3, 4))
    camera = np.broadcast_to([1.0, 2.0, 3.0], (4, 3))
    observer = np.broadcast_to([4.0, 5.0, 6.0], (4, 3))
    reference = 0.7 + 0.3 * coordinate
    dataset = render_spectra(wavelengths, lamps, surfaces, camera, observer, reference)
    # Exact integrals of the two affine lamps on this ten-unit interval.
    integrated_lamps = np.array([20.0, 35.0])
    expected_white = integrated_lamps[:, None] * np.array([1.0, 2.0, 3.0])
    expected_raw = expected_white[:, None, :] * albedos[None, :, None]
    expected_xyz = albedos[None, :, None] * np.array([0.8, 1.0, 1.2])
    assert_allclose(dataset.white_points, expected_white, atol=2e-14, rtol=0)
    assert_allclose(dataset.raw, expected_raw, atol=2e-14, rtol=0)
    assert_allclose(dataset.xyz, np.broadcast_to(expected_xyz, (2, 3, 3)), atol=5e-16, rtol=0)


def test_spectral_rendering_matches_independent_interval_integration() -> None:
    wave, lamps, surfaces, camera, observer, reference = spectral_inputs()

    def integral(values: NDArray[np.float64]) -> float:
        return sum(
            float((values[i] + values[i + 1]) * (wave[i + 1] - wave[i]) / 2.0)
            for i in range(len(wave) - 1)
        )

    dataset = render_spectra(wave, lamps, surfaces, camera, observer, reference)
    expected_raw = np.empty((2, 3, 3))
    expected_white = np.empty((2, 3))
    expected_xyz = np.empty((3, 3))
    normalization = integral(reference * observer[:, 1])
    for light in range(2):
        for channel in range(3):
            expected_white[light, channel] = integral(lamps[light] * camera[:, channel])
            for surface in range(3):
                expected_raw[light, surface, channel] = integral(
                    lamps[light] * surfaces[surface] * camera[:, channel]
                )
    for surface in range(3):
        for channel in range(3):
            expected_xyz[surface, channel] = (
                integral(reference * surfaces[surface] * observer[:, channel]) / normalization
            )
    assert_allclose(dataset.raw, expected_raw, atol=3e-15, rtol=3e-15)
    assert_allclose(dataset.white_points, expected_white, atol=3e-15, rtol=3e-15)
    assert_allclose(dataset.xyz, np.broadcast_to(expected_xyz, (2, 3, 3)), atol=3e-15)


def test_perfect_white_and_black_reflectors() -> None:
    dataset = render_spectra(*spectral_inputs())
    assert_allclose(dataset.raw[:, 0], dataset.white_points, atol=2e-15, rtol=1e-15)
    assert_allclose(dataset.xyz[:, 0, 1], 1.0, atol=1e-15, rtol=0)
    assert_array_equal(dataset.raw[:, 1], np.zeros((2, 3)))
    assert_array_equal(dataset.xyz[:, 1], np.zeros((2, 3)))
    assert np.all(dataset.raw / dataset.white_points[:, None, :] <= 1.0 + 1e-15)


def test_common_spd_scaling_preserves_xyz_and_raw_white_ratios() -> None:
    inputs = spectral_inputs()
    original = render_spectra(*inputs)
    factor = 3.7
    inputs[1] *= factor
    inputs[5] *= factor
    scaled = render_spectra(*inputs)
    assert_allclose(scaled.raw, original.raw * factor, atol=8e-15)
    assert_allclose(scaled.white_points, original.white_points * factor, atol=8e-15)
    assert_allclose(scaled.xyz, original.xyz, atol=5e-16)
    assert_allclose(
        scaled.raw / scaled.white_points[:, None, :],
        original.raw / original.white_points[:, None, :],
        atol=3e-16,
    )


@pytest.mark.parametrize(
    "wavelengths",
    [[400.0], [400.0, 400.0], [401.0, 400.0], [[400.0, 401.0]], [400.0, np.nan]],
)
def test_invalid_wavelength_grid_is_rejected(wavelengths: object) -> None:
    with pytest.raises(ValueError):
        integration_weights(wavelengths)


@pytest.mark.parametrize("input_index", range(1, 6))
@pytest.mark.parametrize("value", [-0.1, np.nan])
def test_invalid_spectral_values_are_rejected(input_index: int, value: float) -> None:
    inputs = spectral_inputs()
    inputs[input_index].flat[0] = value
    with pytest.raises(ValueError):
        render_spectra(*inputs)


@pytest.mark.parametrize("input_index", range(1, 6))
def test_mismatched_wavelength_dimensions_are_rejected(input_index: int) -> None:
    inputs = spectral_inputs()
    if input_index in (1, 2):
        inputs[input_index] = inputs[input_index][:, :-1]
    else:
        inputs[input_index] = inputs[input_index][:-1]
    with pytest.raises(ValueError):
        render_spectra(*inputs)


def test_reflectance_above_one_is_rejected() -> None:
    inputs = spectral_inputs()
    inputs[2][0, 0] = 1.001
    with pytest.raises(ValueError, match="reflectances"):
        render_spectra(*inputs)


def test_reference_and_camera_white_require_positive_response() -> None:
    inputs = spectral_inputs()
    inputs[5].fill(0.0)
    with pytest.raises(ValueError, match="positive luminance"):
        render_spectra(*inputs)
    inputs = spectral_inputs()
    inputs[1][0].fill(0.0)
    with pytest.raises(ValueError, match="white_points"):
        render_spectra(*inputs)


def test_synthetic_generation_is_reproducible_and_both_split_axes_are_disjoint() -> None:
    first = make_synthetic_splits(seed=127, n_illuminants=11, n_reflectances=17)
    second = make_synthetic_splits(seed=127, n_illuminants=11, n_reflectances=17)
    different = make_synthetic_splits(seed=128, n_illuminants=11, n_reflectances=17)
    names = ("train", "validation", "test")
    for name in names:
        left, right = getattr(first, name), getattr(second, name)
        for attribute in fields(PairedDataset):
            assert_array_equal(getattr(left, attribute.name), getattr(right, attribute.name))
    assert not np.array_equal(first.train.raw, different.train.raw)
    for attribute, count in (("illuminant_ids", 11), ("reflectance_ids", 17)):
        sets = [set(getattr(getattr(first, name), attribute).tolist()) for name in names]
        for left, right in combinations(sets, 2):
            assert left.isdisjoint(right)
        assert set.union(*sets) == set(range(count))
    assert first.train.raw.shape == (6, 10, 3)
    assert first.validation.raw.shape == (2, 3, 3)
    assert first.test.raw.shape == (3, 4, 3)


def test_split_requires_enough_samples_for_three_nonempty_partitions() -> None:
    dataset = render_spectra(*spectral_inputs())
    with pytest.raises(ValueError, match="at least five"):
        split_dataset(dataset)


def test_paired_dataset_copies_arrays_and_rejects_duplicate_ids() -> None:
    dataset = render_spectra(*spectral_inputs())
    raw = dataset.raw.copy()
    copy = PairedDataset(
        raw, dataset.white_points, dataset.xyz, dataset.illuminant_ids, dataset.reflectance_ids
    )
    raw.fill(42.0)
    assert_array_equal(copy.raw, dataset.raw)
    assert not copy.raw.flags.writeable
    assert not copy.illuminant_ids.flags.writeable
    with pytest.raises(ValueError, match="unique"):
        PairedDataset(
            dataset.raw,
            dataset.white_points,
            dataset.xyz,
            np.array([1, 1]),
            dataset.reflectance_ids,
        )


def test_dataset_save_load_preserves_numeric_values_and_identifiers(tmp_path: Path) -> None:
    dataset = make_synthetic_splits(seed=9, n_illuminants=9, n_reflectances=12).train
    path = tmp_path / "artificial.dataset"
    dataset.save(path)
    assert path.is_file()
    restored = PairedDataset.load(path)
    for attribute in fields(PairedDataset):
        assert_array_equal(getattr(restored, attribute.name), getattr(dataset, attribute.name))
        assert not getattr(restored, attribute.name).flags.writeable


def test_uint64_identifiers_preserve_valid_int64_limits_and_reject_overflow() -> None:
    dataset = render_spectra(*spectral_inputs())
    maximum = np.iinfo(np.int64).max
    valid_ids = np.array([maximum - 1, maximum], dtype=np.uint64)
    restored = PairedDataset(
        dataset.raw,
        dataset.white_points,
        dataset.xyz,
        valid_ids,
        dataset.reflectance_ids,
    )
    assert restored.illuminant_ids.dtype == np.int64
    assert_array_equal(restored.illuminant_ids, [maximum - 1, maximum])
    overflowing_ids = np.array([maximum, int(maximum) + 1], dtype=np.uint64)
    with pytest.raises(ValueError):
        PairedDataset(
            dataset.raw,
            dataset.white_points,
            dataset.xyz,
            overflowing_ids,
            dataset.reflectance_ids,
        )
