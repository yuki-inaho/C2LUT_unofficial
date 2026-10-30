"""Spectral pair generation and reproducible, disjoint artificial datasets.

The built-in spectra are mathematical test fixtures, not measured cameras or
official CIE/D65 tables. Supply measured arrays to ``render_spectra`` for research
data. This module never downloads data during training or tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, IntArray, finite_array, positive_int


@dataclass(frozen=True)
class PairedDataset:
    """A Cartesian grid of illuminants and reflectances with persistent IDs."""

    raw: FloatArray
    white_points: FloatArray
    xyz: FloatArray
    illuminant_ids: IntArray
    reflectance_ids: IntArray

    def __post_init__(self) -> None:
        raw = finite_array(self.raw, "raw")
        white = finite_array(self.white_points, "white_points")
        xyz = finite_array(self.xyz, "xyz")
        if raw.ndim != 3 or raw.shape[-1] != 3 or min(raw.shape[:2]) < 1:
            raise ValueError("raw must have nonempty shape (illuminants, reflectances, 3)")
        if white.shape != (raw.shape[0], 3) or np.any(white <= 0):
            raise ValueError("white_points must be positive with shape (illuminants, 3)")
        if xyz.shape != raw.shape:
            raise ValueError("xyz must have the same shape as raw")
        if np.any(raw < 0):
            raise ValueError("raw must be nonnegative")
        for name, array in (("raw", raw), ("white_points", white), ("xyz", xyz)):
            owned = array.copy()
            owned.flags.writeable = False
            object.__setattr__(self, name, owned)
        for name, length in (
            ("illuminant_ids", raw.shape[0]),
            ("reflectance_ids", raw.shape[1]),
        ):
            value = np.asarray(getattr(self, name))
            if value.shape != (length,) or value.dtype.kind not in "iu":
                raise ValueError(f"{name} must be an integer array of length {length}")
            if value.dtype.kind == "u" and np.any(value > np.iinfo(np.int64).max):
                raise ValueError(f"{name} values must fit in int64")
            if len(np.unique(value)) != length:
                raise ValueError(f"{name} must contain unique identifiers")
            identifiers = np.array(value, dtype=np.int64, copy=True)
            identifiers.flags.writeable = False
            object.__setattr__(self, name, identifiers)

    def subset(self, illuminants: IntArray, reflectances: IntArray) -> PairedDataset:
        """Select by position, retaining IDs for train/validation/test audits."""
        return PairedDataset(
            self.raw[illuminants][:, reflectances],
            self.white_points[illuminants],
            self.xyz[illuminants][:, reflectances],
            self.illuminant_ids[illuminants],
            self.reflectance_ids[reflectances],
        )

    def save(self, path: str | Path) -> None:
        """Save numeric arrays only, with no Python pickle objects."""
        with Path(path).open("wb") as stream:
            np.savez_compressed(
                stream,
                raw=self.raw,
                white_points=self.white_points,
                xyz=self.xyz,
                illuminant_ids=self.illuminant_ids,
                reflectance_ids=self.reflectance_ids,
            )

    @classmethod
    def load(cls, path: str | Path) -> PairedDataset:
        with np.load(path, allow_pickle=False) as archive:
            names = {"raw", "white_points", "xyz", "illuminant_ids", "reflectance_ids"}
            if set(archive.files) != names:
                raise ValueError("unexpected or missing dataset fields")
            return cls(**{name: archive[name] for name in names})


@dataclass(frozen=True)
class DatasetSplits:
    train: PairedDataset
    validation: PairedDataset
    test: PairedDataset


def integration_weights(wavelengths: ArrayLike) -> FloatArray:
    """Trapezoidal quadrature weights, including irregular wavelength spacing."""
    wave = finite_array(wavelengths, "wavelengths")
    if wave.ndim != 1 or wave.size < 2 or np.any(np.diff(wave) <= 0):
        raise ValueError("wavelengths must be a strictly increasing 1D array")
    steps = np.diff(wave)
    weights = np.empty_like(wave)
    weights[0], weights[-1] = steps[0] / 2.0, steps[-1] / 2.0
    weights[1:-1] = (steps[:-1] + steps[1:]) / 2.0
    return weights


def render_spectra(
    wavelengths: ArrayLike,
    illuminants: ArrayLike,
    reflectances: ArrayLike,
    sensor_sensitivities: ArrayLike,
    color_matching_functions: ArrayLike,
    reference_illuminant: ArrayLike,
) -> PairedDataset:
    """Implement paper Eqs. 1-3 by trapezoidal spectral integration.

    Shapes are wavelength (W,), illuminants (L,W), reflectances (P,W),
    sensor/observer (W,3), reference (W,). Raw and white share one scale;
    reference XYZ is normalized so that its perfect reflector has Y=1.
    Inputs must use a common wavelength grid and compatible radiometric units.
    """
    weights = integration_weights(wavelengths)
    lamps = finite_array(illuminants, "illuminants")
    surfaces = finite_array(reflectances, "reflectances")
    sensors = finite_array(sensor_sensitivities, "sensor_sensitivities")
    observer = finite_array(color_matching_functions, "color_matching_functions")
    reference = finite_array(reference_illuminant, "reference_illuminant")
    w = weights.size
    for name, array in (("illuminants", lamps), ("reflectances", surfaces)):
        if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != w:
            raise ValueError(f"{name} must have a nonempty shape (count, wavelengths)")
    if sensors.shape != (w, 3) or observer.shape != (w, 3) or reference.shape != (w,):
        raise ValueError("sensor, observer and reference wavelength dimensions must match")
    if any(np.any(array < 0) for array in (lamps, surfaces, sensors, observer, reference)):
        raise ValueError("spectral inputs must be nonnegative")
    if np.any(surfaces > 1):
        raise ValueError("reflectances must be in [0, 1]")
    normalization = float(np.sum(reference * observer[:, 1] * weights))
    if normalization <= 0:
        raise ValueError("reference illuminant must produce positive luminance")
    raw = np.einsum("lw,pw,wc,w->lpc", lamps, surfaces, sensors, weights, optimize=True)
    white = np.einsum("lw,wc,w->lc", lamps, sensors, weights, optimize=True)
    target = np.einsum("w,pw,wc,w->pc", reference, surfaces, observer, weights, optimize=True)
    target /= normalization
    return PairedDataset(
        raw=np.asarray(raw),
        white_points=np.asarray(white),
        xyz=np.broadcast_to(target, raw.shape).copy(),
        illuminant_ids=np.arange(lamps.shape[0]),
        reflectance_ids=np.arange(surfaces.shape[0]),
    )


def split_dataset(dataset: PairedDataset, *, seed: int = 0) -> DatasetSplits:
    """Split both axes 60/20/20; no illuminant or reflectance crosses a split."""
    rng = np.random.default_rng(seed)
    divisions: list[list[IntArray]] = []
    for count in dataset.raw.shape[:2]:
        if count < 5:
            raise ValueError("at least five illuminants and reflectances are needed")
        order = rng.permutation(count).astype(np.int64)
        end_train, end_validation = int(0.6 * count), int(0.8 * count)
        divisions.append(list(np.split(order, [end_train, end_validation])))
    parts = [dataset.subset(divisions[0][i], divisions[1][i]) for i in range(3)]
    return DatasetSplits(*parts)


def make_synthetic_splits(
    *,
    seed: int = 42,
    n_illuminants: int = 48,
    n_reflectances: int = 240,
) -> DatasetSplits:
    """Generate an independent artificial spectral world, then separate both axes.

    Smooth and peaked lights illuminate bounded smooth reflectances. Camera and
    observer sensitivities differ. The artificial observer is scaled to the
    conventional D65 white tristimulus (0.95047,1,1.08883), but is NOT a measured
    CIE observer and the reference spectrum is NOT the standard D65 SPD.
    """
    positive_int(n_illuminants, "n_illuminants", 5)
    positive_int(n_reflectances, "n_reflectances", 5)
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(400.0, 700.0, 61)

    def gaussian(center: float, width: float) -> FloatArray:
        return np.asarray(np.exp(-0.5 * ((wavelengths - center) / width) ** 2))

    sensors = np.stack([gaussian(605, 48), gaussian(540, 39), gaussian(455, 32)], axis=1)
    observer = np.stack(
        [gaussian(600, 38) + 0.32 * gaussian(445, 22), gaussian(550, 43), gaussian(450, 29)],
        axis=1,
    )
    weights = integration_weights(wavelengths)
    reference = np.ones_like(wavelengths)
    observer *= np.array([0.95047, 1.0, 1.08883]) / (weights @ observer)

    lamp_basis = np.stack(
        [0.2 + gaussian(445, 24), 0.2 + gaussian(535, 33), 0.2 + gaussian(620, 30)]
    )
    mixing = rng.dirichlet(np.full(3, 1.8), size=n_illuminants)
    illuminants = mixing @ lamp_basis
    # A smooth nonlinear surface family is independent of the model architecture.
    surface_basis = np.stack([gaussian(445, 48), gaussian(535, 51), gaussian(625, 48)])
    coefficients = rng.uniform(-2.7, 2.7, size=(n_reflectances, 3))
    reflectances = 0.03 + 0.94 / (
        1.0 + np.exp(-(coefficients @ surface_basis + rng.normal(0, 0.45, (n_reflectances, 1))))
    )
    dataset = render_spectra(wavelengths, illuminants, reflectances, sensors, observer, reference)
    return split_dataset(dataset, seed=seed + 1)
