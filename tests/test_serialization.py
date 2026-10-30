"""Round-trip, malformed-input, and pickle-safety checks of portable archives."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from c2lut import C2LUT, ModelConfig, ModelParameters, numpy_interpolate
from c2lut.serialization import load_parameters, save_parameters


@pytest.fixture
def parameters() -> ModelParameters:
    return ModelParameters.identity(
        ModelConfig(grid_size=5, rank=3, num_bases=4, hidden_features=5), seed=227
    )


def archive_data(parameters: ModelParameters) -> dict[str, Any]:
    metadata = {"format_version": 1, "config": asdict(parameters.config)}
    return {"metadata": np.asarray(json.dumps(metadata)), **parameters.arrays()}


def test_roundtrip_uses_exact_path_and_preserves_all_parameters(
    tmp_path: Path, parameters: ModelParameters
) -> None:
    path = tmp_path / "model.weights"
    save_parameters(parameters, path)
    assert path.is_file()
    assert not path.with_suffix(".weights.npz").exists()
    restored = load_parameters(path)
    assert restored.config == parameters.config
    for name, array in parameters.arrays().items():
        assert_array_equal(restored.arrays()[name], array)
        assert not restored.arrays()[name].flags.writeable
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["model.weights"]


def test_model_save_load_preserves_predictions(tmp_path: Path, parameters: ModelParameters) -> None:
    model = C2LUT(parameters)
    path = tmp_path / "model.npz"
    model.save(path)
    restored = C2LUT.load(path, interpolator=numpy_interpolate)
    white = np.array([0.8, 0.6, 1.2])
    image = np.random.default_rng(17).random((4, 5, 3)) * white
    assert_allclose(restored.predict(image, white), model.predict(image, white), atol=5e-16)


@pytest.mark.parametrize("version", [0, 2, True, "1"])
def test_unsupported_or_mistyped_version_is_rejected(
    tmp_path: Path, parameters: ModelParameters, version: object
) -> None:
    data = archive_data(parameters)
    data["metadata"] = np.asarray(
        json.dumps({"format_version": version, "config": asdict(parameters.config)})
    )
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError, match="format_version"):
        load_parameters(path)


@pytest.mark.parametrize("mutation", ["missing", "extra", "shape", "nonfinite", "config"])
def test_archive_schema_and_array_content_are_validated(
    tmp_path: Path, parameters: ModelParameters, mutation: str
) -> None:
    data = archive_data(parameters)
    if mutation == "missing":
        del data["cores"]
    elif mutation == "extra":
        data["unexpected"] = np.zeros(1)
    elif mutation == "shape":
        data["cores"] = parameters.cores[:1]
    elif mutation == "nonfinite":
        data["mlp_bias1"] = np.full_like(parameters.mlp_bias1, np.nan)
    elif mutation == "config":
        config = asdict(parameters.config)
        config["grid_size"] = True
        data["metadata"] = np.asarray(json.dumps({"format_version": 1, "config": config}))
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError, match="cannot load model archive"):
        load_parameters(path)


@pytest.mark.parametrize("metadata", ["not-json", "null", "[]", '{"format_version": 1}'])
def test_invalid_metadata_is_rejected(
    tmp_path: Path, parameters: ModelParameters, metadata: str
) -> None:
    data = archive_data(parameters)
    data["metadata"] = np.asarray(metadata)
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError, match="cannot load model archive"):
        load_parameters(path)


def write_unpickle_marker(path: str) -> None:
    Path(path).write_text("unpickled", encoding="utf-8")


class UnpickleProbe:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[Any, tuple[str]]:
        return write_unpickle_marker, (str(self.marker),)


def test_loading_object_arrays_never_executes_pickle_payloads(
    tmp_path: Path, parameters: ModelParameters
) -> None:
    marker = tmp_path / "pickle-was-executed"
    data = archive_data(parameters)
    data["cores"] = np.array([UnpickleProbe(marker)], dtype=object)
    path = tmp_path / "object-model.npz"
    np.savez_compressed(path, **data)
    assert not marker.exists()
    with pytest.raises(ValueError, match="allow_pickle=False"):
        load_parameters(path)
    assert not marker.exists()


def test_truncated_archive_has_a_clear_value_error(
    tmp_path: Path, parameters: ModelParameters
) -> None:
    path = tmp_path / "truncated.npz"
    save_parameters(parameters, path)
    content = path.read_bytes()
    path.write_bytes(content[: len(content) // 2])
    with pytest.raises(ValueError, match="cannot load model archive"):
        load_parameters(path)


def test_failed_save_preserves_previous_model_and_removes_temporary_files(
    tmp_path: Path, parameters: ModelParameters, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "model.npz"
    save_parameters(parameters, path)
    original = path.read_bytes()

    def fail_during_write(*args: object, **kwargs: object) -> None:
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(np, "savez_compressed", fail_during_write)
    with pytest.raises(OSError, match="simulated interrupted write"):
        save_parameters(parameters, path)
    assert path.read_bytes() == original
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["model.npz"]
