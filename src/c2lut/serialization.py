"""Versioned, pickle-free parameter archives; loading never executes code."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np

from .parameters import ModelConfig, ModelParameters

FORMAT_VERSION = 1


def save_parameters(parameters: ModelParameters, path: str | Path) -> None:
    """Atomically write a portable NumPy archive to the exact requested path."""
    destination = Path(path)
    metadata = json.dumps(
        {"format_version": FORMAT_VERSION, "config": asdict(parameters.config)}, sort_keys=True
    )
    contents: dict[str, Any] = {"metadata": np.asarray(metadata), **parameters.arrays()}
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_parameters(path: str | Path) -> ModelParameters:
    """Validate archive schema, configuration, shapes, and finite values."""
    names = {f.name for f in fields(ModelParameters)} - {"config"}
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            if set(archive.files) != names | {"metadata"}:
                raise ValueError("unexpected or missing model archive fields")
            metadata: Any = json.loads(str(archive["metadata"].item()))
            if not isinstance(metadata, dict) or set(metadata) != {"format_version", "config"}:
                raise ValueError("invalid model metadata")
            if type(metadata["format_version"]) is not int or metadata["format_version"] != 1:
                raise ValueError("unsupported model format_version")
            configuration = metadata["config"]
            if not isinstance(configuration, dict) or set(configuration) != {
                f.name for f in fields(ModelConfig)
            }:
                raise ValueError("invalid model configuration")
            return ModelParameters(
                config=ModelConfig(**configuration),
                **{name: archive[name] for name in names},
            )
    except (OSError, ValueError, TypeError, KeyError, EOFError, BadZipFile) as error:
        raise ValueError(f"cannot load model archive: {error}") from error
