"""The installed CLI drives synthetic generation, learning and native inference."""

from __future__ import annotations

import importlib.machinery
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from c2lut import C2LUT, _native
from c2lut.data import PairedDataset
from c2lut.evaluation import color_metrics, predict_dataset


def run_cli(*arguments: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "c2lut", *arguments],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
        timeout=180,
    )


@pytest.fixture(scope="module")
def learned_workflow(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    pytest.importorskip("torch")
    directory = tmp_path_factory.mktemp("c2lut_e2e")
    run_cli("demo", "--output", str(directory), "--steps", "300", "--seed", "42", cwd=directory)
    return directory, json.loads((directory / "report.json").read_text())


@pytest.mark.e2e
def test_learning_improves_disjoint_unseen_spectral_pairs(
    learned_workflow: tuple[Path, dict[str, Any]],
) -> None:
    directory, report = learned_workflow
    assert report["trained_test"]["delta_e76_mean"] < 0.4 * report["initial_test"]["delta_e76_mean"]
    assert report["trained_test"]["xyz_rmse"] < report["initial_test"]["xyz_rmse"]
    assert report["training"]["best_step"] > 0
    assert report["training"]["final_delta_e76"] == report["training"]["best_delta_e76"]
    assert report["reload_max_abs_error"] == 0.0
    assert report["native_backend"] is True
    assert any(
        str(_native.__file__).endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES
    )
    splits = [
        PairedDataset.load(directory / f"{name}.npz") for name in ("train", "validation", "test")
    ]
    for name in ("illuminant_ids", "reflectance_ids"):
        sets = [set(getattr(split, name)) for split in splits]
        assert sets[0].isdisjoint(sets[1])
        assert sets[0].isdisjoint(sets[2])
        assert sets[1].isdisjoint(sets[2])


@pytest.mark.e2e
def test_native_and_differentiable_paths_agree_after_learning(
    learned_workflow: tuple[Path, dict[str, Any]],
) -> None:
    import torch

    from c2lut.training import TrainableC2LUT

    directory, _ = learned_workflow
    dataset = PairedDataset.load(directory / "test.npz")
    model = C2LUT.load(directory / "model.npz")
    trainable = TrainableC2LUT(model.parameters)
    with torch.no_grad():
        differentiable = trainable(
            torch.tensor(dataset.raw.copy()), torch.tensor(dataset.white_points.copy())
        ).numpy()
    assert_allclose(predict_dataset(model, dataset), differentiable, rtol=1e-11, atol=1e-12)
    # Adaptation must survive export: identical balanced RGB, different white points.
    white1, white2 = dataset.white_points[[0, -1]]
    points = np.array([[0.2, 0.6, 0.3], [0.7, 0.3, 0.8]])
    change = np.max(
        np.abs(model.predict(points * white1, white1) - model.predict(points * white2, white2))
    )
    assert change > 1e-6


@pytest.mark.e2e
def test_saved_model_predicts_image_in_new_process(
    learned_workflow: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    directory, _ = learned_workflow
    dataset = PairedDataset.load(directory / "test.npz")
    image = dataset.raw[0].reshape(6, 8, 3)
    np.save(tmp_path / "raw.npy", image)
    run_cli(
        "predict",
        "--model",
        str(directory / "model.npz"),
        "--input",
        str(tmp_path / "raw.npy"),
        "--white",
        *(str(x) for x in dataset.white_points[0]),
        "--output",
        str(tmp_path / "xyz.npy"),
        "--out-of-range",
        "raise",
        cwd=tmp_path,
    )
    expected = np.load(directory / "prediction.npy")[0].reshape(6, 8, 3)
    assert_allclose(np.load(tmp_path / "xyz.npy"), expected, rtol=0, atol=0)
    run_cli(
        "evaluate",
        "--model",
        str(directory / "model.npz"),
        "--data",
        str(directory / "test.npz"),
        "--output",
        str(tmp_path / "metrics.json"),
        cwd=tmp_path,
    )
    actual = json.loads((tmp_path / "metrics.json").read_text())
    target = color_metrics(np.load(directory / "prediction.npy"), dataset.xyz)
    assert actual == target


def test_inference_and_cli_imports_do_not_import_torch(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import c2lut.cli; from c2lut import _native; "
            "assert 'torch' not in sys.modules; print(_native.__file__)",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "_native" in result.stdout


def test_cli_reports_missing_input_cleanly(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "c2lut",
            "evaluate",
            "--model",
            "missing.npz",
            "--data",
            "missing-data.npz",
            "--output",
            "metrics.json",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "c2lut:" in result.stderr
    assert "Traceback" not in result.stderr
