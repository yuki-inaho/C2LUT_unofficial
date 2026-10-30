"""Execute the published synthetic notebook in a fresh, isolated Python kernel."""

from __future__ import annotations

import base64
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from c2lut import C2LUT
from c2lut.data import PairedDataset
from c2lut.evaluation import color_metrics, predict_dataset

nbformat = pytest.importorskip("nbformat", reason="requires the notebook extra")
nbclient = pytest.importorskip("nbclient", reason="requires the notebook extra")
jupyter_client = pytest.importorskip("jupyter_client", reason="requires the notebook extra")
pytest.importorskip("ipykernel", reason="requires the notebook extra")


@pytest.mark.e2e
@pytest.mark.notebook
def test_synthetic_notebook_runs_from_a_fresh_kernel(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    notebook_path = project_root / "notebooks" / "synthetic_color_correction.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    # Discard saved outputs so a stale checked-in execution cannot satisfy this test.
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    kernel_name = "c2lut-notebook-test"
    kernel_directory = tmp_path / "kernels" / kernel_name
    kernel_directory.mkdir(parents=True)
    (kernel_directory / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [
                    sys.executable,
                    "-m",
                    "ipykernel_launcher",
                    "-f",
                    "{connection_file}",
                ],
                "display_name": "C2LUT notebook test",
                "language": "python",
            }
        ),
        encoding="utf-8",
    )
    kernel_specs = jupyter_client.kernelspec.KernelSpecManager(
        kernel_dirs=[str(kernel_directory.parent)]
    )
    kernel_manager = jupyter_client.KernelManager(
        kernel_name=kernel_name, kernel_spec_manager=kernel_specs
    )
    client = nbclient.NotebookClient(
        notebook,
        km=kernel_manager,
        timeout=300,
        resources={"metadata": {"path": str(project_root)}},
        allow_errors=False,
    )
    executed = client.execute(
        cleanup_kc=True,
        env={
            **os.environ,
            "C2LUT_NOTEBOOK_OUTPUT": str(tmp_path),
            "MPLBACKEND": "Agg",
        },
    )
    nbformat.validate(executed)
    code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
    assert code_cells
    assert all(cell.execution_count is not None for cell in code_cells)
    outputs = [output for cell in code_cells for output in cell.outputs]
    assert not any(output.output_type == "error" for output in outputs)
    rendered_images = [
        output.data["image/png"] for output in outputs if "image/png" in output.get("data", {})
    ]
    assert len(rendered_images) >= 5
    for encoded_image in rendered_images:
        assert base64.b64decode(encoded_image).startswith(b"\x89PNG\r\n\x1a\n")

    report = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert report["seed"] == 42
    assert report["training_steps"] == 1600
    assert 0 < report["best_step"] <= report["training_steps"]
    assert report["split_shapes"] == {
        "train": [28, 144, 3],
        "validation": [10, 48, 3],
        "test": [10, 48, 3],
    }
    splits = {
        name: PairedDataset.load(tmp_path / f"{name}.npz")
        for name in ("train", "validation", "test")
    }
    for name, split in splits.items():
        assert list(split.raw.shape) == report["split_shapes"][name]
    for field in ("illuminant_ids", "reflectance_ids"):
        identifiers = [set(getattr(split, field)) for split in splits.values()]
        assert all(left.isdisjoint(right) for left, right in combinations(identifiers, 2))

    metrics = report["metrics"]
    assert metrics["trained"]["delta_e76_mean"] < 0.5 * metrics["initial"]["delta_e76_mean"]
    assert metrics["trained"]["delta_e76_mean"] < metrics["fixed_ccm"]["delta_e76_mean"]
    assert report["reload_max_abs_error"] == 0.0
    assert 0.0 <= report["native_torch_max_abs_error"] < 1e-10
    assert report["basis_weight_spread"] > 1e-4
    assert isinstance(report["native_backend"], str)
    assert "_native" in report["native_backend"]
    # Independently read the exported artifact, then recompute the displayed results.
    saved_model = C2LUT.load(tmp_path / "model.npz")
    actual_metrics = color_metrics(predict_dataset(saved_model, splits["test"]), splits["test"].xyz)
    for name, value in actual_metrics.items():
        np.testing.assert_allclose(value, metrics["trained"][name], rtol=1e-12, atol=1e-12)

    assert len(report["figure_files"]) >= 5
    assert len(set(report["figure_files"])) == len(report["figure_files"])
    for name in report["figure_files"]:
        figure = tmp_path / name
        assert figure.resolve().is_relative_to(tmp_path.resolve())
        assert figure.suffix == ".png"
        assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
