"""Evaluation helpers shared by the command-line workflow and experiments."""

from __future__ import annotations

import numpy as np

from ._validation import FloatArray
from .color import angular_error_degrees, delta_e76, delta_e2000, xyz_to_lab
from .data import PairedDataset
from .model import C2LUT


def predict_dataset(model: C2LUT, dataset: PairedDataset) -> FloatArray:
    """Reuse a generated LUT for all samples of each illuminant."""
    return np.stack(
        [
            model.predict(raw, white)
            for raw, white in zip(dataset.raw, dataset.white_points, strict=True)
        ]
    )


def color_metrics(prediction: FloatArray, target: FloatArray) -> dict[str, float]:
    """Report held-out errors in the relative XYZ/Lab coordinate convention."""
    if prediction.shape != target.shape or prediction.size == 0:
        raise ValueError("prediction and target must have equal, nonempty shapes")
    lab_prediction, lab_target = xyz_to_lab(prediction), xyz_to_lab(target)
    de76 = delta_e76(lab_prediction, lab_target)
    de00 = delta_e2000(lab_prediction, lab_target)
    angular = angular_error_degrees(prediction, target)
    return {
        "xyz_rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "delta_e76_mean": float(np.mean(de76)),
        "delta_e76_p95": float(np.quantile(de76, 0.95)),
        "delta_e2000_mean": float(np.mean(de00)),
        "delta_e2000_p95": float(np.quantile(de00, 0.95)),
        "angular_error_mean_degrees": float(np.mean(angular)),
    }


def fit_fixed_ccm(dataset: PairedDataset) -> FloatArray:
    """Least-squares 3x3 baseline, fitted exclusively on the training split."""
    balanced = dataset.raw / dataset.white_points[:, None, :]
    matrix, _, _, _ = np.linalg.lstsq(
        balanced.reshape(-1, 3), dataset.xyz.reshape(-1, 3), rcond=None
    )
    return np.asarray(matrix)


def evaluate_fixed_ccm(matrix: FloatArray, dataset: PairedDataset) -> dict[str, float]:
    prediction = (dataset.raw / dataset.white_points[:, None, :]) @ matrix
    return color_metrics(prediction, dataset.xyz)
