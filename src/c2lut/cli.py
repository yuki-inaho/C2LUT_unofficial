"""Reproducible synthetic-data, training, evaluation, and native-inference commands."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .data import PairedDataset, make_synthetic_splits
from .evaluation import color_metrics, evaluate_fixed_ccm, fit_fixed_ccm, predict_dataset
from .model import C2LUT
from .parameters import ModelConfig, ModelParameters


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=9)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--num-bases", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--smoothness-weight", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=1)


def _train(
    train: PairedDataset, validation: PairedDataset, arguments: argparse.Namespace
) -> tuple[C2LUT, dict[str, Any]]:
    # Import optional dependencies only in commands that actually learn weights.
    import torch

    from .training import TrainableC2LUT, TrainingConfig, fit

    if arguments.threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(arguments.threads)
    config = ModelConfig(arguments.grid_size, arguments.rank, arguments.num_bases)
    initial = ModelParameters.identity(config, seed=arguments.seed, perturbation=0.03)
    trainable = TrainableC2LUT(initial)
    training_config = TrainingConfig(
        steps=arguments.steps,
        learning_rate=arguments.learning_rate,
        min_learning_rate=1e-6,
        batch_illuminants=8,
        samples_per_illuminant=64,
        smoothness_weight=arguments.smoothness_weight,
        seed=arguments.seed,
    )
    result = fit(trainable, train, validation=validation, config=training_config)
    metadata = {
        "model_config": asdict(config),
        "training_config": asdict(training_config),
        "training": asdict(result),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "threads": arguments.threads,
    }
    return C2LUT(trainable.export_parameters()), metadata


def _demo(arguments: argparse.Namespace) -> None:
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    splits = make_synthetic_splits(seed=arguments.seed)
    for name in ("train", "validation", "test"):
        getattr(splits, name).save(output / f"{name}.npz")
    model, report = _train(splits.train, splits.validation, arguments)
    model.save(output / "model.npz")
    reloaded = C2LUT.load(output / "model.npz")
    prediction = predict_dataset(reloaded, splits.test)
    original_prediction = predict_dataset(model, splits.test)
    initial = C2LUT(ModelParameters.identity(model.parameters.config, seed=arguments.seed))
    report.update(
        {
            "paper": "https://arxiv.org/abs/2607.11681v2",
            "dataset_kind": "artificial spectra and observer; not a published camera benchmark",
            "selection": "best validation Delta E76; test split is never used for fitting",
            "initial_test": color_metrics(predict_dataset(initial, splits.test), splits.test.xyz),
            "fixed_ccm_test": evaluate_fixed_ccm(fit_fixed_ccm(splits.train), splits.test),
            "trained_test": color_metrics(prediction, splits.test.xyz),
            "reload_max_abs_error": float(np.max(np.abs(prediction - original_prediction))),
            "native_backend": True,
            "parameter_count": model.parameters.parameter_count,
            "split_shapes": {
                name: list(getattr(splits, name).raw.shape)
                for name in ("train", "validation", "test")
            },
            "split_ids": {
                name: {
                    "illuminants": getattr(splits, name).illuminant_ids.tolist(),
                    "reflectances": getattr(splits, name).reflectance_ids.tolist(),
                }
                for name in ("train", "validation", "test")
            },
        }
    )
    np.save(output / "prediction.npy", prediction, allow_pickle=False)
    _write_json(output / "report.json", report)
    print(json.dumps({"model": str(output / "model.npz"), "test": report["trained_test"]}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    synthetic = commands.add_parser("synthetic", help="generate disjoint artificial spectral pairs")
    synthetic.add_argument("--output", type=Path, required=True)
    synthetic.add_argument("--seed", type=int, default=42)
    synthetic.add_argument("--illuminants", type=int, default=48)
    synthetic.add_argument("--reflectances", type=int, default=240)

    demo = commands.add_parser("demo", help="generate, train, save, reload, and evaluate")
    demo.add_argument("--output", type=Path, required=True)
    _add_training_arguments(demo)

    train = commands.add_parser("train", help="train on saved paired datasets")
    train.add_argument("--train", type=Path, required=True)
    train.add_argument("--validation", type=Path, required=True)
    train.add_argument("--model", type=Path, required=True)
    train.add_argument("--report", type=Path, required=True)
    _add_training_arguments(train)

    evaluate = commands.add_parser("evaluate", help="evaluate using the native inference backend")
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)

    predict = commands.add_parser("predict", help="transform a linear RGB .npy array")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--white", type=float, nargs=3, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--out-of-range", choices=("clip", "raise"), default="clip")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one workflow and return a shell status; all persisted arrays are numeric."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "synthetic":
            arguments.output.mkdir(parents=True, exist_ok=True)
            splits = make_synthetic_splits(
                seed=arguments.seed,
                n_illuminants=arguments.illuminants,
                n_reflectances=arguments.reflectances,
            )
            for name in ("train", "validation", "test"):
                getattr(splits, name).save(arguments.output / f"{name}.npz")
        elif arguments.command == "demo":
            _demo(arguments)
        elif arguments.command == "train":
            model, report = _train(
                PairedDataset.load(arguments.train),
                PairedDataset.load(arguments.validation),
                arguments,
            )
            model.save(arguments.model)
            _write_json(arguments.report, report)
        elif arguments.command == "evaluate":
            dataset = PairedDataset.load(arguments.data)
            prediction = predict_dataset(C2LUT.load(arguments.model), dataset)
            _write_json(arguments.output, color_metrics(prediction, dataset.xyz))
        elif arguments.command == "predict":
            raw = np.load(arguments.input, allow_pickle=False)
            prediction = C2LUT.load(arguments.model).predict(
                raw, arguments.white, out_of_range=arguments.out_of_range
            )
            with arguments.output.open("wb") as stream:
                np.save(stream, prediction, allow_pickle=False)
    except ImportError as error:
        print(f"c2lut: {error}; training requires 'uv sync --extra train'", file=sys.stderr)
        return 2
    except (OSError, ValueError) as error:
        print(f"c2lut: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
