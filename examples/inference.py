"""Run with ``uv run python examples/inference.py artifacts/model.npz``."""

from __future__ import annotations

import argparse

import numpy as np

from c2lut import C2LUT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="portable .npz archive produced by c2lut train/demo")
    arguments = parser.parse_args()
    model = C2LUT.load(arguments.model)
    # Raw and white must be in the same linear sensor scale; not gamma-encoded RGB.
    white = np.array([1.1, 1.0, 0.8])
    balanced = np.array([[[0.2, 0.4, 0.6], [0.9, 0.7, 0.3]]])
    prepared = model.prepare(white)
    print(prepared(balanced * white, out_of_range="raise"))


if __name__ == "__main__":
    main()
