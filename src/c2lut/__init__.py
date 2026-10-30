"""C²LUT inference, with NumPy and nanobind as its only runtime requirements."""

from .interpolation import Interpolator, native_interpolate, numpy_interpolate
from .model import C2LUT, PreparedTransform
from .parameters import ModelConfig, ModelParameters

__version__ = "0.1.0"
__all__ = [
    "C2LUT",
    "Interpolator",
    "ModelConfig",
    "ModelParameters",
    "PreparedTransform",
    "native_interpolate",
    "numpy_interpolate",
]
