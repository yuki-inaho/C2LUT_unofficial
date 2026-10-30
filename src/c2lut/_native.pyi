"""Typed declarations for the C++ interpolation extension."""

import numpy as np
from numpy.typing import NDArray

def trilinear(lut: NDArray[np.float64], points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Interpolate C-contiguous LUT (3,N,N,N) at normalized RGB points (M,3)."""
