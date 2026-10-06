"""Internal helpers for preserving numeric output precision."""

import numpy as np
from typing import overload


@overload
def _normalize_float(value: float | np.floating) -> float | np.longdouble: ...


@overload
def _normalize_float(value: np.ndarray) -> np.ndarray: ...


def _normalize_float(value: float | np.floating | np.ndarray
                     ) -> float | np.longdouble | np.ndarray:
    """Normalize ordinary floating scalars; preserve long doubles and arrays."""
    if isinstance(value, np.longdouble):
        return value
    if isinstance(value, np.floating):
        return float(value)
    return value
