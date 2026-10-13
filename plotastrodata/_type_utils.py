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


def _normalize_float_result(value: float | np.floating | np.ndarray,
                            *inputs: object
                            ) -> float | np.longdouble | np.ndarray:
    """Normalize a real-valued numerical result at its public boundary.

    Array inputs retain array results, including zero-dimensional results
    collapsed by NumPy arithmetic. Existing array dtypes are not changed.
    For scalar inputs, unwrap zero-dimensional temporary arrays and preserve
    extended-precision floating scalars.
    """
    if any(isinstance(a, np.ndarray) for a in inputs):
        return np.asarray(value)
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            return value
        value = value[()]
    return value if isinstance(value, np.longdouble) else float(value)
