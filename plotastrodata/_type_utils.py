"""Internal helpers for preserving numeric output precision."""

import numpy as np
from typing import Literal, overload


_RealValue = int | float | np.integer | np.floating | np.ndarray
_ScalarOrArray = int | float | np.integer | np.longdouble | np.ndarray


@overload
def _normalize_float(value: _RealValue, *, force_float: bool = False,
                     array_output: Literal[True]) -> np.ndarray: ...


@overload
def _normalize_float(value: np.ndarray, *, force_float: bool = False,
                     array_output: None = None) -> np.ndarray: ...


@overload
def _normalize_float(value: float | np.floating, *, force_float: bool = False,
                     array_output: Literal[False] | None = None
                     ) -> float | np.longdouble: ...


@overload
def _normalize_float(value: _RealValue, *, force_float: bool = False,
                     array_output: bool | None = None) -> _ScalarOrArray: ...


def _normalize_float(value: _RealValue, *, force_float: bool = False,
                     array_output: bool | None = None) -> _ScalarOrArray:
    """Normalize a real numerical result at its public output boundary.

    By default, convert ordinary NumPy floating scalars to Python floats,
    leaving integers and arrays unchanged. Extended-precision floating
    scalars retain their precision.

    force_float converts scalar integers to Python floats as well.
    array_output=True preserves or restores array form via np.asarray;
    False unwraps a zero-dimensional temporary before scalar normalization;
    None preserves the result's existing scalar/array form. Arrays with
    dimensions greater than zero always remain arrays. Array dtypes are
    never changed by force_float, including for zero-dimensional arrays
    retained with array_output=True or None.
    """
    if array_output is True:
        return np.asarray(value)
    if isinstance(value, np.ndarray):
        if array_output is not False or value.ndim != 0:
            return value
        value = value[()]
    if isinstance(value, np.longdouble):
        return value
    if force_float or isinstance(value, np.floating):
        return float(value)
    return value
