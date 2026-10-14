"""Internal helpers for preserving numeric output precision."""

import numpy as np
from typing import Annotated, Any, Callable, Literal, overload
from pydantic import WrapValidator


# Shared input types; validated settings are separate from uncoerced inputs.
RealScalar = float | np.floating | np.integer
RealSequence = list[RealScalar] | tuple[RealScalar, ...] | np.ndarray
PositionPair = tuple[RealScalar, RealScalar] | list[RealScalar] | np.ndarray
Position = str | PositionPair
Positions = Position | list[Position] | tuple[Position, ...]


def _validate_float(value: Any, handler: Callable) -> float | np.longdouble:
    """Preserve extended precision before Pydantic coerces other values."""
    return value if isinstance(value, np.longdouble) else handler(value)


ValidatedFloat = Annotated[float | np.longdouble,
                           WrapValidator(_validate_float)]

# Raw analysis inputs retain their containers and scalar types.
BeamInput = np.ndarray | list[RealScalar | None] | tuple[RealScalar | None,
                                                         RealScalar | None,
                                                         RealScalar | None]
# Plotting converts inputs to lists and validates shape/completeness separately.
ValidatedBeamTriple = list[ValidatedFloat | None]
ValidatedBeams = ValidatedBeamTriple | list[ValidatedBeamTriple]



_FloatNormalizationInput = int | float | np.integer | np.floating | np.ndarray
_FloatNormalizationResult = int | float | np.integer | np.longdouble | np.ndarray


@overload
def _normalize_float(value: _FloatNormalizationInput, *,
                     force_float: bool = False,
                     array_output: Literal[True]
                     ) -> np.ndarray: ...


@overload
def _normalize_float(value: np.ndarray, *,
                     force_float: bool = False,
                     array_output: None = None
                     ) -> np.ndarray: ...


@overload
def _normalize_float(value: float | np.floating, *,
                     force_float: bool = False,
                     array_output: Literal[False] | None = None
                     ) -> float | np.longdouble: ...


@overload
def _normalize_float(value: _FloatNormalizationInput, *,
                     force_float: bool = False,
                     array_output: bool | None = None
                     ) -> _FloatNormalizationResult: ...


def _normalize_float(value: _FloatNormalizationInput, *,
                     force_float: bool = False,
                     array_output: bool | None = None
                     ) -> _FloatNormalizationResult:
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
