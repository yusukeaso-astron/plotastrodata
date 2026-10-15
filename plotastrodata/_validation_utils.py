"""Shared input validation without coercion or changes to numeric precision."""
from typing import Any

import numpy as np


def _is_finite_real_scalar(value: Any) -> bool:
    """Accept finite Python/NumPy real numeric scalars without coercion.

    Accept int, float, np.integer, and np.floating, excluding booleans.
    Arrays (including 0D), complex values, strings, and other objects are
    rejected. Return a Python bool and preserve extended floating precision.
    """
    if (isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer, np.floating))):
        return False
    # Integers are finite; np.isfinite cannot handle arbitrary-size Python ints.
    if isinstance(value, (int, np.integer)):
        return True
    return bool(np.isfinite(value))


def _is_positive_real_scalar(value: Any) -> bool:
    """Require finite scalar form and strict positivity without coercion.

    Return a Python bool; preserve the finite check's extended precision
    and rejection of booleans, complex values, and arrays (including 0D).
    """
    return _is_finite_real_scalar(value) and bool(value > 0)


def _require_positive_real_scalar(value: Any, name: str) -> None:
    """Raise ValueError for invalid positive scalar input, without coercion."""
    if not _is_positive_real_scalar(value):
        raise ValueError(f'{name} must be a finite positive real scalar.')


def _require_integer(value: Any, name: str, minimum: int = 1) -> None:
    """Require a Python/NumPy integer at or above minimum, excluding booleans.

    Floating values (even integral ones) and arrays (including 0D) are
    rejected. Accepted values are left unchanged; normalization belongs at
    public output boundaries.
    """
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum):
        req01 = {0: 'a nonnegative integer', 1: 'a positive integer'}
        requirement = req01.get(minimum, f'an integer of at least {minimum}')
        raise ValueError(f'{name} must be {requirement}.')
