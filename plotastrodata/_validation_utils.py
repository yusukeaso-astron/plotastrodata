"""Shared input validation without coercion or changes to numeric precision."""
from typing import Any

import numpy as np


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
