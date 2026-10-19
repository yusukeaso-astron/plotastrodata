import shlex
import subprocess
from typing import Any

import numpy as np

from plotastrodata import const_utils as cu
from plotastrodata._type_utils import _normalize_float
from plotastrodata._validation_utils import _require_integer


def terminal(cmd: str, check: bool = True,
             **kwargs: Any) -> subprocess.CompletedProcess:
    """Run a terminal command through subprocess.run.

    The command is split with ``shlex.split`` and executed without a
    shell. It can still run destructive programs, so pass only trusted
    command strings.

    Args:
        cmd (str): Terminal command.
        check (bool, optional): Whether to raise an exception if the
            command fails. Defaults to True.

    Returns:
        subprocess.CompletedProcess: Result from subprocess.run.
    """
    return subprocess.run(shlex.split(cmd), check=check, **kwargs)


def runpython(filename: str, check: bool = True,
              **kwargs: Any) -> subprocess.CompletedProcess:
    """Run a python file.

    The file is executed as a Python script and can perform arbitrary
    file or system operations. Pass only trusted files.

    Args:
        filename (str): Python file name.
        check (bool, optional): Whether to raise an exception if the
            script fails. Defaults to True.

    Returns:
        subprocess.CompletedProcess: Result from subprocess.run.
    """
    return subprocess.run(['python', filename], check=check, **kwargs)


def BnuT(T: float | np.floating | np.ndarray = 30,
         nu: float | np.floating | np.ndarray = 230e9
         ) -> float | np.longdouble | np.ndarray:
    """Planck function.

    Args:
        T (float or np.floating or np.ndarray, optional): Temperature in the
            unit of K. Defaults to 30.
        nu (float or np.floating or np.ndarray, optional): Frequency in the
            unit of Hz. Defaults to 230e9.

    Returns:
        float or np.longdouble or np.ndarray: Planck function in SI units.
        Ordinary scalar results are Python floats; extended-precision
        scalars and computed array dtypes are preserved. Array inputs, including
        zero-dimensional arrays, produce arrays.
    """
    result = 2 * cu.h * nu**3 / cu.c**2 / (np.exp(cu.h * nu / cu.k_B / T) - 1)
    array_output = isinstance(T, np.ndarray) or isinstance(nu, np.ndarray)
    return _normalize_float(result, force_float=True, array_output=array_output)


def JnuT(T: float | np.floating | np.ndarray = 30,
         nu: float | np.floating | np.ndarray = 230e9
         ) -> float | np.longdouble | np.ndarray:
    """Brightness templerature from the Planck function.

    Args:
        T (float or np.floating or np.ndarray, optional): Temperature in the
            unit of K. Defaults to 30.
        nu (float or np.floating or np.ndarray, optional): Frequency in the
            unit of Hz. Defaults to 230e9.

    Returns:
        float or np.longdouble or np.ndarray: Brightness temperature in K.
        Ordinary scalar results are Python floats; extended-precision
        scalars and computed array dtypes are preserved. Array inputs, including
        zero-dimensional arrays, produce arrays.
    """
    result = cu.h * nu / cu.k_B / (np.exp(cu.h * nu / cu.k_B / T) - 1)
    array_output = isinstance(T, np.ndarray) or isinstance(nu, np.ndarray)
    return _normalize_float(result, force_float=True, array_output=array_output)


def costaper(shape: tuple[int | np.integer, ...],
             npix: int | np.integer = 0
             ) -> np.ndarray:
    """Return a separable cosine window with a 1D or 2D shape.

    Multiply the returned window by an array to apply the taper, for
    example, ``data * costaper(data.shape, npix=2)``.

    Args:
        shape (tuple of int or np.integer): One or two nonnegative axis
            lengths, for example, ``(nx,)`` or ``(ny, nx)``. Boolean axis
            lengths are not accepted.
        npix (int or np.integer, optional): Number of tapered pixels at each
            edge along every axis. Must be nonnegative and no greater than half
            the length of any axis. Zero returns an all-ones window.
            Boolean values are not accepted. Defaults to 0 (no taper).

    Returns:
        np.ndarray: Float64 weights with the requested ``shape``.
        Along each axis, the weight at distance ``d < npix`` from the
        nearest edge is ``(1 - cos(pi * d / npix)) / 2``; all other
        weights are one. Thus the outermost pixels are zero for a positive
        width. In 2D, the window is the outer product of the axis windows.

    Raises:
        ValueError: If ``shape`` is not a tuple of one or two nonnegative
            integer axis lengths (excluding booleans), or ``npix`` is not
            a nonnegative Python or NumPy integer (excluding
            booleans), or the two edge tapers would overlap.
    """
    if not isinstance(shape, tuple) or len(shape) not in (1, 2):
        raise ValueError('shape must be a tuple of one or two axis lengths.')
    for axis, length in enumerate(shape):
        _require_integer(length, f'shape[{axis}]', minimum=0)
    shape = tuple(int(length) for length in shape)
    _require_integer(npix, 'npix', minimum=0)
    npix = int(npix)
    if npix > min(shape) // 2:
        raise ValueError('npix must be at most half each axis length.')

    def taper1d(nx: int) -> np.ndarray:
        weights = np.ones(nx)
        if npix > 0:
            ramp = (1 - np.cos(np.pi * np.arange(npix) / npix)) / 2
            weights[:npix] = ramp
            weights[-npix:] = ramp[::-1]
        return weights

    if len(shape) == 1:
        return taper1d(shape[0])
    return taper1d(shape[0])[:, None] * taper1d(shape[1])[None, :]
