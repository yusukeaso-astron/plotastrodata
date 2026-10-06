import numpy as np
import shlex
import subprocess
from typing import Any

from plotastrodata import const_utils as cu
from plotastrodata._type_utils import _normalize_float


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
        scalars and array dtypes are preserved.
    """
    result = 2 * cu.h * nu**3 / cu.c**2 / (np.exp(cu.h * nu / cu.k_B / T) - 1)
    return _normalize_float(result)


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
        scalars and array dtypes are preserved.
    """
    result = cu.h * nu / cu.k_B / (np.exp(cu.h * nu / cu.k_B / T) - 1)
    return _normalize_float(result)
