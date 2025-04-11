import subprocess
import shlex
import numpy as np

from plotastrodata import const_utils as cu


def terminal(cmd: str, **kwargs) -> None:
    """Run a terminal command through subprocess.run.

    Args:
        cmd (str): Terminal command.
    """
    subprocess.run(shlex.split(cmd), **kwargs)


def runpython(filename: str, **kwargs) -> None:
    """Run a python file.

    Args:
        filename (str): Python file name.
    """
    terminal(f'python {filename}', **kwargs)


def BnuT(T: float = 30, nu: float = 230e9) -> float:
    """Planck function.

    Args:
        T (float, optional): Temperature in the unit of K. Defaults to 30.
        nu (float, optional): Frequency in the unit of Hz. Defaults to 230e9.

    Returns:
        float: Planck function in the SI units.
    """
    return 2 * cu.h * nu**3 / cu.c**2 / (np.exp(cu.h * nu / cu.k_B / T) - 1)


def JnuT(T: float = 30, nu: float = 230e9) -> float:
    """Brightness templerature from the Planck function.

    Args:
        T (float, optional): Temperature in the unit of K. Defaults to 30.
        nu (float, optional): Frequency in the unit of Hz. Defaults to 230e9.

    Returns:
        float: Brightness temperature of Planck function in the unit of K.
    """
    return cu.h * nu / cu.k_B / (np.exp(cu.h * nu / cu.k_B / T) - 1)
