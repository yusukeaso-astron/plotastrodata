import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI

from plotastrodata.matrix_utils import Mrot, dot2d


def listing(*args) -> list:
    """Output a list of the input when the input is string or number.

    Returns:
        list: With a single non-list input, the output is a list like ['a'], rather than [['a']].
    """
    strnum = [str, float, int, np.float64, np.int64, np.float32, np.int32]
    b = [None] * len(args)
    for i, a in enumerate(args):
        b[i] = [a] if type(a) in strnum else a
    if len(args) == 1:
        b = b[0]
    return b


def isdeg(s: str) -> bool:
    """Whether the given string means degree.

    Args:
        s (str): The string to be checked.

    Returns:
        bool: Whether the given string means degree.
    """
    if type(s) is str:
        return s.strip() in ['deg', 'DEG', 'degree', 'DEGREE']
    else:
        return False


def trim(data: np.ndarray | None = None, x: np.ndarray | None = None,
         y: np.ndarray | None = None, v: np.ndarray | None = None,
         xlim: list[float] | None = None,
         ylim: list[float] | None = None,
         vlim: list[float] | None = None,
         pv: bool = False
         ) -> tuple[np.ndarray | None, list[np.ndarray | None]]:
    """Trim 2D or 3D data by given coordinates and their limits.

    Args:
        data (np.ndarray, optional): 2D or 3D array. Defaults to None.
        x (np.ndarray, optional): 1D array. Defaults to None.
        y (np.ndarray, optional): 1D array. Defaults to None.
        v (np.ndarray, optional): 1D array. Defaults to None.
        xlim (list, optional): [xmin, xmax]. Defaults to None.
        ylim (list, optional): [ymin, ymax]. Defaults to None.
        vlim (list, optional): [vmin, vmax]. Defaults to None.

    Returns:
        tuple: Trimmed (data, [x,y,v]).
    """
    def get_bounds(arr, lim):
        if arr is None or lim is None or None in lim:
            return arr, 0, None
        lo = np.argmin(np.abs(arr - max(np.min(arr), lim[0])))
        hi = np.argmin(np.abs(arr - min(np.max(arr), lim[1])))
        lo, hi = sorted((lo, hi))
        return arr[lo:hi + 1], lo, hi + 1

    xout, i0, i1 = get_bounds(x, xlim)
    yout, j0, j1 = get_bounds(y, ylim)
    vout, k0, k1 = get_bounds(v, vlim)

    if data is None:
        return None, [xout, yout, vout]
    
    d = np.squeeze(data)

    if d.ndim == 0:
        print("data has only one pixel.")
        return data, [xout, yout, vout]

    if d.ndim == 2:
        if pv:
            j0, j1 = k0, k1
        dataout = d[j0:j1, i0:i1]
    else:
        d = np.moveaxis(d, [-3, -2, -1], [0, 1, 2])
        d = d[k0:k1, j0:j1, i0:i1]
        d = np.moveaxis(d, [0, 1, 2], [-3, -2, -1])
        dataout = d

    return dataout, [xout, yout, vout]


def to4dim(data: np.ndarray) -> np.ndarray:
    """Change a 2D, 3D, or 4D array to a 4D array.

    Args:
        data (np.ndarray): Input data. 2D, 3D, or 4D.

    Returns:
        np.ndarray: Output 4D array.
    """
    if np.ndim(data) == 2:
        d = np.array([[data]])
    elif np.ndim(data) == 3:
        d = np.array([data])
    else:
        d = np.array(data)
    return d


def RGIxy(y: np.ndarray, x: np.ndarray, data: np.ndarray,
          yxnew: tuple[np.ndarray, np.ndarray] | None = None,
          **kwargs) -> object | np.ndarray:
    """RGI for x and y at each channel.

    Args:
        y (np.ndarray): 1D array. Second coordinate.
        x (np.ndarray): 1D array. First coordinate.
        data (np.ndarray): 2D, 3D, or 4D array.
        yxnew (tuple, optional): (ynew, xnew), where ynew and xnew are 1D or 2D arrays. Defaults to None.

    Returns:
        np.ndarray: The RGI function or the interpolated array.
    """
    if np.ndim(data) not in [2, 3, 4]:
        print('data must be 2D, 3D, or 4D.')
        return

    _kw = {'bounds_error': False, 'fill_value': np.nan,
           'method': 'linear'}
    _kw.update(kwargs)
    c4d = to4dim(data)
    c4d[np.isnan(c4d)] = 0
    f = [[RGI((y, x), c2d, **_kw)
          for c2d in c3d] for c3d in c4d]
    if yxnew is None:
        if len(f) == 1:
            f = f[0]
        if len(f) == 1:
            f = f[0]
        return f
    else:
        return np.squeeze([[f2d(tuple(yxnew)) for f2d in f3d] for f3d in f])


def RGIxyv(v: np.ndarray, y: np.ndarray, x: np.ndarray, data: np.ndarray,
           vyxnew: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
           **kwargs) -> object | np.ndarray:
    """RGI in the x-y-v space.

    Args:
        v (np.ndarray): 1D array. Third coordinate.
        y (np.ndarray): 1D array. Second coordinate.
        x (np.ndarray): 1D array. First coordinate.
        data (np.ndarray): 3D or 4D array.
        vyxnew (tuple, optional): (vnew, ynew, xnew), where vnew, ynew, and xnew are 1D or 2D arrays. Defaults to None.

    Returns:
        np.ndarray: The RGI function or the interpolated array.
    """
    if np.ndim(data) not in [3, 4]:
        print('data must be 3D or 4D.')
        return

    _kw = {'bounds_error': False, 'fill_value': np.nan,
           'method': 'linear'}
    _kw.update(kwargs)
    c4d = to4dim(data)
    c4d[np.isnan(c4d)] = 0
    f = [RGI((v, y, x), c3d, **_kw) for c3d in c4d]
    if vyxnew is None:
        if len(f) == 1:
            f = f[0]
        return f
    else:
        return np.squeeze([f3d(tuple(vyxnew)) for f3d in f])


def gaussian2d(xy: np.ndarray,
               amplitude: float, xo: float, yo: float,
               fwhm_major: float, fwhm_minor: float, pa: float
               ) -> np.ndarray:
    """Two dimensional Gaussian function.

    Args:
        xy (np.ndarray): A pair of (x, y).
        amplitude (float): Peak value.
        xo (float): Offset in the x direction.
        yo (float): Offset in the y direction.
        fwhm_major (float): Full width at half maximum in the major axis (but can be shorter than the minor axis).
        fwhm_minor (float): Full width at half maximum in the minor axis (but can be longer then the major axis).
        pa (float): Position angle of the major axis from the +y axis to the +x axis in the unit of degree.

    Returns:
        g (np.ndarray): 2D numpy array.
    """
    s, t = dot2d(Mrot(-pa), [xy[1] - yo, xy[0] - xo])
    g = amplitude * np.exp2(-4 * ((s / fwhm_major)**2 + (t / fwhm_minor)**2))
    return g


def close_figure(fig: object, savefig: dict | str | None = None,
                 show: bool = False, tight: bool = True) -> None:
    """Save, show, and close the figure.

    Args:
        fig (object): External plt.figure(). Defaults to None.
        savefig (dict or str, optional): For plt.figure().savefig(). Defaults to None.
        show (bool, optional): True means doing plt.show(). Defaults to False.
        tight (bool, optional): True means doing fig.tight_layout(). Defaults to False.
    """
    savefig0 = {'bbox_inches': 'tight', 'transparent': True}
    if tight:
        fig.tight_layout()
    if savefig is not None:
        s = {'fname': savefig} if type(savefig) is str else savefig
        savefig0.update(s)
        fig.savefig(**savefig0)
    if show:
        plt.show()
    plt.close()
