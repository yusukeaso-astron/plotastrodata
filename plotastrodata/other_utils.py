import matplotlib.pyplot as plt
import numpy as np
import warnings
from scipy.interpolate import RegularGridInterpolator as RGI


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


def nearest_index(arr: np.ndarray, x: float = 0) -> int:
    """Get the index of the input arrary that gives a value nearest to the given value x. np.searchsorted() does not work with a descending array.

    Args:
        arr (np.ndarray): Sorted array.
        x (float, optional): Value to approach. Defaults to 0.

    Returns:
        int: The index that gives a value nearest to x.
    """
    return np.argmin(np.abs(arr - x))


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
        lo = nearest_index(arr, max(np.min(arr), np.min(lim)))
        hi = nearest_index(arr, min(np.max(arr), np.max(lim)))
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


def reform_grid(v: np.ndarray | None = None,
                k0: int | None = None, k1: int | None = None,
                vmin: float | None = None, vmax: float | None = None
                ) -> np.ndarray:
    """Extend or cut the given 1D array based on the given range.

    Args:
        v (np.ndarray | None, optional): Input 1D array. Defaults to None.
        k0 (int | None, optional): How many channels are added before v[0]; the minus sign means extension. k0 has the priority over vmin. Defaults to None.
        k1 (int | None, optional): How many channels are added after v[-1]; the plus sign means extension. k1 has the priority over vmax. Defaults to None.
        vmin (float | None, optional): New minimum velocity. Defaults to None.
        vmax (float | None, optional): New maximum velocity. Defaults to None.

    Returns:
        np.ndarray: Extended or cut 1D array.
    """
    if v is None or len(v) <= 1:
        return v

    dv = v[1] - v[0]
    if k0 is None and vmin is not None:
        k0 = int(round((vmin - v[0]) / dv))
    if k0 is not None and k0 != 0:
        if k0 < 0:
            vpre = v[0] + dv * np.arange(k0, 0)
            v = np.concatenate((vpre, v))
        else:
            v = v[k0:]
    if k1 is None and vmax is not None:
        k1 = int(round((vmax - v[-1]) / dv))
    if k1 is not None and k1 != 0:
        if k1 > 0:
            vpost = v[-1] + dv * np.arange(1, k1 + 1)
            v = np.concatenate((v, vpost))
        else:
            v = v[:len(v) + k1]
    return v


def reform_data(c: np.ndarray, v_in: np.ndarray | None,
                nv: int, v_org: np.ndarray | None = None,
                vskip: int = 1) -> np.ndarray:
    """Skip and fill channels with nan.

    Args:
        c (np.ndarray): The input 2D or 3D arrays.
        v_in (np.ndarray): The input velocity 1D array.
        nv (int): The number of channels with a label.
        v (np.ndarray, optional): The velocity 1D array, including the channels with and without a label. Defaults to None.
        vskip (int, optional): How many channels are skipped. Defaults to 1.

    Returns:
        np.ndarray: 3D arrays skipped and filled with nan.
    """
    if v_org is None:
        return c

    ndim = np.ndim(c)
    if ndim not in [2, 3]:
        print('c must be 2D or 3D.')
        return

    if ndim == 2:
        d = np.full((nv, *np.shape(c)), c)
    elif v_in is not None:
        dv_org = v_org[1] - v_org[0]
        dv_in = (v_in[1] - v_in[0]) * vskip
        k0 = nearest_index(v_org, v_in[0])
        k1 = nearest_index(v_org, v_in[-1])
        if np.abs(dv_in - dv_org) / dv_org < 0.01:
            d = c
        else:
            s = 'Velocity resolution mismatch (>1%).' \
                + ' The cube needs to be regridded' \
                + ' outside plotastrodata.'
            warnings.warn(s, UserWarning)
            n_valid = k1 - k0
            d = [None] * n_valid
            for k in range(n_valid):
                k_tmp = nearest_index(v_in, v_org[k])
                diffvel = np.abs(v_in[k_tmp] - v_org[k])
                nearby = diffvel < dv_org * 0.5
                d[k] = c[k_tmp] if nearby else c[0] * np.nan
            d = np.array(d)
        if k0 > 0:
            prenan = np.full((k0, *np.shape(d)[1:]), np.nan)
            d = np.concatenate((prenan, d))
        d = d[::vskip]
    shape = np.shape(d)
    shape = (len(v_org) - shape[0], shape[1], shape[2])
    postnan = np.full(shape, np.nan)
    d = np.concatenate((d, postnan))
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
