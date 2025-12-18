import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf
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


def estimate_rms(data: np.ndarray, sigma: float | str | None = 'hist'
                 ) -> float:
    """Estimate a noise level of a N-D array.
       When a float number or None is given, this function just outputs it.
       Following methos are acceptable.
       'edge': use data[0] and data[-1].
       'neg': use only negative values.
       'med': use the median of data^2 assuming Gaussian.
       'iter': exclude outliers.
       'out': exclude inner 60% about axes=-2 and -1.
       'hist': fit histgram with Gaussian.
       'hist-pbcor': fit histgram with PB-corrected Gaussian.

    Args:
        data (np.ndarray): N-D array.
        sigma (float or str): One of the methods above. Defaults to 'hist'.

    Returns:
        float: the estimated root mean square of noise.
    """
    if sigma is None:
        return None

    nums = [float, int, np.float64, np.int64, np.float32, np.int32]
    if type(sigma) in nums:
        noise = sigma
    elif np.ndim(np.squeeze(data)) == 0:
        print('sigma cannot be estimated from only one pixel.')
        noise = 0.0
    elif sigma == 'edge':
        ave = np.nanmean(data[::len(data) - 1])
        noise = np.nanstd(data[::len(data) - 1])
        if np.abs(ave) > 0.2 * noise:
            print('Warning: The intensity offset is larger than 0.2 sigma.')
    elif sigma == 'neg':
        noise = np.sqrt(np.nanmean(data[data < 0]**2))
    elif sigma == 'med':
        noise = np.sqrt(np.nanmedian(data**2) / 0.454936)
    elif sigma == 'iter':
        n = data.copy()
        for _ in range(5):
            ave, sig = np.nanmean(n), np.nanstd(n)
            n = n - ave
            n = n[np.abs(n) < 3.5 * sig]
        ave = np.nanmean(n)
        noise = np.nanstd(n)
        if np.abs(ave) > 0.2 * noise:
            print('Warning: The intensity offset is larger than 0.2 sigma.')
    elif sigma == 'out':
        n, n0, n1 = data.copy(), len(data), len(data[0])
        n = np.moveaxis(n, [-2, -1], [0, 1])
        n[n0//5: n0*4//5, n1//5: n1*4//5] = np.nan
        if np.all(np.isnan(n)):
            print('sigma=\'neg\' instead of \'out\' because'
                  + ' the outer region is filled with nan.')
            noise = np.sqrt(np.nanmean(data[data < 0]**2))
        else:
            ave = np.nanmean(n)
            noise = np.nanstd(n)
            if np.abs(ave) > 0.2 * noise:
                print('Warning: The intensity offset is larger than 0.2 sigma.')
    elif sigma[:4] == 'hist':
        m0, s0 = np.nanmean(data), np.nanstd(data)
        hist, hbin = np.histogram(data[~np.isnan(data)],
                                  bins=100, density=True,
                                  range=(m0 - s0 * 5, m0 + s0 * 5))
        hist, hbin = hist * s0, (hbin[:-1] + hbin[1:]) / 2 / s0
        if sigma[4:] == '-pbcor':
            def g(x, s, c, R):
                xn = (x - c) / np.sqrt(2) / s
                return (erf(xn) - erf(xn * np.exp(-R**2))) / (2 * (x-c) * R**2)
        else:
            def g(x, s, c, R):
                return np.exp(-((x-c)/s)**2 / 2) / np.sqrt(2*np.pi) / s
        popt, _ = curve_fit(g, hbin, hist, p0=[1, 0, 1])
        ave = popt[1]
        noise = popt[0]
        if np.abs(ave) > 0.2 * noise:
            print('Warning: The intensity offset is larger than 0.2 sigma.')
        noise = noise * s0
    return noise


def trim(data: np.ndarray | None = None, x: np.ndarray | None = None,
         y: np.ndarray | None = None, v: np.ndarray | None = None,
         xlim: list[float, float] | None = None,
         ylim: list[float, float] | None = None,
         vlim: list[float, float] | None = None,
         pv: bool = False
         ) -> tuple[np.ndarray, list[np.ndarray, np.ndarray, np.ndarray]]:
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
    xout, yout, vout, dataout = x, y, v, data
    i0 = j0 = k0 = 0
    i1 = j1 = k1 = 100000
    if x is not None and xlim is not None:
        if None not in xlim:
            x0 = np.max([np.min(x), xlim[0]])
            x1 = np.min([np.max(x), xlim[1]])
            i0 = np.argmin(np.abs(x - x0))
            i1 = np.argmin(np.abs(x - x1))
            i0, i1 = sorted([i0, i1])
            xout = x[i0:i1+1]
    if y is not None and ylim is not None:
        if None not in ylim:
            y0 = np.max([np.min(y), ylim[0]])
            y1 = np.min([np.max(y), ylim[1]])
            j0 = np.argmin(np.abs(y - y0))
            j1 = np.argmin(np.abs(y - y1))
            j0, j1 = sorted([j0, j1])
            yout = y[j0:j1+1]
    if v is not None and vlim is not None:
        if None not in vlim:
            v0 = np.max([np.min(v), vlim[0]])
            v1 = np.min([np.max(v), vlim[1]])
            k0 = np.argmin(np.abs(v - v0))
            k1 = np.argmin(np.abs(v - v1))
            k0, k1 = sorted([k0, k1])
            vout = v[k0:k1+1]
    if data is not None:
        d = np.squeeze(data)
        if np.ndim(d) == 0:
            print('data has only one pixel.')
            d = data
        if np.ndim(d) == 2:
            if pv:
                j0, j1 = k0, k1
            dataout = d[j0:j1+1, i0:i1+1]
        else:
            d = np.moveaxis(d, [-3, -2, -1], [0, 1, 2])
            d = d[k0:k1+1, j0:j1+1, i0:i1+1]
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
