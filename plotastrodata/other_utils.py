import subprocess
import shlex
import numpy as np
from astropy.coordinates import SkyCoord, FK5, FK4
from astropy import units
from scipy.optimize import curve_fit
from scipy.special import erf

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


def listing(*args) -> list:
    """Output a list of the input when the input is string or number.

    Returns:
        list: With a single non-list input, the output is a list like ['a'], rather than [['a']].
    """
    nums = [float, int, np.float64, np.int64, np.float32, np.int32]
    b = [None] * len(args)
    for i, a in enumerate(args):
        b[i] = [a] if type(a) in (nums + [str]) else a
    if len(args) == 1:
        b = b[0]
    return b


def _getframe(coord: str, s: str = '') -> tuple:
    """Internal function to pick up the frame name from the coordinates.

    Args:
        coord (str): something like "J2000 01h23m45.6s 01d23m45.6s"
        s (str, optional): To distinguish coord and coordorg. Defaults to ''.

    Returns:
        tuple: updated coord and frame. frame is FK5(equinox='J2000), FK4(equinox='B1950'), or 'icrs'.
    """
    if len(c := coord.split()) == 3:
        coord = f'{c[1]} {c[2]}'
        if 'J2000' in c[0]:
            frame = FK5(equinox='J2000')
        elif 'FK5' in c[0]:
            frame = FK5(equinox='J2000')
        elif 'B1950' in c[0]:
            frame = FK4(equinox='B1950')
        elif 'FK4' in c[0]:
            frame = FK4(equinox='B1950')
        elif 'ICRS' in c[0]:
            frame = 'icrs'
        else:
            print(f'Unknown equinox found in coord{s}. ICRS is used')
            frame = 'icrs'
    else:
        frame = None
    return coord, frame


def _updateframe(frame: str) -> str:
    """Internal function to str frame to astropy frame.

    Args:
        frame (str): _description_

    Returns:
        str: frame as is, FK5(equinox='J2000'), FK4(equinox='B1950'), or 'icrs'.
    """
    if 'ICRS' in frame:
        a = 'icrs'
    elif 'J2000' in frame or 'FK5' in frame:
        a = FK5(equinox='J2000')
    elif 'B1950' in frame or 'FK4' in frame:
        a = FK4(equinox='B1950')
    else:
        a = frame
    return a


def coord2xy(coords: str | list, coordorg: str = '00h00m00s 00d00m00s',
             frame: str | None = None, frameorg: str | None = None,
             ) -> np.ndarray:
    """Transform R.A.-Dec. to relative (deg, deg).

    Args:
        coords (str, list): something like '01h23m45.6s 01d23m45.6s'. The input can be a list of str in an arbitrary shape.
        coordorg (str, optional): something like '01h23m45.6s 01d23m45.6s'. The origin of the relative (deg, deg). Defaults to '00h00m00s 00d00m00s'.
        frame (str, optional): coordinate frame. Defaults to None.
        frameorg (str, optional): coordinate frame of the origin. Defaults to None.

    Returns:
        np.ndarray: [(array of) alphas, (array of) deltas] in degree. The shape of alphas and deltas is the input shape. With a single input, the output is [alpha0, delta0].
    """
    coordorg, frameorg_c = _getframe(coordorg, 'org')
    frameorg = frameorg_c if frameorg is None else _updateframe(frameorg)
    if type(coords) is list:
        for i in range(len(coords)):
            coords[i], frame_c = _getframe(coords[i])
    else:
        coords, frame_c = _getframe(coords)
    frame = frame_c if frame is None else _updateframe(frame)
    if frame is None and frameorg is not None:
        frame = frameorg
    if frame is not None and frameorg is None:
        frameorg = frame
    if frame is None and frameorg is None:
        frame = frameorg = 'icrs'
    clist = SkyCoord(coords, frame=frame)
    c0 = SkyCoord(coordorg, frame=frameorg)
    c0 = c0.transform_to(frame=frame)
    xy = c0.spherical_offsets_to(clist)
    return np.array([xy[0].degree, xy[1].degree])


def xy2coord(xy: list, coordorg: str = '00h00m00s 00d00m00s',
             frame: str | None = None, frameorg: str | None = None,
             ) -> str:
    """Transform relative (deg, deg) to R.A.-Dec.

    Args:
        xy (list): [(array of) alphas, (array of) deltas] in degree. alphas and deltas can have an arbitrary shape.
        coordorg (str): something like '01h23m45.6s 01d23m45.6s'. The origin of the relative (deg, deg). Defaults to '00h00m00s 00d00m00s'.
        frame (str): coordinate frame. Defaults to None.
        frameorg (str): coordinate frame of the origin. Defaults to None.

    Returns:
        str: something like '01h23m45.6s 01d23m45.6s'. With multiple inputs, the output has the input shape.
    """
    coordorg, frameorg_c = _getframe(coordorg, 'org')
    frameorg = frameorg_c if frameorg is None else _updateframe(frameorg)
    if frameorg is None:
        frameorg = 'icrs'
    frame = frameorg if frame is None else _updateframe(frame)
    c0 = SkyCoord(coordorg, frame=frameorg)
    coords = c0.spherical_offsets_by(*xy * units.degree)
    coords = coords.transform_to(frame=frame)
    return coords.to_string('hmsdms')


def rel2abs(xrel: float, yrel: float,
            x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Transform relative coordinates to absolute ones.

    Args:
        xrel (float): 0 <= xrel <= 1. 0 and 1 correspond to x[0] and x[-1], respectively. Arbitrary shape.
        yrel (float): same as xrel.
        x (np.ndarray): [x0, x0+dx, x0+2dx, ...]
        y (np.ndarray): [y0, y0+dy, y0+2dy, ...]

    Returns:
        np.ndarray: [xabs, yabs]. Each has the input's shape.
    """
    xabs = (1. - xrel)*x[0] + xrel*x[-1]
    yabs = (1. - yrel)*y[0] + yrel*y[-1]
    return np.array([xabs, yabs])


def abs2rel(xabs: float, yabs: float,
            x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Transform absolute coordinates to relative ones.

    Args:
        xabs (float): In the same frame of x.
        yabs (float): In the same frame of y.
        x (np.ndarray): [x0, x0+dx, x0+2dx, ...]
        y (np.ndarray): [y0, y0+dy, y0+2dy, ...]

    Returns:
        ndarray: [xrel, yrel]. Each has the input's shape. 0 <= xrel, yrel <= 1. 0 and 1 correspond to x[0] and x[-1], respectively.
    """
    xrel = (xabs - x[0]) / (x[-1] - x[0])
    yrel = (yabs - y[0]) / (y[-1] - y[0])
    return np.array([xrel, yrel])


def estimate_rms(data: np.ndarray, sigma: float | str | None = 'hist') -> float:
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
         pv: bool = False) -> tuple[np.ndarray, list[np.ndarray, np.ndarray, np.ndarray]]:
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
    if not (x is None or xlim is None):
        if not (None in xlim):
            x0 = np.max([np.min(x), xlim[0]])
            x1 = np.min([np.max(x), xlim[1]])
            i0 = np.argmin(np.abs(x - x0))
            i1 = np.argmin(np.abs(x - x1))
            i0, i1 = sorted([i0, i1])
            xout = x[i0:i1+1]
    if not (y is None or ylim is None):
        if not (None in ylim):
            y0 = np.max([np.min(y), ylim[0]])
            y1 = np.min([np.max(y), ylim[1]])
            j0 = np.argmin(np.abs(y - y0))
            j1 = np.argmin(np.abs(y - y1))
            j0, j1 = sorted([j0, j1])
            yout = y[j0:j1+1]
    if not (v is None or vlim is None):
        if not (None in vlim):
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


def Mfac(f0: float = 1, f1: float = 1) -> np.ndarray:
    """2 x 2 matrix for (x,y) --> (f0 * x, f1 * y).

    Args:
        f0 (float, optional): Defaults to 1.
        f1 (float, optional): Defaults to 1.

    Returns:
        np.ndarray: Matrix for the multiplication.
    """
    return np.array([[f0, 0], [0, f1]])


def Mrot(pa: float = 0) -> np.ndarray:
    """2 x 2 matrix for rotation.

    Args:
        pa (float, optional): How many degrees are the image rotated by. Defaults to 0.

    Returns:
        np.ndarray: Matrix for the rotation.
    """
    p = np.radians(pa)
    return np.array([[np.cos(p), -np.sin(p)], [np.sin(p),  np.cos(p)]])


def dot2d(M: np.ndarray = [[1, 0], [0, 1]],
          a: np.ndarray = [0, 0]) -> np.ndarray:
    """To maltiply a 2 x 2 matrix to (x,y) with arrays of x and y.

    Args:
        M (np.ndarray, optional): 2 x 2 matrix. Defaults to [[1, 0], [0, 1]].
        a (np.ndarray, optional): 2D vector (of 1D arrays). Defaults to [0].

    Returns:
        np.ndarray: The 2D vector after the matrix multiplied.
    """
    x = M[0, 0] * np.array(a[0]) + M[0, 1] * np.array(a[1])
    y = M[1, 0] * np.array(a[0]) + M[1, 1] * np.array(a[1])
    return np.array([x, y])


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


def gaussian2d(xy: np.ndarray,
               amplitude: float, xo: float, yo: float,
               fwhm_major: float, fwhm_minor: float, pa: float) -> np.ndarray:
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
    s, t = dot2d(Mrot(pa), [xy[1] - yo, xy[0] - xo])
    g = amplitude * np.exp2(-4 * ((s / fwhm_major)**2 + (t / fwhm_minor)**2))
    return g
