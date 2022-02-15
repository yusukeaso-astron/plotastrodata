from astropy.coordinates import SkyCoord 
import numpy as np



def coord2xy(coords: str, frame: str = 'icrs') -> list:
    """Transform R.A.-Dec. to (arcsec, arcsec).

    Args:
        coords (str): something like '01h23m45.6s 01d23m45.6s'
                      The input can be a list of str in an arbitrary shape.
        frame (str): coordinate frame. Defaults to 'icrs'.

    Returns:
        ndarray: [(array of) alphas, (array of) deltas] in degree.
                 The shape of alphas and deltas is the input shape.
                 With a single input, the output is [alpha0, delta0].
    """
    clist = np.ravel(coords)
    cx = [None] * len(clist)
    cy = [None] * len(clist)
    for i, c in enumerate(clist):
        c = SkyCoord(c, frame=frame)
        cx[i] = c.ra.degree
        cy[i] = c.dec.degree
    one = (type(coords) is str)
    shape = np.shape(coords)
    cx = cx[0] if one else np.reshape(cx, shape)
    cy = cy[0] if one else np.reshape(cy, shape)
    return np.array([cx, cy])


def xy2coord(xy: list) -> str:
    """Transform (degree, degree) to R.A.-Dec.

    Args:
        xy (list): [(array of) alphas, (array of) deltas] in degree.
                   alphas and deltas can have an arbitrary shape.

    Returns:
        str: something like '01h23m45.6s 01d23m45.6s'.
             With multiple inputs, the output has the input shape.
    """
    if one: xy = [xy]
    coords = []
    for c in xy:
        x, y = c[0] / 15., c[1]
        ra  = f'{x:02d}h'
        dec = f'{y:02d}d'
        x, y = 60 * (x - int(x)), 60 * (y - int(y))
        ra  += f'{x:02.0f}m'
        dec += f'{y:02.0f}m'
        x, y = 60 * (x - int(x)), 60 * (y - int(y))
        ra  += f'{x:09.6f}s'
        dec += f'{y:09.6f}s'
        coords.append(ra + ' ' + dec)
    shape = np.shape(xy[0])
    one = (shape == (2,))
    coords = coords[0] if one else np.reshape(coords, shape)
    return coords


def rel2abs(xrel: float, yrel: float, x: list, y: list) -> list:
    """Transform relative coordinates to absolute ones.

    Args:
        xrel (float): 0 <= xrel <= 1.
                      0 and 1 correspond to x[0] and x[-1], respectively.
                      Arbitrary shape.
        yrel (float): same as xrel.
        x (list): [x0, x0+dx, x0+2dx, ...]
        y (list): [y0, y0+dy, y0+2dy, ...]

    Returns:
        ndarray: [xabs, yabs]. Each has the input's shape.
    """
    a = (1. - xrel)*x[0] + xrel*x[-1]
    b = (1. - yrel)*y[0] + yrel*y[-1]
    return np.array([a, b])


def estimate_rms(data: list, sigma: float or str) -> float:
    """Estimate a noise level of a N-D array.

    Args:
        data (list): N-D array.
        sigma (float or str): One of the following methods.
                              When a float number is given,
                              this function just outputs it.
        methods --- 'edge': use data[0] and data[-1].
                    'neg': use only negative values.
                    'med': use the median of data^2.
                    'iter': exclude outliers.
                    'out': exclude inner 60% about axes=-2 and -1.

    Returns:
        float: the estimated room mean square of noise.
    """
    nums = [float, int, np.float64, np.int64, np.float32, np.int32]
    if type(sigma) in nums: noise = sigma
    elif sigma == 'edge': noise = np.nanstd(data[::len(data) - 1])
    elif sigma == 'neg': noise = np.sqrt(np.nanmean(data[data < 0]**2))
    elif sigma == 'med': noise = np.sqrt(np.nanmedian(data**2) / 0.454936)
    elif sigma == 'iter':
        n = data.copy()
        for _ in range(20):
            ave, sig = np.nanmean(n), np.nanstd(n)
            n = n - ave
            n = n[np.abs(n) < 3.5 * sig]
        noise = np.nanstd(n)
    elif sigma == 'out':
        n, n0, n1 = data.copy(), len(data), len(data[0])
        n = np.moveaxis(n, [-2, -1], [0, 1])
        n[n0//5 : n0*4//5, n1//5 : n1*4//5] = np.nan
        noise = np.nanstd(n)
    print(f'sigma = {noise:.2e}')
    return noise


def trim(x: list, y: list, xlim: list, ylim: list,
         v: list = None, vlim: list = None, data: list = None) -> list:
    d = np.squeeze(data)
    i0 = np.argmin(np.abs(x - xlim[0]))
    i1 = np.argmin(np.abs(x - xlim[1]))
    i0, i1 = sorted([i0, i1])
    xout = x[i0:i1+1]
    j0 = np.argmin(np.abs(y - ylim[0]))
    j1 = np.argmin(np.abs(y - ylim[1]))
    j0, j1 = sorted([j0, j1])
    yout = x[j0:j1+1]
    if not (vlim is None):
        k0 = np.argmin(np.abs(v - vlim[0]))
        k1 = np.argmin(np.abs(v - vlim[1]))
        k0, k1 = sorted([k0, k1])
        vout = v if v is None else v[k0:k1+1]
    if not (data is None):
        if np.ndim(d := np.squeeze(data)) == 2:
            dataout = d[j0:j1+1, i0:i1+1]
        else:
            dataout = d[k0:k1+1, j0:j1+1, i0:i1+1]
    return [xout, yout, vout, dataout]        


def shiftphase(F: list, u: list, v: list, dx: float, dy: float) -> list:
    """Shift the phase of 2D FFT by (dx, dy).

    Args:
        F (list): 2D FFT.
        u (list): 1D or 2D array. The first frequency coordinate.
        v (list): 1D or 2D array. The second frequency coordinate.
        dx (float): From new to old center.
        dy (float): From new to old center.

    Returns:
        ndarray: shifted FFT.
    """
    (U, V) = np.meshgrid(u, v) if np.ndim(u) == 1 else (u, v)
    return F * np.exp(1j * 2 * np.pi * (U * dx + V * dy))


def fftcentering(f: list, x: list = None, y: list = None,
                 xcyc: tuple = (0, 0)) -> list:
    """FFT with the phase referring to a specific point.

    Args:
        f (list): 2D array for FFT.
        x (list, optional): 1D or 2D array. The first spatial coordinate.
                            Defaults to None.
        y (list, optional): 1D or 2D array. The second spatial coordinate.
                            Defaults to None.
        xcyc (tuple, optional): phase reference point. Defaults to (0, 0).

    Returns:
        list: [F, u, v]. F is FFT of f.
              u and v are 1D arrays of the frequency coordinates.
    """
    ny, nx = np.shape(f)
    if x is None: x = np.arange(nx)
    if y is None: y = np.arange(ny)
    if np.ndim(x): x = x[0, :]
    if np.ndim(y): y = y[:, 0]
    u = np.fft.fftshift(np.fft.fftfreq(nx, d=x[1] - x[0]))
    v = np.fft.fftshift(np.fft.fftfreq(ny, d=y[1] - y[0]))
    F = np.fft.fftshift(np.fft.fft2(f))
    F = shiftphase(F, u, v, xcyc[0] - x[0], xcyc[1] - y[0])
    return [F, u, v]


def ifftcentering(F: list, u: list = None, v: list = None,
                  xcyc: tuple = (0, 0), x0y0: tuple = (None, None),
                  outreal: bool = True) -> list:
    """inverse FFT with the phase referring to a specific point.

    Args:
        F (list): 2D array. A result of FFT.
        u (list, optional): 1D or 2D array. The first frequency coordinate.
                            Defaults to None.
        v (list, optional): 1D or 2D array. The second frequency cooridnate.
                            Defaults to None.
        xcyc (tuple, optional): central spatial coordinates.
                                Defaults to (0, 0).
        x0y0 (tuple, optional): spatial coordinates of x[0] and y[0].
                                Defaults to (None, None).
        outreal (bool, optional): whether output only the real part.
                                  Defaults to True.

    Returns:
        list: [f, x, y]. f is iFFT of F.
              x and y are 1D arrays of the spatial coordinates.
    """
    ny, nx = np.shape(F)
    if u is None: u = np.fft.fftshift(np.fft.fftfreq(nx, d=1))
    if v is None: v = np.fft.fftshift(np.fft.fftfreq(ny, d=1))
    x = (np.arange(nx) - (nx-1)/2.) / (u[1]-u[0]) / nx + xcyc[0]
    y = (np.arange(ny) - (ny-1)/2.) / (v[1]-v[0]) / ny + xcyc[1]
    if not (x0y0[0] is None): x = x - x[0] + x0y0[0]
    if not (x0y0[1] is None): y = y - y[0] + x0y0[1]
    F = shiftphase(F, u, v, x[0] - xcyc[0], y[0] - xcyc[1])
    f = np.fft.ifft2(np.fft.ifftshift(F))
    if outreal: f = np.real(f)
    return [f, x, y]
