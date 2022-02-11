from astropy.coordinates import SkyCoord 
import numpy as np


def listing(*args) -> list:
    """Output a list of the inputs.
    
    Returns:
        list: A list or ndarray is appended as it is.
              The empy text '' is appended as the empty list [].
              Any other is appeded as a list of itself, like ['a'].
              With a single non-list input,
              the output is a list like ['a'], rather than [['a']].
              With a single list input, the output is the input itself.
    """
    one = (len(args) == 1)
    b = []
    for a in args:
        if not (type(a) in [list, np.ndarray]):
            a = [] if a == '' else [a]
        b.append(a)
    if one: b = b[0]
    return b


def coord2xy(coords: str, frame: str = 'icrs') -> list:
    """Transform R.A.-Dec. to (degree, degree).

    Args:
        coords (str): something like '01h23m45.6s 01d23m45.6s'
                      The input can be a list of str in an arbitrary shape.
        frame (str): coordinate frame. Defaults to 'icrs'.

    Returns:
        ndarray: [alphas, deltas] in degree.
                 The shape of alphas and deltas is the input shape.
                 With a single input, the output is [alpha0, delta0].
    """
    one = True if type(coords) is str else False
    sh = np.shape(coords)
    coords = listing(np.ravel(coords))
    cx, cy = [], []
    if len(coords) > 0:
        for c in coords:
            c = SkyCoord(c, frame=frame)
            cx.append(c.ra.degree)
            cy.append(c.dec.degree)
    if one:
        cx, cy = cx[0], cy[0]
    else:
        cx, cy = np.reshape(cx, sh), np.reshape(cy, sh)
    return np.array([cx, cy])


def xy2coord(xy: list) -> str:
    """Transform (degree, degree) to R.A.-Dec.

    Args:
        xy (list): list of [alphas, deltas] in degree.
                   alphas and deltas can have an arbitrary shape.

    Returns:
        str: something like '01h23m45.6s 01d23m45.6s'.
             With multiple inputs, the output has the input shape.
    """
    one = True if np.shape(xy) == (2,) else False
    sh = None
    if one:
        xy = [xy]
    else:
        sh = np.shape(xy[0])
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
    coords = coords[0] if one else np.reshape(coords, sh)
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
