from astropy.coordinates import SkyCoord 
import numpy as np



def listing(*args) -> list:
    """Output a list of the input when the input is string or number.
    
    Returns:
        list: With a single non-list input,
              the output is a list like ['a'], rather than [['a']].
    """
    nums = [float, int, np.float64, np.int64, np.float32, np.int32]
    b = [None] * len(args)
    for i, a in enumerate(args):
        b[i] = [a] if type(a) in (nums + [str]) else a
    if len(args) == 1: b = b[0]
    return b

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
    one = np.shape(xy) == (2,)
    if one: xy = [xy]
    coords = []
    for c in xy:
        x, y = c[0] / 15., c[1] / (decsign := np.sign(c[1]))
        intx, inty = int(x), int(y)
        ra  = f'{intx:02d}h'
        dec = ('-' if decsign < 0 else '+') + f'{inty:02d}d'
        x, y = 60 * (x - intx), 60 * (y - inty)
        intx, inty = int(x), int(y)
        ra  += f'{intx:02d}m'
        dec += f'{inty:02d}m'
        x, y = 60 * (x - intx), 60 * (y - inty)
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
        if np.all(np.isnan(n)):
            print('sigma=\'neg\' instead of \'out\' because'
                  + ' the outer region is filled with nan.')
            noise = np.sqrt(np.nanmean(data[data < 0]**2))
        else:
            noise = np.nanstd(n)
    print(f'sigma = {noise:.2e}')
    return noise


def trim(data: list = None, x: list = None, y: list = None, v: list = None,
         xlim: list = None, ylim: list = None, vlim: list = None,
         pv: bool = False) -> tuple:
    """Trim 2D or 3D data by given coordinates and their limits.

    Args:
        x (list, optional): 1D array. Defaults to None.
        y (list, optional): 1D array. Defaults to None.
        v (list, optional): 1D array. Defaults to None.
        xlim (list, optional): [xmin, xmax]. Defaults to None.
        ylim (list, optional): [ymin, ymax]. Defaults to None.
        vlim (list, optional): [vmin, vmax]. Defaults to None.
        data (list, optional): 2D or 3D array. Defaults to None.

    Returns:
        tuple: Trimmed ([data, [x,y,v]). 
    """
    xout, yout, vout, dataout = x, y, v, data
    if not (x is None or xlim is None):
        if not (None in xlim):
            i0 = np.argmin(np.abs(x - xlim[0]))
            i1 = np.argmin(np.abs(x - xlim[1]))
            i0, i1 = sorted([i0, i1])
            xout = x[i0:i1+1]
    if not (y is None or ylim is None):
        if not (None in ylim):
            j0 = np.argmin(np.abs(y - ylim[0]))
            j1 = np.argmin(np.abs(y - ylim[1]))
            j0, j1 = sorted([j0, j1])
            yout = y[j0:j1+1]
    if not (v is None or vlim is None):
        if not (None in vlim):
            k0 = np.argmin(np.abs(v - vlim[0]))
            k1 = np.argmin(np.abs(v - vlim[1]))
            k0, k1 = sorted([k0, k1])
            vout = v[k0:k1+1]
    if data is not None:
        d = np.squeeze(data)
        if pv:
            dataout = d[k0:k1+1, i0:i1+1]
        elif np.ndim(d) == 2:
            dataout = d[j0:j1+1, i0:i1+1]
        else:
            dataout = d[k0:k1+1, j0:j1+1, i0:i1+1]
    return dataout, [xout, yout, vout]
