import numpy as np
from astropy.coordinates import SkyCoord, FK5, FK4
from astropy import units


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
