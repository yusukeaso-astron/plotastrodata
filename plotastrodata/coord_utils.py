import numpy as np
from astropy.coordinates import SkyCoord, FK5, FK4
from astropy import units


def _updateframe(frame: str) -> str:
    """Internal function to str frame to astropy frame.

    Args:
        frame (str): This should be one of 'J2000', 'B1950', 'FK5', 'FK4', and 'ICRS'.

    Returns:
        str: frame as is, FK5(equinox='J2000'), FK4(equinox='B1950'), or 'icrs'.
    """
    if 'ICRS' in frame:
        a = 'icrs'
    elif 'J2000' in frame or 'FK5' in frame:
        a = FK5(equinox='J2000')
    elif 'B1950' in frame or 'FK4' in frame:
        a = FK4(equinox='B1950')
    elif type(frame) is str:
        print(f'Unknown frame ({frame}) was found. Use ICRS instead.')
        a = 'icrs'
    else:
        a = frame
    return a


def _getframe(coord: str) -> tuple:
    """Internal function to pick up the frame name from the coordinates. When coord is a list, frame and framename are picked up from the first element.

    Args:
        coord (str): something like "J2000 01h23m45.6s 01d23m45.6s" or a list of them.

    Returns:
        tuple: updated coord and frame. frame is FK5(equinox='J2000), FK4(equinox='B1950'), or 'icrs'.
    """
    def getframe_single(s: str) -> tuple:
        c = s.split()
        hasframe = len(c) == 3
        hmsdms = f'{c[1]} {c[2]}' if hasframe else s
        frame = _updateframe(c[0]) if hasframe else None
        framename = c[0] if hasframe else None
        return hmsdms, frame, framename

    if type(coord) is str:
        return getframe_single(coord)
    else:
        outlist = [getframe_single(c) for c in coord]
        hmsdms = [a[0] for a in outlist]
        frame = outlist[0][1]
        framename = outlist[0][2]
        return hmsdms, frame, framename


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
    coordorg, frameorg_in, _ = _getframe(coordorg)
    frameorg = frameorg_in if frameorg is None else _updateframe(frameorg)
    coords, frame_in, _ = _getframe(coords)
    frame = frame_in if frame is None else _updateframe(frame)
    if frame is None:
        frame = frameorg
    elif frameorg is None:
        frameorg = frame
    c0 = SkyCoord(coordorg, frame=frameorg)  # frame=None means ICRS.
    if frame is not None:
        c0 = c0.transform_to(frame=frame)
    clist = SkyCoord(coords, frame=frame)  # frame=None means ICRS.
    xy = c0.spherical_offsets_to(clist)
    xy = np.array([xy[0].degree, xy[1].degree])
    return xy


def xy2coord(xy: list, coordorg: str = '00h00m00s 00d00m00s',
             frame: str | None = None, frameorg: str | None = None,
             ) -> str | np.ndarray:
    """Transform relative (deg, deg) to R.A.-Dec.

    Args:
        xy (list): [(array of) alphas, (array of) deltas] in degree. alphas and deltas can have an arbitrary shape.
        coordorg (str): something like '01h23m45.6s 01d23m45.6s'. The origin of the relative (deg, deg). Defaults to '00h00m00s 00d00m00s'.
        frame (str): coordinate frame. Defaults to None.
        frameorg (str): coordinate frame of the origin. Defaults to None.

    Returns:
        str: something like '01h23m45.6s 01d23m45.6s'. With multiple inputs, the output has the input shape.
    """
    coordorg, frameorg_in, framenameorg = _getframe(coordorg)
    frameorg = frameorg_in if frameorg is None else _updateframe(frameorg)
    framename = framenameorg if frame is None else frame
    frame = frameorg if frame is None else _updateframe(frame)
    c0 = SkyCoord(coordorg, frame=frameorg)  # frame=None means ICRS.
    coords = c0.spherical_offsets_by(*xy * units.degree)
    if frame is not None:
        coords = coords.transform_to(frame=frame)
    coords = coords.to_string('hmsdms')
    if framename is not None:
        if type(coords) is str:
            coords = f'{framename} {coords}'
        else:
            coords = np.array([f'{framename} {s}' for s in coords])
    return coords


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
