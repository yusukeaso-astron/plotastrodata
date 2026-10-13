import numpy as np
from astropy import units
from astropy.coordinates import BaseCoordinateFrame, FK4, FK5, SkyCoord


def _updateframe(frame: str) -> str | FK4 | FK5:
    """Internal function to str frame to astropy frame.

    Args:
        frame (str): This should be one of 'J2000', 'B1950', 'FK5',
            'FK4', and 'ICRS'.

    Returns:
        str or FK4 or FK5: FK5(equinox='J2000'), FK4(equinox='B1950'), or
        'icrs'.
    """
    if 'ICRS' in frame:
        a = 'icrs'
    elif 'J2000' in frame or 'FK5' in frame:
        a = FK5(equinox='J2000')
    elif 'B1950' in frame or 'FK4' in frame:
        a = FK4(equinox='B1950')
    elif isinstance(frame, str):
        print(f'Unknown frame ({frame}) was found. Use ICRS instead.')
        a = 'icrs'
    else:
        a = frame
    return a


def _getframe(coord: str | list | tuple | np.ndarray, *, as_frame: bool = False
              ) -> tuple[str | list | np.ndarray,
                         str | BaseCoordinateFrame | None,
                         str | None]:
    """Separate coordinate strings from their optional frame prefixes.

    Args:
        coord: A string such as "J2000 01h23m45.6s 01d23m45.6s", or a
            rectangular list, tuple, or array of strings of arbitrary shape.
            Collections use the first element's frame; all entries should
            describe coordinates in that same frame.
        as_frame: If True, return a populated SkyCoord.frame as the second
            result, preserving the coordinate shape. Defaults to False.

    Returns:
        tuple: (hmsdms, frame, framename). hmsdms is a string for scalar
        input, an array for array input, or a nested list for list/tuple
        input, preserving the input shape. By default, frame is
        FK5(equinox='J2000'), FK4(equinox='B1950'), 'icrs', or None when no
        prefix is given. With as_frame=True, it is a populated coordinate
        frame (ICRS by default). framename is the first original prefix or
        None. Empty collections have no prefix.
    """
    values = np.asarray(coord, dtype=object)
    parsed = []
    frame, framename = None, None
    for i, value in enumerate(values.flat):
        if not isinstance(value, str):
            raise ValueError('Coordinates must be strings'
                             + ' in a rectangular collection.')
        parts = value.split()
        hasframe = len(parts) == 3
        parsed.append(f'{parts[1]} {parts[2]}' if hasframe else value)
        if i == 0 and hasframe:
            frame = _updateframe(parts[0])
            framename = parts[0]
    hmsdms = np.asarray(parsed, dtype=str).reshape(values.shape)
    if values.ndim == 0 or not isinstance(coord, np.ndarray):
        hmsdms = hmsdms.tolist()
    if as_frame:
        if values.ndim == 0:
            frame = SkyCoord(hmsdms, frame=frame).frame
        else:
            coords = (SkyCoord(parsed, frame=frame) if parsed else
                      SkyCoord([], [], unit='deg', frame=frame))
            frame = coords.reshape(values.shape).frame
    return hmsdms, frame, framename


def coord2xy(coords: str | list | tuple | np.ndarray,
             coordorg: str = '00h00m00s 00d00m00s',
             frame: str | None = None, frameorg: str | None = None,
             ) -> np.ndarray:
    """Transform R.A.-Dec. to relative (deg, deg).

    Args:
        coords (str, list, tuple, or np.ndarray): Strings such as
            '01h23m45.6s 01d23m45.6s', in any rectangular shape. Collections
            use the first coordinate's frame prefix.
        coordorg (str, optional): something like '01h23m45.6s
            01d23m45.6s'. The origin of the relative (deg, deg).
            Defaults to '00h00m00s 00d00m00s'.
        frame (str, optional): coordinate frame. Defaults to None.
        frameorg (str, optional): coordinate frame of the origin.
            Defaults to None.

    Returns:
        np.ndarray: [(array of) alphas, (array of) deltas] in degree.
        The shape of alphas and deltas is the input shape. With a single
        input, the output is [alpha0, delta0].
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
    values = np.asarray(coords)
    if values.size == 0:
        return np.empty((2,) + values.shape)
    clist = SkyCoord(values.ravel().tolist(), frame=frame)
    xy = c0.spherical_offsets_to(clist)
    xy = np.array([xy[0].degree, xy[1].degree])
    return xy.reshape((2,) + values.shape)


def xy2coord(xy: list | tuple | np.ndarray,
             coordorg: str = '00h00m00s 00d00m00s',
             frame: str | None = None, frameorg: str | None = None,
             ) -> str | np.ndarray:
    """Transform relative (deg, deg) to R.A.-Dec.

    Args:
        xy (list, tuple, or np.ndarray): [alphas, deltas] in degrees,
            with shape (2, ...). Both coordinates must have the same shape.
        coordorg (str): something like '01h23m45.6s 01d23m45.6s'. The
            origin of the relative (deg, deg). Defaults to '00h00m00s
            00d00m00s'.
        frame (str): coordinate frame. Defaults to None.
        frameorg (str): coordinate frame of the origin. Defaults to
            None.

    Returns:
        str or np.ndarray: A coordinate string for a scalar pair, or an
        array of strings with shape xy.shape[1:]. Frame prefixes are applied
        to each coordinate individually, including for multidimensional arrays.
    """
    coordorg, frameorg_in, framenameorg = _getframe(coordorg)
    frameorg = frameorg_in if frameorg is None else _updateframe(frameorg)
    framename = framenameorg if frame is None else frame
    frame = frameorg if frame is None else _updateframe(frame)
    c0 = SkyCoord(coordorg, frame=frameorg)  # frame=None means ICRS.
    offsets = np.asarray(xy)
    if offsets.ndim == 0 or offsets.shape[0] != 2:
        raise ValueError('xy must have shape (2, ...)'
                         + ' with matching coordinate shapes.')
    shape = offsets.shape[1:]
    if offsets.size == 0:
        return np.empty(shape, dtype=str)
    coords = c0.spherical_offsets_by(*(offsets.reshape(2, -1) * units.degree))
    if frame is not None:
        coords = coords.transform_to(frame=frame)
    strings = coords.to_string('hmsdms')
    if framename is not None:
        strings = [f'{framename} {value}' for value in strings]
    result = np.asarray(strings).reshape(shape)
    return result.item() if result.ndim == 0 else result


def rel2abs(xrel: float | np.ndarray, yrel: float | np.ndarray,
            x: np.ndarray | list[float], y: np.ndarray | list[float]
            ) -> np.ndarray:
    """Transform relative coordinates to absolute ones.

    Args:
        xrel (float or np.ndarray): 0 <= xrel <= 1. 0 and 1 correspond to x[0] and
            x[-1], respectively. Arbitrary shape.
        yrel (float or np.ndarray): same as xrel.
        x (np.ndarray or list): [x0, x0+dx, x0+2dx, ...]
        y (np.ndarray or list): [y0, y0+dy, y0+2dy, ...]

    Returns:
        np.ndarray: [xabs, yabs]. Each has the input's shape.
    """
    xabs = (1. - xrel)*x[0] + xrel*x[-1]
    yabs = (1. - yrel)*y[0] + yrel*y[-1]
    return np.array([xabs, yabs])


def abs2rel(xabs: float | np.ndarray, yabs: float | np.ndarray,
            x: np.ndarray | list[float], y: np.ndarray | list[float]
            ) -> np.ndarray:
    """Transform absolute coordinates to relative ones.

    Args:
        xabs (float or np.ndarray): In the same frame of x.
        yabs (float or np.ndarray): In the same frame of y.
        x (np.ndarray or list): [x0, x0+dx, x0+2dx, ...]
        y (np.ndarray or list): [y0, y0+dy, y0+2dy, ...]

    Returns:
        ndarray: [xrel, yrel]. Each has the input's shape. 0 <= xrel,
        yrel <= 1. 0 and 1 correspond to x[0] and x[-1], respectively.
    """
    xrel = (xabs - x[0]) / (x[-1] - x[0])
    yrel = (yabs - y[0]) / (y[-1] - y[0])
    return np.array([xrel, yrel])


def get_sec(coord: str, mode: str) -> str:
    """Pick up the second number from a hmsdms string.

    Args:
        coord (str): hmsdms string.
        mode (str): 'ra' or 'dec'

    Returns:
        str: The second number as a string without the unit.
    """
    i_axis = 0 if mode == 'ra' else 1
    return coord.split(' ')[i_axis].split('m')[1].strip('s')


def get_min(coord: str, mode: str) -> str:
    """Pick up the minute number from a hmsdms string.

    Args:
        coord (str): hmsdms string.
        mode (str): 'ra' or 'dec'

    Returns:
        str: The minute number as a string without the unit.
    """
    i_axis = 0 if mode == 'ra' else 1
    s = 'h' if mode == 'ra' else 'd'
    return coord.split(' ')[i_axis].split(s)[1].split('m')[0]


def get_hmdm(coord: str, mode: str) -> str:
    """Pick up the coordinate string before the second part from a
    hsmdms string.

    Args:
        coord (str): hmsdms string.
        mode (str): 'ra' or 'dec'

    Returns:
        str: The hm or dm string with the units.
    """
    i_axis = 0 if mode == 'ra' else 1
    return coord.split(' ')[i_axis].split('m')[0] + 'm'
