import numpy as np


def Mrot3d(t: float, axis: int = 3) -> np.ndarray:
    """3D rotation matrix around a specified axis.

    This function creates a 3x3 rotation matrix for rotating coordinates around
    the x-axis (axis=1), y-axis (axis=2), or z-axis (axis=3) by t degrees.

    Args:
        t (float): Rotation angle in degrees.
        axis (int, optional): Axis to rotate around - 1 for x-axis, 2 for y-axis, 3 for z-axis. Defaults to 3.

    Returns:
        np.ndarray: 3x3 rotation matrix that rotates coordinates around the specified axis by t degrees.
    """
    cos_t = np.cos(np.radians(t))
    sin_t = np.sin(np.radians(t))
    match axis:
        case  1:
            m = [[1, 0, 0],
                 [0, cos_t, -sin_t],
                 [0, sin_t, cos_t]]
        case 2:
            m = [[cos_t, 0, sin_t],
                 [0, 1, 0],
                 [-sin_t, 0, cos_t]]
        case 3:
            m = [[cos_t, -sin_t, 0],
                 [sin_t, cos_t, 0],
                 [0, 0, 1]]
    return m


def obs2sys(xobs: np.ndarray, yobs: np.ndarray, zobs: np.ndarray,
            pa: float = 0, incl: float = 0, phi0: float = 0, theta0: float = 90,
            polar: bool = False) -> np.ndarray:
    """Convert observed coordinates to system coordinates. In the system coordinates, the observer is at the direction of (0, -sin i, cos i). The observer's +z (i.e., line-of-sight) is from the observer to the system center. The system's x coordinate and the observer's x coordinate have opposite signs.

    Args:
        xobs (np.ndarray): Observed x-coordinates. The distance to the east.
        yobs (np.ndarray): Observed y-coordinates. The distance to the north.
        zobs (np.ndarray): Observed z-coordinates. The line-of-sight distance.
        pa (float, optional): Position angle of the system in degrees from yobs (north) to xobs (east). Defaults to 0.
        incl (float, optional): Inclination of the system in degrees. i=0 means face-on. Defaults to 0.
        phi0 (float, optional): Azimuthal angle of the system in degrees, relative to the system that is observed. Defaults to 0.
        theta0 (float, optional): Polar angle of the x-axis of the system in degrees, relative to the x-axis of the system that is observed. Defaults to pi/2.
        polar (bool, optional): If True, the coordinates are in polar coordinates, where theta and phi are in radian. Defaults to False.

    Returns:
        np.ndarray: System x, y, z coordinates ([xsys, ysys, zsys]) or r, theta, phi coordinates ([r, theta, phi]). The polar coordinates are in radian.
    """
    x = np.array([xobs, yobs, zobs])
    x = np.tensordot(Mrot3d(pa, axis=3), x, axes=([1], [0]))
    x = np.tensordot(Mrot3d(-incl, axis=1), x, axes=([1], [0]))
    x = np.tensordot(np.diag([-1, 1, -1]), x, axes=([1], [0]))
    x = np.tensordot(Mrot3d(-phi0, axis=3), x, axes=([1], [0]))
    x = np.tensordot(Mrot3d(90 - theta0, axis=2), x, axes=([1], [0]))

    if polar:
        xsys, ysys, zsys = x
        r = np.sqrt(xsys**2 + ysys**2 + zsys**2)
        theta = np.arccos(zsys / r)
        phi = np.arctan2(ysys, xsys)
        return np.array([r, theta, phi])
    else:
        return x


def sys2obs(xsys: np.ndarray, ysys: np.ndarray, zsys: np.ndarray,
            pa: float = 0, incl: float = 0, phi0: float = 0, theta0: float = 90,
            polar: bool = False) -> np.ndarray:
    """Convert system coordinates to observed coordinates. In the system coordinates, the observer is at the direction of (0, -sin i, cos i). The observer's +z (i.e., line-of-sight) is from the observer to the system center. The system's x coordinate and the observer's x coordinate have opposite signs.

    Args:
        xsys (np.ndarray): System x-coordinates (or r).
        ysys (np.ndarray): System y-coordinates (or theta).
        zsys (np.ndarray): System z-coordinates (or phi).
        pa (float, optional): Position angle of the system in degrees from yobs (north) to xobs (east). Defaults to 0.
        incl (float, optional): Inclination of the system in degrees. i=0 means face-on. Defaults to 0.
        phi0 (float, optional): Azimuthal angle of the system in degrees, relative to the system that is observed. Defaults to 0.
        theta0 (float, optional): Polar angle of the x-axis of the system in degrees, relative to the x-axis of the system that is observed. Defaults to pi/2.
        polar (bool, optional): If True, the coordinates are in polar coordinates, where theta and phi are in radian. Defaults to False.

    Returns:
        np.ndarray: Observed x, y, z coordinates ([xobs, yobs, zobs]).
    """
    if polar:
        r, theta, phi = xsys, ysys, zsys
        x = np.array([r * np.sin(theta) * np.cos(phi),
                      r * np.sin(theta) * np.sin(phi),
                      r * np.cos(theta)])
    else:
        x = np.array([xsys, ysys, zsys])

    x = np.tensordot(Mrot3d(theta0 - 90, axis=2), x, axes=([1], [0]))
    x = np.tensordot(Mrot3d(phi0, axis=3), x, axes=([1], [0]))
    x = np.tensordot(np.diag([-1, 1, -1]), x, axes=([1], [0]))
    x = np.tensordot(Mrot3d(incl, axis=1), x, axes=([1], [0]))
    x = np.tensordot(Mrot3d(-pa, axis=3), x, axes=([1], [0]))
    return x


def polarvel2losvel(v_r: np.ndarray, v_theta: np.ndarray, v_phi: np.ndarray,
                    theta: np.ndarray, phi: np.ndarray,
                    incl: float = 0, phi0: float = 0, theta0: float = 90) -> np.ndarray:
    """Convert the polar velocities in the system's coordinates to the line-of-sight velocity in the observer's coordinates. In the system coordinates, the observer is at the direction of (0, -sin i, cos i). The observer's +z (i.e., line-of-sight) is from the observer to the system. The system's x coordinate and the observer's x coordinate have opposite signs.

    Args:
        v_r (np.ndarray): The velocity component in the radial direction.
        v_theta (np.ndarray): The velocity component in the polar angle direction.
        v_phi (np.ndarray): The velocity component in the azimuthal angle direction.
        theta (np.ndarray): The polar angle in radian from the z-axis.
        phi (np.ndarray): The azimuthal angle in radian from the x-axis.
        incl (float, optional): Inclination of the system in degrees. i=0 means face-on. Defaults to 0.
        phi0 (float, optional): Azimuthal angle of the system in degrees, relative to the system that is observed. Defaults to 0.
        theta0 (float, optional): Polar angle of the x-axis of the system in degrees, relative to the x-axis of the system that is observed. Defaults to pi/2.

    Returns:
        np.ndarray: The line-of-sight velocity.
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)

    A = np.array([[sin_t * cos_p, cos_t * cos_p, -sin_p],
                 [sin_t * sin_p, cos_t * sin_p, cos_p],
                 [cos_t, -sin_t, np.zeros_like(theta)]])
    A = np.tensordot(Mrot3d(theta0 - 90, axis=2), A, axes=([1], [0]))
    A = np.tensordot(Mrot3d(phi0, axis=3), A, axes=([1], [0]))
    A = np.tensordot(np.diag([-1, 1, -1]), A, axes=([1], [0]))
    A = np.tensordot(Mrot3d(incl, axis=1), A, axes=([1], [0]))
    v_los = A[2, 0] * v_r + A[2, 1] * v_theta + A[2, 2] * v_phi
    return v_los
