import numpy as np



def obs2sys(xobs: np.dnarray, yobs: np.ndarray, zobs: np.ndarray,
            pa: float = 0, incl: float = 0, polar: bool = False) -> np.ndarray:
    """Convert observed coordinates to system coordinates.

    Args:
        xobs (np.dnarray): Observed x-coordinates. The distance to the east.
        yobs (np.ndarray): Observed y-coordinates. The distance to the north.
        zobs (np.ndarray): Observed z-coordinates. The line-of-sight distance.
        pa (float, optional): Position angle of the system in radian from yobs (north) to xobs (east). Defaults to 0.
        incl (float, optional): Inclination of the system in radian. i=0 means face-on. Defaults to 0.
        polar (bool, optional): If True, the coordinates are in polar coordinates. Defaults to False.

    Returns:
        np.ndarray: System x, y, z coordinates or r, theta, phi coordinates. The polar coordinates are in radian.
    """
    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)
    cos_incl = np.cos(incl)
    sin_incl = np.sin(incl)
    
    xsys = cos_pa * xobs - sin_pa * yobs
    ysys = cos_incl * sin_pa * xobs + cos_incl * cos_pa * yobs + sin_incl * zobs
    zsys = -sin_incl * sin_pa * xobs - sin_incl * cos_pa * yobs + cos_incl * zobs

    if polar:
        r = np.sqrt(xsys**2 + ysys**2 + zsys**2)
        theta = np.arccos(zsys / r)
        phi = np.arctan2(ysys, xsys)
        return np.array([r, theta, phi])
    else:
        return np.array([xsys, ysys, zsys])

def polarvel2losvel(v_r: np.ndarray, v_theta: np.ndarray, v_phi: np.ndaray,
                    theta: np.ndarray, phi: np.ndarray, incl: float = 0) -> np.ndarray:
    """Convert the polar velocities to the line-of-sight velocity.

    Args:
        v_r (np.ndarray): The velocity component in the radial direction.
        v_theta (np.ndarray): The velocity component in the polar angle direction.
        v_phi (np.ndaray): The velocity component in the azimuthal angle direction.
        theta (np.ndarray): The polar angle in radian from the z-axis.
        phi (np.ndarray): The azimuthal angle in radian from the x-axis.
        incl (float, optional): The inclination angle of the system in radian. i=0 means face-on. Defaults to 0.

    Returns:
        np.ndarray: The line-of-sight velocity.
    """
    cos_i = np.cos(incl)
    sin_i = np.sin(incl)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    
    v_los = (sin_i * sin_t * sin_p + cos_i * cos_t) * v_r \
            + (sin_i * cos_t * sin_p - cos_i * sin_t) * v_theta \
            + sin_i * cos_p * v_phi
    return v_los
