import numpy as np


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
    x = M[0][0] * np.array(a[0]) + M[0][1] * np.array(a[1])
    y = M[1][0] * np.array(a[0]) + M[1][1] * np.array(a[1])
    return np.array([x, y])


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
