import numpy as np
from astropy import constants

from plotastrodata.fits_utils import fits2data



def shiftphase(F: list, u: list, v: list = None,
               xoff: float = 0, yoff: float = 0) -> list:
    """Shift the phase of 1D or 2D FFT by (xoff, yoff).

    Args:
        F (list): 1D or 2D FFT. 1D array needs only u.
        u (list): 1D or 2D array. The first frequency coordinate.
        v (list): 1D or 2D array. The second frequency coordinate.
                  Defaults to None.
        xoff (float): From old to new center. Defaults to 0.
        yoff (float): From old to new center. Defaults to 0.

    Returns:
        ndarray: phase-shifted FFT.
    """
    dim = np.ndim(F)
    if dim == 1:
        U = u[0, :] if np.ndim(u) == 2 else u
        return F * np.exp(1j * 2 * np.pi * U * xoff)
    elif dim == 2:
        (U, V) = np.meshgrid(u, v) if np.ndim(u) == 1 else (u, v)
        return F * np.exp(1j * 2 * np.pi * (U * xoff + V * yoff))
    else:
        print(f'{dim:d}-D array is not supported.')


def fftcentering(f: list, x: list = None, y: list = None,
                 xcenter: float = 0, ycenter: float = 0) -> list:
    """FFT with the phase referring to a specific point.

    Args:
        f (list): 2D array for FFT.
        x (list, optional): 1D or 2D array. The first spatial coordinate.
                            Defaults to None.
        y (list, optional): 1D or 2D array. The second spatial coordinate.
                            Defaults to None.
        xcenter (float, optional): x of phase reference. Defaults to 0.
        ycenter (float, optional): y of phase reference. Defaults to 0.

    Returns:
        list: [F, u, v]. F is FFT of f.
              u and v are 1D arrays of the frequency coordinates.
    """
    dim = np.ndim(f)
    if dim == 1:
        nx = np.shape(f)[0]
        if x is None: x = np.arange(nx)
        X = x[0, :] if np.ndim(x) == 2 else x
        dx = X[1] - X[0]
        u = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
        F = np.fft.fftshift(np.fft.fft(f))
        F = shiftphase(F, u=u, xoff=xcenter - X[-1] - dx)
        return [F, u]
    elif dim == 2:
        ny, nx = np.shape(f)
        if x is None: x = np.arange(nx)
        if y is None: y = np.arange(ny)
        X = x[0, :] if np.ndim(x) == 2 else x
        Y = y[:, 0] if np.ndim(y) == 2 else y
        dx, dy = X[1] - X[0], Y[1] - Y[0]
        u = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
        v = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
        F = np.fft.fftshift(np.fft.fft2(f))
        F = shiftphase(F, u, v, xcenter - X[-1] - dx, ycenter - Y[-1] - dy)
        return [F, u, v]
    else:
        print(f'{dim:d}-D array is not supported.')


def ifftcentering(F: list, u: list = None, v: list = None,
                  xcenter: float = 0, ycenter: float = 0,
                  x0: float = None, y0: float = None,
                  outreal: bool = True) -> list:
    """inverse FFT with the phase referring to a specific point.

    Args:
        F (list): 2D array. A result of FFT.
        u (list, optional): 1D or 2D array. The first frequency coordinate.
                            Defaults to None.
        v (list, optional): 1D or 2D array. The second frequency cooridnate.
                            Defaults to None.
        xcenter (float, optional): x of phase reference (used in fftcentering).
                                   Defaults to (0, 0).
        ycenter (float, optional): y of phase reference (used in fftcentering).
                                   Defaults to (0, 0).
        x0 (float, optional): spatial coordinate of x[0].
                              Defaults to (None, None).
        y0 (float, optional): spatial coordinate of y[0].
                              Defaults to (None, None).
        outreal (bool, optional): whether output only the real part.
                                  Defaults to True.

    Returns:
        list: [f, x, y]. f is iFFT of F.
              x and y are 1D arrays of the spatial coordinates.
    """
    dim = np.ndim(F)
    if dim == 1:
        nx = np.shape(F)[0]
        if u is None: u = np.fft.fftshift(np.fft.fftfreq(nx, d=1))
        x = (np.arange(nx) - (nx-1)/2.) / (u[1]-u[0]) / nx + xcenter
        if x0 is not None: x = x - x[0] + x0
        dx = x[1] - x[0]
        F = shiftphase(F, u=u, xoff=x[-1] + dx - xcenter)
        f = np.fft.ifft(np.fft.ifftshift(F))
        if outreal: f = np.real(f)
        return [f, x]
    elif dim == 2:
        ny, nx = np.shape(F)
        if u is None: u = np.fft.fftshift(np.fft.fftfreq(nx, d=1))
        if v is None: v = np.fft.fftshift(np.fft.fftfreq(ny, d=1))
        x = (np.arange(nx) - (nx-1)/2.) / (u[1]-u[0]) / nx + xcenter
        y = (np.arange(ny) - (ny-1)/2.) / (v[1]-v[0]) / ny + ycenter
        if x0 is not None: x = x - x[0] + x0
        if y0 is not None: y = y - y[0] + y0
        dx, dy = x[1] - x[0], y[1] - y[0]
        F = shiftphase(F, u, v, x[-1] + dx - xcenter, y[-1] + dy - ycenter)
        f = np.fft.ifft2(np.fft.ifftshift(F))
        if outreal: f = np.real(f)
        return [f, x, y]
    else:
        print(f'{dim:d}-D array is not supported.')


def fftfits(fitsimage: str, center: str = None, restfrq: float = None) -> list:
    """FFT a fits image with the phase referring to a specific point.

    Args:
        fitsimage (str): Input fits name.
        center (str, optional): Text coordinate. Defaults to None.
        restfrq (float, optional):
            If not None, return u and v in the unit of meter.
            Defaults to None.

    Returns:
        list: [F, u, v]. F is FFT of f.
              u and v are 1D arrays in the unit of lambda.
    """
    f, (x, y, v), _, _, _ \
        = fits2data(fitsimage, center=center, restfrq=restfrq)
    arcsec = np.radians(1) / 3600.
    F, u, v = fftcentering(f, x * arcsec, y * arcsec)
    if restfrq is not None:
        lam = constants.c.to('m/s').value / restfrq
        u, v = u * lam, v * lam  # lambda -> m
    return [F, u, v]
