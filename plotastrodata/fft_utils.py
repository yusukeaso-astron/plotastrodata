import numpy as np
import matplotlib.pyplot as plt

from plotastrodata.fits_utils import fits2data
from plotastrodata.plot_utils import set_rcparams


def shiftphase(F: np.ndarray, u: np.ndarray, xoff: float = 0) -> np.ndarray:
    """Shift the phase of 1D FFT by xoff.

    Args:
        F (np.ndarray): 1D FFT.
        u (np.ndarray): 1D array. The first frequency coordinate.
        xoff (float): From old to new center. Defaults to 0.

    Returns:
        np.ndarray: phase-shifted FFT.
    """
    return F * np.exp(1j * 2 * np.pi * u * xoff)


def shiftphase2(F: np.ndarray, u: np.ndarray, v: np.ndarray,
                xoff: float = 0, yoff: float = 0) -> np.ndarray:
    """Shift the phase of 2D FFT by (xoff, yoff).

    Args:
        F (np.ndarray): 2D FFT.
        u (np.ndarray): 1D or 2D array. The first frequency coordinate.
        v (np.ndarray): 1D or 2D array. The second frequency coordinate. Defaults to None.
        xoff (float): From old to new center. Defaults to 0.
        yoff (float): From old to new center. Defaults to 0.

    Returns:
        np.ndarray: phase-shifted FFT.
    """
    (U, V) = np.meshgrid(u, v) if np.ndim(u) == 1 else (u, v)
    return F * np.exp(1j * 2 * np.pi * (U * xoff + V * yoff))


def fftcentering(f: np.ndarray, x: np.ndarray | None = None,
                 xcenter: float = 0
                 ) -> tuple[np.ndarray, np.ndarray]:
    """FFT with the phase referring to a specific point.

    Args:
        f (np.ndarray): 1D array for FFT.
        x (np.ndarray, optional): 1D array. The spatial coordinate. Defaults to None.
        xcenter (float, optional): x of phase reference. Defaults to 0.

    Returns:
        tuple: (F, u). F is FFT of f. u is a 1D array of the frequency coordinate.
    """
    nx = np.shape(f)[0]
    if x is None:
        x = np.arange(nx)
    X = x[0, :] if np.ndim(x) == 2 else x
    dx = X[1] - X[0]
    u = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    F = np.fft.fftshift(np.fft.fft(f))
    F = shiftphase(F, u=u, xoff=xcenter - X[-1] - dx)
    return F, u


def fftcentering2(f: np.ndarray,
                  x: np.ndarray | None = None, y: np.ndarray | None = None,
                  xcenter: float = 0, ycenter: float = 0
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FFT with the phase referring to a specific point.

    Args:
        f (np.ndarray): 2D array for FFT.
        x (np.ndarray, optional): 1D or 2D array. The first spatial coordinate. Defaults to None.
        y (np.ndarray, optional): 1D or 2D array. The second spatial coordinate. Defaults to None.
        xcenter (float, optional): x of phase reference. Defaults to 0.
        ycenter (float, optional): y of phase reference. Defaults to 0.

    Returns:
        tuple: (F, u, v). F is FFT of f. u and v are 1D arrays of the frequency coordinates.
    """
    ny, nx = np.shape(f)
    if x is None:
        x = np.arange(nx)
    if y is None:
        y = np.arange(ny)
    X = x[0, :] if np.ndim(x) == 2 else x
    Y = y[:, 0] if np.ndim(y) == 2 else y
    dx, dy = X[1] - X[0], Y[1] - Y[0]
    u = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    v = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    F = np.fft.fftshift(np.fft.fft2(f))
    F = shiftphase2(F, u, v, xcenter - X[-1] - dx, ycenter - Y[-1] - dy)
    return F, u, v


def ifftcentering(F: np.ndarray, u: np.ndarray | None = None,
                  xcenter: float = 0, x0: float = None, outreal: bool = True
                  ) -> tuple[np.ndarray, np.ndarray]:
    """inverse FFT with the phase referring to a specific point.

    Args:
        F (np.ndarray): 1D array. A result of FFT.
        u (np.ndarray, optional): 1D array. The frequency coordinate. Defaults to None.
        xcenter (float, optional): x of phase reference (used in fftcentering). Defaults to 0.
        x0 (float, optional): spatial coordinate of x[0]. Defaults to None.
        outreal (bool, optional): whether output only the real part. Defaults to True.

    Returns:
        tuple: (f, x). f is iFFT of F. x is a 1D array of the spatial coordinate.
    """
    nx = np.shape(F)[0]
    if u is None:
        u = np.fft.fftshift(np.fft.fftfreq(nx, d=1))
    x = (np.arange(nx) - (nx-1)/2.) / (u[1]-u[0]) / nx + xcenter
    if x0 is not None:
        x = x - x[0] + x0
    dx = x[1] - x[0]
    F = shiftphase(F, u=u, xoff=x[-1] + dx - xcenter)
    f = np.fft.ifft(np.fft.ifftshift(F))
    if outreal:
        f = np.real(f)
    return f, x


def ifftcentering2(F: np.ndarray,
                   u: np.ndarray | None = None, v: np.ndarray | None = None,
                   xcenter: float = 0, ycenter: float = 0,
                   x0: float | None = None, y0: float | None = None,
                   outreal: bool = True
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """inverse FFT with the phase referring to a specific point.

    Args:
        F (np.ndarray): 2D array. A result of FFT.
        u (np.ndarray, optional): 1D or 2D array. The first frequency coordinate. Defaults to None.
        v (np.ndarray, optional): 1D or 2D array. The second frequency cooridnate. Defaults to None.
        xcenter (float, optional): x of phase reference (used in fftcentering2). Defaults to 0.
        ycenter (float, optional): y of phase reference (used in fftcentering2). Defaults to 0.
        x0 (float, optional): spatial coordinate of x[0]. Defaults to None.
        y0 (float, optional): spatial coordinate of y[0]. Defaults to None.
        outreal (bool, optional): whether output only the real part. Defaults to True.

    Returns:
        tuple: (f, x, y). f is iFFT of F. x and y are 1D arrays of the spatial coordinates.
    """
    ny, nx = np.shape(F)
    if u is None:
        u = np.fft.fftshift(np.fft.fftfreq(nx, d=1))
    if v is None:
        v = np.fft.fftshift(np.fft.fftfreq(ny, d=1))
    x = (np.arange(nx) - (nx-1)/2.) / (u[1]-u[0]) / nx + xcenter
    y = (np.arange(ny) - (ny-1)/2.) / (v[1]-v[0]) / ny + ycenter
    if x0 is not None:
        x = x - x[0] + x0
    if y0 is not None:
        y = y - y[0] + y0
    dx, dy = x[1] - x[0], y[1] - y[0]
    F = shiftphase(F, u, v, x[-1] + dx - xcenter, y[-1] + dy - ycenter)
    f = np.fft.ifft2(np.fft.ifftshift(F))
    if outreal:
        f = np.real(f)
    return f, x, y


def zeropadding(f: np.ndarray, x: np.ndarray, y: np.ndarray,
                xlim: list, ylim: list
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad an outer region with zero.

    Args:
        f (np.ndarray): Input 2D array.
        x (np.ndarray): 1D array.
        y (np.ndarray): 1D array.
        xlim (list): range of x after the zero padding.
        ylim (list): range of y after the zero padding.

    Returns:
        tuple: (fnew, xnew, ynew). fnew is an 2D array and xnew and ynew are 1D arrays after the zero padding.
    """
    nx, ny = len(x), len(y)
    dx, dy = x[1] - x[0], y[1] - y[0]
    if dx < 0:
        xlim = [xlim[1], xlim[0]]
    if dy < 0:
        ylim = [ylim[1], ylim[0]]
    nx0 = max(int((x[0] - xlim[0]) / dx), 0)
    nx1 = max(int((xlim[1] - x[-1]) / dx), 0)
    nxnew = nx0 + nx + nx1
    xnew = np.linspace(x[0] - nx0*dx, x[-1] + nx1*dx, nxnew)
    ny0 = max(int((y[0] - ylim[0]) / dy), 0)
    ny1 = max(int((ylim[1] - y[-1]) / dy), 0)
    nynew = ny0 + ny + ny1
    ynew = np.linspace(y[0] - ny0*dy, y[-1] + ny1*dy, nynew)
    fnew = np.zeros((nynew, nxnew))
    fnew[ny0:ny0 + ny, nx0:nx0 + nx] = f
    return fnew, xnew, ynew


def fftfits(fitsimage: str, center: str | None = None, lam: float = 1,
            xlim: list | None = None, ylim: list | None = None,
            plot: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FFT a fits image with the phase referring to a specific point.

    Args:
        fitsimage (str): Input fits name in the unit of Jy/pixel.
        center (str, optional): Text coordinate. Defaults to None.
        lam (float, optional): Return u * lam and v * lam. Defaults to 1.
        xlim (list, optional): Range of x for zero padding in arcsec.
        ylim (list, optional): Range of y for zero padding in arcsec.
        plot (bool, optional): Check F through images.

    Returns:
        tuple: (F, u, v). F is FFT of f in the unit of Jy. u and v are 1D arrays in the unit of lambda or meter if lam it not unity.
    """
    f, (x, y, v), _, _, _ = fits2data(fitsimage, center=center)
    if xlim is not None and ylim is not None:
        f, x, y = zeropadding(f, x, y, xlim, ylim)
    arcsec = np.radians(1) / 3600.
    F, u, v = fftcentering2(f, x * arcsec, y * arcsec)
    u, v = u * lam, v * lam
    if plot:
        set_rcparams()
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(2, 2, 1)
        m = ax.pcolormesh(u, v, np.real(F), shading='nearest', cmap='jet')
        fig.colorbar(m, ax=ax, label='Real', format='%.1e')
        ax.set_ylabel(r'v ($\lambda$)')
        ax = fig.add_subplot(2, 2, 2)
        m = ax.pcolormesh(u, v, np.imag(F), shading='nearest', cmap='jet')
        fig.colorbar(m, ax=ax, label='Imaginary', format='%.1e')
        ax = fig.add_subplot(2, 2, 3)
        m = ax.pcolormesh(u, v, np.abs(F), shading='nearest', cmap='jet')
        fig.colorbar(m, ax=ax, label='Amplitude', format='%.1e')
        ax.set_xlabel(r'u ($\lambda$)')
        ax.set_ylabel(r'v ($\lambda$)')
        ax = fig.add_subplot(2, 2, 4)
        m = ax.pcolormesh(u, v, np.angle(F) * np.degrees(1),
                          shading='nearest', cmap='jet')
        fig.colorbar(m, ax=ax, label='Phase (deg)', format='%.0f')
        ax.set_xlabel(r'u ($\lambda$)')
        fig.tight_layout()
        plt.show()
        plt.close()
    return F, u, v


def findindex(u: np.ndarray | None = None, v: np.ndarray | None = None,
              uobs: np.ndarray | None = None, vobs: np.ndarray | None = None
              ) -> np.ndarray:
    """Find indicies of the observed visibility points.

    Args:
        u (np.ndarray, optional): 1D array. The first frequency coordinate. Defaults to None.
        v (np.ndarray, optional): 1D array. The second frequency cooridnate. Defaults to None.
        uobs (np.ndarray, optional): 1D array. Observed u. Defaults to None.
        vobs (np.ndarray, optional): 1D array. Observed v. Defaults to None.

    Returns:
        np.ndarray: Indicies or an array of indicies.
    """
    if u is not None:
        Nu, du = len(u), u[1] - u[0]
    if v is not None:
        Nv, dv = len(v), v[1] - v[0]
    idx_u, idx_v = None, None
    if uobs is not None:
        idx_u = np.round(uobs / du + Nu // 2).astype(np.int64)
    if vobs is not None:
        idx_v = np.round(vobs / dv + Nv // 2).astype(np.int64)
    if idx_u is not None and idx_v is not None:
        return np.array([idx_u, idx_v])
    if idx_u is not None:
        return idx_u


def fftfitssample(fitsimage: str, center: str | None = None,
                  index_u: np.ndarray | None = None,
                  index_v: np.ndarray | None = None,
                  xlim: list | None = None, ylim: list | None = None,
                  getindex: bool = False,
                  u_sample: np.ndarray | None = None,
                  v_sample: np.ndarray | None = None) -> np.ndarray:
    """Find indicies or the visibilities on them from an image fits file.

    Args:
        fitsimage (str): Input fits name in the unit of Jy/pixel.
        center (str, optional): Text coordinate. Defaults to None.
        index_u (np.ndarray, optional): Indicies. Output from the getindex mode. Defaults to None.
        index_v (np.ndarray, optional): Indicies. Output from the getindex mode. Defaults to None.
        xlim (list, optional): Range of x for zero padding in arcsec.
        ylim (list, optional): Range of y for zero padding in arcsec.
        getindex (bool, optional): True outputs [index_u, index_v]. Defaults to False.
        u_sample (np.ndarray, optional): 1D array. Observed u. Defaults to None.
        v_sample (np.ndarray, optional): 1D array. Observed u. Defaults to None.

    Returns:
        np.ndarray: Array of indicies or sampled FFT.
    """
    F, u, v = fftfits(fitsimage=fitsimage, center=center, xlim=xlim, ylim=ylim)
    if index_u is None or index_v is None:
        index_u, index_v = findindex(u, v, u_sample, v_sample)
    if getindex:
        return np.array([index_u, index_v])
    else:
        return F[index_v, index_u]
