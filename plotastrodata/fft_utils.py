import matplotlib.pyplot as plt
import numpy as np
from typing import Callable
from os import PathLike

from plotastrodata._type_utils import _normalize_float

from plotastrodata.fits_utils import fits2data
from plotastrodata.other_utils import close_figure


def shiftphase(F: np.ndarray, u: np.ndarray,
               xoff: float = 0) -> np.ndarray:
    """Shift the phase of 1D FFT by xoff.

    Args:
        F (np.ndarray): 1D FFT.
        u (np.ndarray): 1D array. The frequency coordinate.
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
        v (np.ndarray): 1D or 2D array. The second frequency coordinate.
        xoff (float): From old to new center. Defaults to 0.
        yoff (float): From old to new center. Defaults to 0.

    Returns:
        np.ndarray: phase-shifted FFT.
    """
    (U, V) = np.meshgrid(u, v) if np.ndim(u) == 1 else (u, v)
    return F * np.exp(1j * 2 * np.pi * (U * xoff + V * yoff))


def fftcentering(f: np.ndarray, x: np.ndarray | None = None,
                 xcenter: float = 0, rfft: bool = False
                 ) -> tuple[np.ndarray, np.ndarray]:
    """FFT with the phase referring to a specific point.

    Args:
        f (np.ndarray): 1D array for FFT.
        x (np.ndarray, optional): 1D array. The spatial coordinate.
            Defaults to None.
        xcenter (float, optional): x of phase reference. Defaults to 0.
        rfft (bool, optional): True means using rFFT. Defaults to False.

    Returns:
        tuple: (F, u). F is FFT of f. u is a 1D array of the frequency
        coordinate.
    """
    nx = np.shape(f)[0]
    if x is None:
        x = np.arange(nx)
    X = x[0, :] if np.ndim(x) == 2 else x
    dx = X[1] - X[0]
    if rfft:
        u = np.fft.rfftfreq(nx, d=dx)
        F = np.fft.rfft(f)
    else:
        u = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
        F = np.fft.fftshift(np.fft.fft(f))
    F = shiftphase(F, u=u, xoff=xcenter - X[0])
    return F, u


def fftcentering2(f: np.ndarray,
                  x: np.ndarray | None = None, y: np.ndarray | None = None,
                  xcenter: float = 0, ycenter: float = 0, rfft: bool = False
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FFT with the phase referring to a specific point.

    Args:
        f (np.ndarray): 2D array for FFT.
        x (np.ndarray, optional): 1D or 2D array. The first spatial
            coordinate. Defaults to None.
        y (np.ndarray, optional): 1D or 2D array. The second spatial
            coordinate. Defaults to None.
        xcenter (float, optional): x of phase reference. Defaults to 0.
        ycenter (float, optional): y of phase reference. Defaults to 0.
        rfft (bool, optional): True means using rFFT. Defaults to False.

    Returns:
        tuple: (F, u, v). F is FFT of f. u and v are 1D arrays of the
        frequency coordinates.
    """
    ny, nx = np.shape(f)
    if x is None:
        x = np.arange(nx)
    if y is None:
        y = np.arange(ny)
    X = x[0, :] if np.ndim(x) == 2 else x
    Y = y[:, 0] if np.ndim(y) == 2 else y
    dx = X[1] - X[0]
    dy = Y[1] - Y[0]
    if rfft:
        u = np.fft.rfftfreq(nx, d=dx)
        v = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
        F = np.fft.fftshift(np.fft.rfft2(f), axes=0)
    else:
        u = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
        v = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
        F = np.fft.fftshift(np.fft.fft2(f))
    F = shiftphase2(F, u, v, xcenter - X[0], ycenter - Y[0])
    return F, u, v


def ifftcentering(F: np.ndarray, u: np.ndarray | None = None,
                  xcenter: float = 0, x0: float | None = None,
                  dx: float = 1,
                  outreal: bool = False, rfft: bool = False
                  ) -> tuple[np.ndarray, np.ndarray]:
    """inverse FFT with the phase referring to a specific point.

    Args:
        F (np.ndarray): 1D array. An FFT result in the frequency domain.
        u (np.ndarray, optional): 1D array. The frequency coordinate.
            Defaults to None.
        xcenter (float, optional): x of phase reference (used in
            fftcentering). Defaults to 0.
        x0 (float, optional): spatial coordinate of x[0]. Defaults to
            None.
        dx (float, optional): spatial interval. Defaults to 1.
        outreal (bool, optional): whether output only the real part.
            Defaults to False.
        rfft (bool, optional): True means using rFFT. Defaults to False.

    Returns:
        tuple: (f, x). f is iFFT of F. x is a 1D array of the spatial
        coordinate.
    """
    nx = np.shape(F)[0]
    if u is None:
        if rfft:
            nx = 2 * (nx - 1)  # Follow numpy.fft.irfft behavior.
            u = np.fft.rfftfreq(nx, d=dx)
        else:
            u = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    else:
        if rfft:
            if np.isclose(u[-1], 1 / (2 * dx)):
                nx = 2 * (nx - 1)
            else:
                nx = 2 * (nx - 1) + 1
    x = (np.arange(nx) - (nx-1)/2.) / (u[1]-u[0]) / nx + xcenter
    if x0 is not None:
        x = x - x[0] + x0
    F = shiftphase(F, u=u, xoff=x[0] - xcenter)
    if rfft:
        f = np.fft.irfft(F, n=nx)
    else:
        f = np.fft.ifft(np.fft.ifftshift(F))
    if outreal:
        f = np.real(f)
    return f, x


def ifftcentering2(F: np.ndarray,
                   u: np.ndarray | None = None, v: np.ndarray | None = None,
                   xcenter: float = 0, ycenter: float = 0,
                   x0: float | None = None, y0: float | None = None,
                   dx: float = 1, dy: float = 1,
                   outreal: bool = False, rfft: bool = False
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """inverse FFT with the phase referring to a specific point.

    Args:
        F (np.ndarray): 2D array. An FFT result in the frequency domain.
        u (np.ndarray, optional): 1D or 2D array. The first frequency
            coordinate. Defaults to None.
        v (np.ndarray, optional): 1D or 2D array. The second frequency
            coordinate. Defaults to None.
        xcenter (float, optional): x of phase reference (used in
            fftcentering2). Defaults to 0.
        ycenter (float, optional): y of phase reference (used in
            fftcentering2). Defaults to 0.
        x0 (float, optional): spatial coordinate of x[0]. Defaults to
            None.
        y0 (float, optional): spatial coordinate of y[0]. Defaults to
            None.
        dx (float, optional): spatial interval. Defaults to 1.
        dy (float, optional): spatial interval. Defaults to 1.
        outreal (bool, optional): whether output only the real part.
            Defaults to False.
        rfft (bool, optional): True means using rFFT. Defaults to False.

    Returns:
        tuple: (f, x, y). f is iFFT of F. x and y are 1D arrays of the
        spatial coordinates.
    """
    ny, nx = np.shape(F)
    if u is not None and np.ndim(u) == 2:
        u = u[0, :]
    if v is not None and np.ndim(v) == 2:
        v = v[:, 0]
    if u is None:
        if rfft:
            nx = 2 * (nx - 1)  # Follow numpy.fft.irfft behavior.
            u = np.fft.rfftfreq(nx, d=dx)
        else:
            u = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    else:
        if rfft:
            if np.isclose(u[-1], 1 / (2 * dx)):
                nx = 2 * (nx - 1)
            else:
                nx = 2 * (nx - 1) + 1
    if v is None:
        v = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    x = (np.arange(nx) - (nx-1)/2.) / (u[1]-u[0]) / nx + xcenter
    y = (np.arange(ny) - (ny-1)/2.) / (v[1]-v[0]) / ny + ycenter
    if x0 is not None:
        x = x - x[0] + x0
    if y0 is not None:
        y = y - y[0] + y0
    F = shiftphase2(F, u, v, x[0] - xcenter, y[0] - ycenter)
    if rfft:
        f = np.fft.irfft2(np.fft.ifftshift(F, axes=0), s=(ny, nx))
    else:
        f = np.fft.ifft2(np.fft.ifftshift(F))
    if outreal:
        f = np.real(f)
    return f, x, y


class FftCentering():
    """Set FFT conditions and functions.

    Args:
        x (np.ndarray): 1D array.
        y (np.ndarray, optional): 1D array. Defaults to None.
        xcenter (float, optional): x of phase reference. Defaults to 0.
        ycenter (float, optional): y of phase reference. Defaults to 0.
        rfft (bool, optional): True means using rFFT. Defaults to False.
    """
    def __init__(self, x: np.ndarray,
                 y: np.ndarray | None = None,
                 xcenter: float = 0,
                 ycenter: float = 0,
                 rfft: bool = False) -> None:
        nx = len(x)
        self.x = x
        self.dx = dx = _normalize_float(x[1] - x[0])
        self.xcenter = xcenter
        self.ndim = 2 if isinstance(y, np.ndarray) else 1
        self.rfft = rfft
        if rfft:
            u = np.fft.rfftfreq(nx, d=dx)
        else:
            u = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
        self.u = u
        if self.ndim == 2:
            ny = len(y)
            self.y = y
            self.dy = dy = _normalize_float(y[1] - y[0])
            self.ycenter = ycenter
            v = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
            self.v = v

    def fft(self, f: np.ndarray | None = None
            ) -> np.ndarray | Callable[[np.ndarray], np.ndarray]:
        """FFT calculation done by considering 1D/2D and fft/rfft.

        Args:
            f (np.ndarray | None, optional): 1D or 2D array. Defaults to
                None.

        Returns:
            np.ndarray or Callable: FFT result. When f is None, the return is the
            FFT function.
        """
        if self.ndim == 1:
            def func(f: np.ndarray) -> np.ndarray:
                F, _ = fftcentering(f=f, x=self.x,
                                    xcenter=self.xcenter,
                                    rfft=self.rfft)
                return F

        else:
            def func(f: np.ndarray) -> np.ndarray:
                F, _, _ = fftcentering2(f=f, x=self.x, y=self.y,
                                        xcenter=self.xcenter,
                                        ycenter=self.ycenter,
                                        rfft=self.rfft)
                return F

        return func if f is None else func(f)

    def ifft(self, F: np.ndarray | None = None, outreal: bool = False
             ) -> np.ndarray | Callable[[np.ndarray], np.ndarray]:
        """iFFT calculation done by considering 1D/2D and fft/rfft.

        Args:
            F (np.ndarray | None, optional): An FFT result. Defaults to
                None.

        Returns:
            np.ndarray or Callable: iFFT result. When F is None, the return is the
            iFFT function.
        """
        if self.ndim == 1:
            def func(F: np.ndarray) -> np.ndarray:
                f, _ = ifftcentering(F=F, u=self.u,
                                     xcenter=self.xcenter,
                                     x0=self.x[0],
                                     dx=self.dx,
                                     outreal=outreal,
                                     rfft=self.rfft)
                return f

        else:
            def func(F: np.ndarray) -> np.ndarray:
                f, _, _ = ifftcentering2(F=F, u=self.u, v=self.v,
                                         xcenter=self.xcenter,
                                         ycenter=self.ycenter,
                                         x0=self.x[0], y0=self.y[0],
                                         dx=self.dx, dy=self.dy,
                                         outreal=outreal,
                                         rfft=self.rfft)
                return f

        return func if F is None else func(F)


def zeropadding(f: np.ndarray, x: np.ndarray, y: np.ndarray,
                xlim: list[float] | tuple[float, float] | np.ndarray,
                ylim: list[float] | tuple[float, float] | np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad an outer region with zero.

    Args:
        f (np.ndarray): Input 2D array.
        x (np.ndarray): 1D array.
        y (np.ndarray): 1D array.
        xlim (list or tuple or np.ndarray): range of x after the zero padding.
        ylim (list or tuple or np.ndarray): range of y after the zero padding.

    Returns:
        tuple: (fnew, xnew, ynew). fnew is an 2D array and xnew and ynew
        are 1D arrays after the zero padding.
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
    fnew = np.zeros((nynew, nxnew), dtype=f.dtype)
    fnew[ny0:ny0 + ny, nx0:nx0 + nx] = f
    return fnew, xnew, ynew


def fftfits(fitsimage: str | PathLike[str],
            center: str | None = None, lam: float = 1,
            xlim: list[float] | tuple[float, float] | np.ndarray | None = None,
            ylim: list[float] | tuple[float, float] | np.ndarray | None = None,
            savefig: dict | str | None = None, show: bool = False,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FFT a fits image with the phase referring to a specific point.

    Args:
        fitsimage (str or os.PathLike): Input fits name in the unit of Jy/pixel.
        center (str, optional): Text coordinate. Defaults to None.
        lam (float, optional): Return u * lam and v * lam. Defaults to
            1.
        xlim (list or tuple or np.ndarray, optional): Range of x for zero padding in arcsec.
        ylim (list or tuple or np.ndarray, optional): Range of y for zero padding in arcsec.
        savefig (dict or str, optional): Passed to ``close_figure``.
            Existing files may be overwritten, and the figure is closed
            after saving/showing. Defaults to None.
        show (bool, optional): True means doing plt.show(). Defaults to
            False.

    Returns:
        tuple: (F, u, v). F is FFT of f in the unit of Jy. u and v are
        1D arrays in the unit of lambda or meter if lam it not unity.
    """
    f, (x, y, v), _, _, _ = fits2data(fitsimage, center=center)
    if xlim is not None and ylim is not None:
        f, x, y = zeropadding(f, x, y, xlim, ylim)
    f[np.isnan(f)] = 0
    arcsec = np.radians(1) / 3600.
    F, u, v = fftcentering2(f, x * arcsec, y * arcsec)
    u, v = u * lam, v * lam
    if savefig is not None or show:
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))
        m = ax[0, 0].pcolormesh(u, v, np.real(F), cmap='jet')
        fig.colorbar(m, ax=ax[0, 0], label='Real', format='%.1e')
        ax[0, 0].set_ylabel(r'v ($\lambda$)')
        m = ax[0, 1].pcolormesh(u, v, np.imag(F), cmap='jet')
        fig.colorbar(m, ax=ax[0, 1], label='Imaginary', format='%.1e')
        m = ax[1, 0].pcolormesh(u, v, np.abs(F), cmap='jet')
        fig.colorbar(m, ax=ax[1, 0], label='Amplitude', format='%.1e')
        ax[1, 0].set_xlabel(r'u ($\lambda$)')
        ax[1, 0].set_ylabel(r'v ($\lambda$)')
        m = ax[1, 1].pcolormesh(u, v, np.angle(F) * np.degrees(1),
                                cmap='jet')
        fig.colorbar(m, ax=ax[1, 1], label='Phase (deg)', format='%.0f')
        ax[1, 1].set_xlabel(r'u ($\lambda$)')
        close_figure(fig, savefig, show)
    return F, u, v


def findindex(u: np.ndarray | None = None, v: np.ndarray | None = None,
              uobs: float | np.floating | np.ndarray | None = None,
              vobs: float | np.floating | np.ndarray | None = None
              ) -> int | np.ndarray | None:
    """Return nearest grid indices for scalar or array observations.

    Each supplied observation requires its corresponding 1D grid with at
    least two points. One supplied axis returns its indices; two return an
    array [index_u, index_v]. No observations returns None. Scalar indices
    are Python integers; array indices retain their array form.
    """
    def indices(grid: np.ndarray | None,
                observed: float | np.floating | np.ndarray | None,
                name: str) -> int | np.ndarray | None:
        if observed is None:
            return None
        if grid is None or np.ndim(grid) != 1 or len(grid) < 2:
            raise ValueError(f'{name} observations require a 1D {name} grid '
                             + 'with at least two points.')
        step = grid[1] - grid[0]
        if not np.isfinite(step) or step == 0:
            raise ValueError(f'{name} grid spacing must be finite and nonzero.')
        idx = np.round(observed / step + len(grid) // 2).astype(np.int64)
        if isinstance(observed, np.ndarray):
            return np.asarray(idx)
        return int(idx)

    idx_u = indices(u, uobs, 'u')
    idx_v = indices(v, vobs, 'v')
    if idx_u is not None and idx_v is not None:
        return np.array([idx_u, idx_v])
    return idx_u if idx_u is not None else idx_v


def fftfitssample(fitsimage: str | PathLike[str], center: str | None = None,
                  index_u: int | np.integer | np.ndarray | None = None,
                  index_v: int | np.integer | np.ndarray | None = None,
                  xlim: list[float] | tuple[float, float] | np.ndarray | None = None,
                  ylim: list[float] | tuple[float, float] | np.ndarray | None = None,
                  getindex: bool = False,
                  u_sample: float | np.floating | np.ndarray | None = None,
                  v_sample: float | np.floating | np.ndarray | None = None
                  ) -> np.ndarray | complex | np.clongdouble:
    """Find indices or the visibilities on them from an image FITS file.

    Args:
        fitsimage (str or os.PathLike): Input fits name in the unit of Jy/pixel.
        center (str, optional): Text coordinate. Defaults to None.
        index_u (int or np.integer or np.ndarray, optional): Indices. Output from the
            getindex mode. Defaults to None.
        index_v (int or np.integer or np.ndarray, optional): Indices. Output from the
            getindex mode. Defaults to None.
        xlim (list or tuple or np.ndarray, optional): Range of x for zero padding in arcsec.
        ylim (list or tuple or np.ndarray, optional): Range of y for zero padding in arcsec.
        getindex (bool, optional): True outputs [index_u, index_v].
            Defaults to False.
        u_sample (float or np.floating or np.ndarray, optional): 1D array. Observed u. Defaults
            to None.
        v_sample (float or np.floating or np.ndarray, optional): 1D array. Observed u. Defaults
            to None.

    Returns:
        np.ndarray or complex or np.clongdouble: Indices or sampled FFT.
        Ordinary scalar samples are Python complex values; extended
        precision and array results are preserved.

    Raises:
        ValueError: If either axis has neither indices nor sample coordinates.
    """
    if index_u is None and u_sample is None:
        raise ValueError('Provide index_u or u_sample for FFT sampling.')
    if index_v is None and v_sample is None:
        raise ValueError('Provide index_v or v_sample for FFT sampling.')
    F, u, v = fftfits(fitsimage=fitsimage, center=center, xlim=xlim, ylim=ylim)
    if index_u is None:
        index_u = findindex(u=u, uobs=u_sample)
    if index_v is None:
        index_v = findindex(v=v, vobs=v_sample)
    if getindex:
        return np.array([index_u, index_v])
    else:
        result = F[index_v, index_u]
        if isinstance(result, np.complexfloating) and not isinstance(result, np.clongdouble):
            return complex(result)
        return result
