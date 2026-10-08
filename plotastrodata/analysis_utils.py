import numpy as np
import warnings
import re
from dataclasses import dataclass, field
from os import PathLike
from functools import wraps
import numbers
from astropy.io.fits import Header
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.signal import convolve
from pydantic import ConfigDict, WrapValidator
from pydantic.dataclasses import dataclass as pydantic_dataclass
from typing import (Annotated, Any, Callable, Literal,
                    TypeVar, ParamSpec, TypedDict)

from plotastrodata import const_utils as cu
from plotastrodata._type_utils import _normalize_float
from plotastrodata.coord_utils import coord2xy, rel2abs, xy2coord, _getframe
from plotastrodata.fits_utils import data2fits, FitsData, Jy2K
from plotastrodata.fitting_utils import (EmceeCorner, gaussfit1d,
                                         gaussfit2d, gaussian2d)
from plotastrodata.matrix_utils import dot2d, Mfac, Mrot
from plotastrodata.noise_utils import estimate_rms
from plotastrodata.other_utils import (isdeg, nearest_index,
                                       RGIxy, RGIxyv, to4dim, trim)


def quadrantmean(data: np.ndarray, x: np.ndarray, y: np.ndarray,
                 quadrants: Literal['13', '24'] = '13'
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Take mean between 1st and 3rd (or 2nd and 4th) quadrants.

    Args:
        data (np.ndarray): 2D array.
        x (np.ndarray): 1D array. First coordinate.
        y (np.ndarray): 1D array. Second coordinate.
        quadrants (str, optional): '13' or '24'. Defaults to '13'.

    Returns:
        tuple: Averaged (data, x, y).
    """
    if np.ndim(data) != 2:
        raise ValueError('data must be a 2D array.')

    if quadrants not in ['13', '24']:
        raise ValueError("quadrants must be '13' or '24'.")

    if len(x) < 2 or len(y) < 2:
        raise ValueError('quadrantmean requires at least two pixels per axis.')
    if x[1] < x[0]:
        x, data = x[::-1], data[:, ::-1]
    if y[1] < y[0]:
        y, data = y[::-1], data[::-1, :]
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    nx = int(np.ceil(np.max(np.abs(x)) / dx))
    ny = int(np.ceil(np.max(np.abs(y)) / dy))
    xnew = np.linspace(-nx, nx, 2 * nx + 1) * dx
    ynew = np.linspace(-ny, ny, 2 * ny + 1) * dy
    s = 1 if quadrants == '13' else -1
    f = RGI((y, s * x), data, bounds_error=False, fill_value=np.nan)
    datanew = f(np.meshgrid(ynew, xnew, indexing='ij'))
    datanew = (datanew + datanew[::-1, ::-1]) / 2.
    return datanew[ny:, nx:], xnew[nx:], ynew[ny:]


def filled2d(data: np.ndarray, x: np.ndarray, y: np.ndarray,
             n: int | np.integer = 1,
             **kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill 2D data, 1D x, and 1D y by a factor of n using RGI.

    Args:
        data (np.ndarray): 2D or 3D array.
        x (np.ndarray): 1D array.
        y (np.ndarray): 1D array.
        n (int or np.integer, optional): How many times more the new grid is.
            Defaults to 1.

    Returns:
        tuple: The interpolated (data, x, y).
    """
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError('n must be a positive integer.')
    xnew = np.linspace(x[0], x[-1], n * (len(x) - 1) + 1)
    ynew = np.linspace(y[0], y[-1], n * (len(y) - 1) + 1)
    d = RGIxy(y, x, data, np.meshgrid(ynew, xnew, indexing='ij'),
              **kwargs)
    return d, xnew, ynew


_MethodArgs = ParamSpec('_MethodArgs')
_MethodResult = TypeVar('_MethodResult')


def _need_multipixels(method: Callable[_MethodArgs, _MethodResult]
                      ) -> Callable[_MethodArgs, _MethodResult]:
    @wraps(method)
    def wrapper(*args: _MethodArgs.args, **kwargs: _MethodArgs.kwargs
                ) -> _MethodResult:
        cls = args[0] if args else kwargs['self']
        singlepixel = cls.dx is None or cls.dy is None
        if singlepixel:
            raise ValueError(f'{method.__name__}() requires at least'
                             + ' two x and y pixels.')
        return method(*args, **kwargs)
    return wrapper


def _is_data_list(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    c0 = any(a is not None for a in value)
    c1 = all(a is None or isinstance(a, np.ndarray) for a in value)
    return c0 and c1


# Field aliases also describe the per-dataset forms used internally by plotting.
_T = TypeVar('_T')
_PerDataset = _T | list[_T]
_RealScalar = float | np.floating | np.integer
_FitsPath = str | PathLike[str]
_NoiseSpec = str | float | numbers.Number | None
_Beam = (np.ndarray | list[_RealScalar | None]
         | tuple[_RealScalar | None, _RealScalar | None, _RealScalar | None])


class _Fit2DResult(TypedDict):
    popt: np.ndarray
    plow: np.ndarray
    pmid: np.ndarray
    phigh: np.ndarray
    model: np.ndarray
    residual: np.ndarray


class _GaussFit2DResult(TypedDict):
    popt: np.ndarray
    perr: np.ndarray
    model: np.ndarray
    residual: np.ndarray
    center: str | None


@dataclass
class AstroData():
    """Data to be processed and parameters for processing the data.

    Args:
        data (np.ndarray or array-like, optional): A single dataset.
            Nested numeric lists or tuples are converted to an array.
            Defaults to None.
        x (np.ndarray, optional): 1D array. Defaults to None.
        y (np.ndarray, optional): 1D array. Defaults to None.
        v (np.ndarray, optional): 1D array. Defaults to None.
        beam (np.ndarray, list, or tuple, optional): [bmaj, bmin, bpa].
            Defaults to [None, None, None].
        fitsimage (str or os.PathLike, optional): Input fits name.
            Defaults to None.
        Tb (bool, optional): True means the data array is brightness
            temperature. Defaults to False.
        sigma (float or str, optional): Noise level or method for
            measuring it. Defaults to 'hist'.
        center (str, optional): Text coordinates. 'common' means
            initialized value. Defaults to 'common'.
        restfreq (float, optional): Used for velocity and brightness
            temperature. Defaults to None.
        cfactor (float, optional): The data array is multiplied by
            cfactor. Defaults to 1.
        pvpa (float, optional): Position angle of the PV cut. Defaults
            to None.
        pv (bool, optional): True means the data array is a
            position-velocity diagram. Defaults to False.
        bunit (str, optional): The unit of the data array. Defaults to
            ''.
        dist (float or np.longdouble, optional): Distance factor used to
            scale spatial coordinates, recorded by AstroFrame.read. Defaults
            to 1.0. Changing this value alone does not rescale coordinates
            or beams. Extended-precision NumPy floats are preserved.

    Note:
        External use normally supplies one dataset. PlotAstroData also uses
        lists of arrays (with optional None placeholders) or FITS paths and
        per-dataset metadata lists internally. Initialization determines n;
        AstroFrame.read performs metadata expansion and loading later.
    """
    data: np.ndarray | list[Any] | tuple[Any, ...] | None = None
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    v: np.ndarray | None = None
    beam: _PerDataset[_Beam] = (None, None, None)
    fitsimage: _PerDataset[_FitsPath | None] = None
    Tb: _PerDataset[bool] = False
    sigma: _PerDataset[_NoiseSpec] = 'hist'
    center: _PerDataset[str | None] = 'common'
    restfreq: _PerDataset[_RealScalar | None] = None
    cfactor: _PerDataset[_RealScalar] = 1
    pvpa: _PerDataset[_RealScalar | None] = None
    pv: _PerDataset[bool] = False
    bunit: _PerDataset[str | None] = ''
    dist: float | np.longdouble = 1.0
    dx: int | float | np.longdouble | None = field(
        init=False, default=None, repr=False, compare=False)
    dy: int | float | np.longdouble | None = field(
        init=False, default=None, repr=False, compare=False)
    dv: int | float | np.longdouble | None = field(
        init=False, default=None, repr=False, compare=False)
    n: int = field(init=False, repr=False, compare=False)
    fitsimage_org: _PerDataset[_FitsPath | None] = field(
        init=False, default=None, repr=False, compare=False)
    sigma_org: _PerDataset[_NoiseSpec] = field(
        init=False, default=None, repr=False, compare=False)
    beam_org: _PerDataset[_Beam | None] = field(
        init=False, default=None, repr=False, compare=False)
    fitsheader: _PerDataset[Header | dict[str, Any] | None] = field(
        init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        fits_values = (self.fitsimage if isinstance(self.fitsimage, list)
                       else [self.fitsimage])
        has_fits = any(a is not None for a in fits_values)
        if isinstance(self.data, list):
            has_data = any(a is not None for a in self.data)
        else:
            has_data = self.data is not None
        if has_fits and has_data:
            raise ValueError('Provide either data or fitsimage, not both.')
        if not has_fits and not has_data:
            raise ValueError('Either data or fitsimage must be given.')

        n = 0
        if has_fits:
            if not isinstance(self.fitsimage, list):
                n = 1
            else:
                n = len(self.fitsimage)
            self.data = None
        if has_data:
            if _is_data_list(self.data):
                n = len(self.data)
            elif (isinstance(self.data, list)
                  and not any(a is not None for a in self.data)):
                n = 0
            else:
                self.data = np.asarray(self.data)
                n = 1
            datasets = self.data if _is_data_list(self.data) else [self.data]
            for i, data in enumerate(datasets):
                if data is not None and np.ndim(data) == 0:
                    raise ValueError(f'Dataset {i} must be an array '
                                     + 'with spatial axes, not a scalar'
                                     + ' or zero-dimensional array.')
        self.n = n
        self.fitsimage_org = None
        self.sigma_org = None
        self.beam_org = None
        self.fitsheader = None

    def _binning_one(self, t: str, width: int | np.integer) -> None:
        width = int(width)
        grid = getattr(self, t)
        if width == 1:
            return
        if grid is None:
            raise ValueError(f'Binning in the {t}-axis requires'
                             + f' a {t} coordinate array.')

        dt = f'd{t}'
        sep = getattr(self, dt)
        if sep is None:
            s = f'Skip binning in the {t}-axis because {dt} is None.'
            warnings.warn(s, UserWarning)
            return

        i = {'v': 1, 'y': 2, 'x': 3}[t]
        sizenew = self.size[i] // width
        self.size[i] = sizenew
        data = np.moveaxis(self.data, i, 0)
        dtype = (self.data.dtype
                 if self.data.dtype.kind in "fc"
                 else np.dtype(float))
        datanew = np.moveaxis(np.zeros(self.size, dtype=dtype), i, 0)
        grid_dtype = grid.dtype if grid.dtype.kind in "fc" else np.dtype(float)
        gridnew = np.zeros(sizenew, dtype=grid_dtype)
        for start in range(width):
            stop = start + sizenew * width
            datanew += data[start:stop:width]
            gridnew += grid[start:stop:width]
        self.data = np.moveaxis(datanew, 0, i) / width
        setattr(self, t, gridnew / width)
        spacing = sep * int(width)
        setattr(self, dt, int(spacing) if isinstance(spacing, np.integer)
                else _normalize_float(spacing))

    def binning(self, width: list[int | np.integer]
                | tuple[int | np.integer, ...] | np.ndarray = [1, 1, 1]
                ) -> None:
        """Binning up neighboring pixels in the v, y, and x domain.

        Args:
            width (list, tuple, or np.ndarray, optional): Number of channels, 
                y-pixels, and x-pixels for binning. Defaults to [1, 1, 1].
        """
        if not 1 <= len(width) <= 3 or any(not isinstance(a, (int, np.integer))
                                           or a < 1 for a in width):
            raise ValueError('width must contain one to three positive integers.')
        w = np.array([1] * (3 - len(width)) + list(width), dtype=int)
        if self.pv:
            w[1] = max(w[0], w[1])
            w[2] = 1
        data4d = to4dim(self.data)
        size = np.array(np.shape(data4d))
        if np.any(w > size[1:]):
            w = np.minimum(w, size[1:])
            ws = ', '.join([f'{s:d}' for s in w])
            print(f'width was changed to [{ws}].')
        for axis, factor in zip(('v', 'y', 'x'), w):
            if factor > 1 and getattr(self, axis) is None:
                raise ValueError(f'Binning in the {axis}-axis requires'
                                 + ' a coordinate array.')
        velocity_binning = (not self.pv and w[0] > 1) or (self.pv and w[1] > 1)
        if velocity_binning and isinstance(self.sigma, str):
            raise ValueError('Estimate sigma before velocity binning,'
                             + ' or set sigma=None.')
        self.data = data4d
        if velocity_binning:
            width_v = w[1] if self.pv else w[0]
            if self.sigma is not None:
                print(f'sigma has been divided by sqrt({width_v:d})'
                      + ' because of binning in the v-axis.')
                self.sigma = _normalize_float(self.sigma / np.sqrt(width_v))
        if (not self.pv and w[1] > 1) or w[2] > 1:
            print('Binning in the x- or y-axis does not update sigma.')
        self.size = size
        for t, ww in zip(['v', 'y', 'x'], w):
            self._binning_one(t, ww)
        self.data = np.squeeze(self.data)
        del self.size
        if self.pv:
            self.v = self.y
            self.dv = self.dy

    def centering(self, includexy: bool = True,
                  includev: bool = False,
                  **kwargs: Any) -> None:
        """Spatial regridding to set the center at (x,y,v)=(0,0,0).

        Args:
            includexy (bool, optional): Centering in the x and y
                directions at each channel. Defaults to True.
            includev (bool, optional): Centering in the v direction at
                each position. Defaults to False.
        """
        if includexy:
            xnew = self.x - self.x[nearest_index(self.x)]
            ynew = self.y - self.y[nearest_index(self.y)]
        if includev:
            vnew = self.v - self.v[nearest_index(self.v)]
        if includexy and includev:
            self.data = RGIxyv(self.v, self.y, self.x, self.data,
                               np.meshgrid(vnew, ynew, xnew, indexing='ij'),
                               **kwargs)
            self.v, self.y, self.x = vnew, ynew, xnew
        elif includexy:
            self.data = RGIxy(self.y, self.x, self.data,
                              np.meshgrid(ynew, xnew, indexing='ij'),
                              **kwargs)
            self.y, self.x = ynew, xnew
        elif includev:
            nx, ny, nv = len(self.x), len(self.y), len(self.v)
            dtype = (self.data.dtype if self.data.dtype.kind in "fc"
                     else np.dtype(float))
            a = np.empty((ny, nx, nv), dtype=dtype)
            for i in range(ny):
                for j in range(nx):
                    f = RGI((self.v,), self.data[:, i, j], method='linear',
                            bounds_error=False, fill_value=np.nan)
                    a[i, j] = f(vnew)
            self.data = np.moveaxis(a, -1, 0)
            self.v = vnew
        else:
            print('No change because includexy=False and includev=False.')

    @_need_multipixels
    def circularbeam(self) -> None:
        """Make the beam circular by convolving with 1D Gaussian
        """
        if None in self.beam:
            raise ValueError('circularbeam() requires a complete beam.')

        bmaj, bmin, bpa = self.beam
        if not np.all(np.isfinite([bmaj, bmin, bpa])) or not 0 < bmin <= bmaj:
            raise ValueError('circularbeam requires finite beam values'
                             + ' with major >= minor > 0.')
        if bmaj == bmin:
            return
        self.rotate(-bpa)
        nx = len(self.x) if len(self.x) % 2 == 1 else len(self.x) - 1
        ny = len(self.y) if len(self.y) % 2 == 1 else len(self.y) - 1
        y = np.linspace(-(ny-1) / 2, (ny-1) / 2, ny) * np.abs(self.dy)
        g1 = np.exp(-4 * np.log(2) * y**2 / (bmaj**2 - bmin**2))
        e = np.sqrt(1 - bmin**2 / bmaj**2)
        g1 /= np.sqrt(np.pi / 4 / np.log(2) * bmin * e)
        g = np.zeros((ny, nx))
        g[:, (nx - 1) // 2] = g1
        d = self.data.copy()
        d[np.isnan(d)] = 0
        self.data = np.squeeze(convolve(to4dim(d), [[g]], mode='same'))
        self.rotate(bpa)
        self.beam[1] = self.beam[0]
        self.beam[2] = 0

    def deproject(self, pa: float | np.floating = 0,
                  incl: float | np.floating = 0,
                  **kwargs: Any) -> None:
        """Exapnd by a factor of 1/cos(incl) in the direction of pa+90
        deg.

        Args:
            pa (float, optional): Position angle in the unit of degree.
                Defaults to 0.
            incl (float, optional): Inclination angle in the unit of
                degree. Defaults to 0.
        """
        ci = np.cos(np.radians(incl))
        A = np.linalg.multi_dot([Mrot(pa), Mfac(1, ci), Mrot(-pa)])
        yxnew = dot2d(A, np.meshgrid(self.y, self.x, indexing='ij'))
        self.data = RGIxy(self.y, self.x, self.data, yxnew, **kwargs)
        if None not in self.beam:
            bmaj, bmin, bpa = self.beam
            a, b = np.linalg.multi_dot([Mfac(1 / bmaj, 1 / bmin),
                                        Mrot(pa - bpa),
                                        Mfac(1, ci),
                                        Mrot(-pa)]).T
            alpha = (np.dot(a, a) + np.dot(b, b)) / 2
            beta = np.dot(a, b)
            gamma = (np.dot(a, a) - np.dot(b, b)) / 2
            bpa_new = np.arctan(beta / gamma) / 2 * np.degrees(1)
            if beta * bpa_new >= 0:
                bpa_new += 90
            Det = np.sqrt(beta**2 + gamma**2)
            bmaj_new = 1 / np.sqrt(alpha - Det)
            bmin_new = 1 / np.sqrt(alpha + Det)
            self.beam = np.array([bmaj_new, bmin_new, bpa_new])

    def _fit_image(self, chan: int | np.integer | None) -> np.ndarray:
        if self.data.ndim == 2:
            if chan is not None:
                raise ValueError('chan must be None when fitting a 2D image.')
            return self.data
        if self.data.ndim != 3:
            raise ValueError('2D fitting requires an image or a cube.')
        if (not isinstance(chan, (int, np.integer))
            or isinstance(chan, (bool, np.bool_))):
            raise ValueError('A cube requires an integer chan for 2D fitting.')
        if not -len(self.data) <= chan < len(self.data):
            raise ValueError('chan is outside the cube channel range.')
        return self.data[int(chan)]

    def _fit_pixelperbeam(self) -> float | np.longdouble:
        if self.beam[0] is None or self.beam[1] is None:
            return 1.0
        sizes = np.asarray(self.beam[:2])
        if not np.all(np.isfinite(sizes)) or np.any(sizes <= 0):
            raise ValueError('Fitting requires finite, positive beam sizes.')
        area = np.abs(self.dx * self.dy)
        if not np.isfinite(area) or area <= 0:
            raise ValueError('Fitting requires finite, nonzero pixel sizes.')
        factor = _normalize_float(np.pi * sizes[0] * sizes[1]
                                  / (4 * np.log(2) * area))
        s = 'In the fitting, sigma is multiplied by sqrt(pixel-per-beam)' \
            + ' to account for beam noise correlation.'
        warnings.warn(s, UserWarning)
        return factor

    @_need_multipixels
    def fit2d(self,
              model: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
              bounds: np.ndarray | list[list[float]],
              progressbar: bool = False,
              kwargs_fit: dict[str, Any] | None = None,
              kwargs_plotcorner: dict[str, Any] | None = None,
              chan: int | np.integer | None = None) -> _Fit2DResult:
        """Fit a given 2D model function to self.data.

        Requires finite, positive scalar sigma. Missing beam sizes disable
        beam correction. Likelihood scalars preserve extended precision.

        Default keyword values:
            kwargs_plotcorner: ``show=False`` and ``savefig=None``.
            User-supplied values in ``kwargs_plotcorner`` override these
            defaults.

        Args:
            model (function): The model function in the form of f(par,
                x, y). It must return an ndarray matching the selected image.
            bounds (np.ndarray or list): bounds for fitting_utils.EmceeCorner.
            progressbar (bool, optional): progressbar for
                fitting_utils.EmceeCorner. Defaults to False.
            kwargs_fit (dict, optional): Arguments for
                fitting_utils.EmceeCorner.fit.
            kwargs_plotcorner (dict, optional): Arguments for
                fitting_utils.EmceeCorner.plotcorner.
            chan (int, optional): The channel number where the 2D model
                is fitted. Required for cubes; must be None for images.
                NumPy integer indices are also accepted.

        Returns:
            dict: The parameter sets (popt, plow, pmid, and phigh), the
            best 2D model array (model), and the residual from the model
            (residual).
        """
        d = self._fit_image(chan)
        if (not isinstance(self.sigma, (numbers.Real, np.floating, np.integer))
                or not np.isfinite(self.sigma) or self.sigma <= 0):
            raise ValueError('fit2d requires finite, positive scalar sigma.')
        x, y = np.meshgrid(self.x, self.y)
        if d.shape != x.shape:
            raise ValueError('Selected image shape must match the x/y grids.')
        pixelperbeam = self._fit_pixelperbeam()

        def evaluate(p: np.ndarray) -> np.ndarray:
            result = model(p, x, y)
            if not isinstance(result, np.ndarray) or result.shape != d.shape:
                raise ValueError('model must return an ndarray matching'
                                 + ' the selected image shape.')
            return result

        def logl(p: np.ndarray) -> float | np.longdouble:
            rss = np.nansum((evaluate(p) - d)**2)
            return _normalize_float(-0.5 * rss / self.sigma**2 / pixelperbeam)

        mcmc = EmceeCorner(bounds=bounds, logl=logl,
                           progressbar=progressbar)
        kwargs_fit0 = {}
        kwargs_fit0.update(kwargs_fit or {})
        mcmc.fit(**kwargs_fit0)
        kwargs_plotcorner0 = {'show': False, 'savefig': None}
        kwargs_plotcorner0.update(kwargs_plotcorner or {})
        kw_pl = kwargs_plotcorner0
        if kw_pl['show'] or kw_pl['savefig'] is not None:
            mcmc.plotcorner(**kw_pl)
        popt = mcmc.popt
        plow = mcmc.plow
        pmid = mcmc.pmid
        phigh = mcmc.phigh
        modelopt = evaluate(popt)
        residual = d - modelopt
        return {'popt': popt, 'plow': plow, 'pmid': pmid, 'phigh': phigh,
                'model': modelopt, 'residual': residual}

    @_need_multipixels
    def gaussfit2d(self, chan: int | np.integer | None = None
                   ) -> _GaussFit2DResult:
        """Fit a 2D Gaussian function to self.data using
        fitting_utils.gaussfit2d(). With sigma=None, noise is estimated
        from an initial fit. Missing beam sizes disable beam correction.

        Args:
            chan (int): The channel number where the 2D Gaussian is
                fitted. Required for cubes; must be None for images.
                NumPy integer indices are also accepted.

        Returns:
            dict: The best parameter set (popt), the error set (perr),
            the best 2D Gaussian array (model), the residual from the
            model (residual), and the coordinates of the best-fit center
            (center).
        """
        z = self._fit_image(chan)
        if z.shape != (len(self.y), len(self.x)):
            raise ValueError('Selected image shape must match the x/y grids.')
        if self.sigma is not None and (
                not isinstance(self.sigma, (numbers.Real,
                                            np.floating,
                                            np.integer))
                or not np.isfinite(self.sigma) or self.sigma <= 0):
            raise ValueError('gaussfit2d requires finite, positive scalar'
                             + ' sigma or None.')
        # Automatic estimation is handled by the underlying fitter; there is
        # no supplied sigma to scale in that case.
        sigma = None
        if self.sigma is not None:
            sigma = _normalize_float(self.sigma * np.sqrt(self._fit_pixelperbeam()))
        res = gaussfit2d(xdata=self.x, ydata=self.y, zdata=z,
                         sigma=sigma, show=False, nwalkersperdim=4)
        popt, perr = res['popt'], res['perr']
        model = gaussian2d(np.meshgrid(self.x, self.y), *popt)
        residual = z - model
        if self.center is not None:
            newcenter = xy2coord(popt[1:3] / 3600, coordorg=self.center)
        else:
            newcenter = None
        return {'popt': popt, 'perr': perr,
                'model': model, 'residual': residual,
                'center': newcenter}

    def histogram(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        """Output histogram of self.data using numpy.histogram. This
        method can take the arguments of numpy.histogram.

        Returns:
            tuple: (bin centers, histogram)
        """
        valid = ~np.isnan(self.data)
        if kwargs.get('weights') is not None:
            weights = np.asarray(kwargs['weights'])
            if weights.shape != self.data.shape:
                raise ValueError('weights must have the same shape as data.')
            kwargs['weights'] = weights[valid]
        hist, hbin = np.histogram(self.data[valid], **kwargs)
        hbin = (hbin[:-1] + hbin[1:]) / 2
        return hbin, hist

    def mask(self, dataformask: np.ndarray | None = None,
             includepix: list[float] | tuple[float, float] | np.ndarray = [],
             excludepix: list[float] | tuple[float, float] | np.ndarray = []
             ) -> None:
        """Mask self.data using a 2D or 3D array of dataformask.

        Args:
            dataformask (np.ndarray, optional): 2D or 3D array is used
                for specifying the mask.
            includepix (list, optional): Data in this range survives.
                Defaults to [].
            excludepix (list, optional): Data in this range is masked.
                Defaults to [].
        """
        if dataformask is None:
            dataformask = self.data
        if len(includepix) not in [0, 2] or len(excludepix) not in [0, 2]:
            raise ValueError(
                'includepix and excludepix must be empty or [min, max].')
        if np.ndim(self.data) > np.ndim(dataformask):
            print('The mask is broadcasted.')
            try:
                mask = np.broadcast_to(dataformask, np.shape(self.data))
            except ValueError as exc:
                raise ValueError(
                    'dataformask must be broadcastable to the data shape; '
                    f'got {np.shape(dataformask)} and {np.shape(self.data)}.'
                ) from exc
        else:
            mask = dataformask
        if np.shape(self.data) != np.shape(mask):
            raise ValueError(
                'dataformask must have the same shape as data after '
                f'broadcasting; got {np.shape(dataformask)} and '
                f'{np.shape(self.data)}.')

        if ((len(includepix) == 2 or len(excludepix) == 2)
            and self.data.dtype.kind in 'biu'):
            self.data = self.data.astype(float)
        if len(includepix) == 2:
            self.data[(mask < includepix[0]) + (includepix[1] < mask)] = np.nan
        if len(excludepix) == 2:
            self.data[(excludepix[0] < mask) * (mask < excludepix[1])] = np.nan

    def _gfit_profile(self, prof: np.ndarray, gaussfit: bool
                      ) -> dict[str, list[np.ndarray]]:
        if not gaussfit:
            return {}

        gfitres = {}
        nprof = len(prof)
        res = [None] * nprof
        for i in range(nprof):
            res[i] = gaussfit1d(xdata=self.v, ydata=prof[i],
                                sigma=None, show=True,
                                nwalkersperdim=8)
        gfitres['best'] = [a['popt'][:3] for a in res]
        gfitres['error'] = [a['perr'][:3] for a in res]
        return gfitres

    def profile(self, coords: list[str] | tuple[str, ...] | np.ndarray = [],
                xlist: list[_RealScalar] | tuple[_RealScalar, ...] | np.ndarray = [],
                ylist: list[_RealScalar] | tuple[_RealScalar, ...] | np.ndarray = [],
                ellipse: list[_RealScalar] | tuple[_RealScalar, _RealScalar, _RealScalar]
                | list[list[_RealScalar] | tuple[_RealScalar, _RealScalar, _RealScalar]]
                | np.ndarray | None = None,
                ninterp: int | np.integer = 1,
                flux: bool = False, gaussfit: bool = False
                ) -> tuple[np.ndarray, np.ndarray, dict[str, list[np.ndarray]]]:
        """Get a list of line profiles at given spatial coordinates.

        Args:
            coords (list, tuple, or np.ndarray, optional): Text coordinates. 
                Defaults to [].
            xlist (list, tuple, or np.ndarray, optional): Offset from center. 
                Defaults to [].
            ylist (list, tuple, or np.ndarray, optional): Offset from center. 
                Defaults to [].
            ellipse (list, tuple, or np.ndarray, optional): One
                [major, minor, pa] triple shared by all positions, or one
                triple per position. None selects nearest pixels.
            ninterp (int or np.integer, optional): Number of points for interpolation.
                Defaults to 1.
            flux (bool, optional): Jy/beam to Jy. Defaults to False.
            gaussfit (bool, optional): Fit the profiles. Defaults to
                False.

        Returns:
            tuple: (v, profiles, fit results). Profiles have shape
            (number of positions, number of channels). Fit results are {}
            when disabled, or best/error lists of parameter arrays.
        """
        if np.ndim(self.data) != 3 or self.v is None:
            raise ValueError('profile() requires 3D data with v, y, and x axes.')
        if len(xlist) != len(ylist):
            raise ValueError('xlist and ylist must have the same length.')
        if len(coords) == 0 and len(xlist) == 0:
            raise ValueError('Provide coords or matching xlist and ylist values.')

        data, xf, yf = filled2d(self.data, self.x, self.y, ninterp)
        x, y = np.meshgrid(xf, yf)
        if len(coords) > 0:
            xlist, ylist = coord2xy(coords, self.center) * 3600.
        nprof = len(xlist)
        dtype = data.dtype if data.dtype.kind in 'fc' else np.dtype(float)
        prof = np.empty((nprof, len(self.v)), dtype=dtype)
        if ellipse is None:
            ellipses = np.zeros((nprof, 3))
        else:
            ellipses = np.asarray(ellipse)
            if ellipses.shape == (3,):
                ellipses = np.broadcast_to(ellipses, (nprof, 3))
            elif ellipses.shape != (nprof, 3):
                raise ValueError('ellipse must be one [major, minor, pa] triple '
                                 'or exactly one triple per position.')
            if (ellipses.dtype.kind not in 'iuf'
                    or not np.all(np.isfinite(ellipses))
                    or np.any(ellipses[:, :2] < 0)):
                raise ValueError('ellipse values must be finite real numbers '
                                 + 'with nonnegative major and minor sizes.')
        calc = np.sum if flux else np.mean
        for i, (xc, yc, e) in enumerate(zip(xlist, ylist, ellipses)):
            major, minor, pa = e
            z = dot2d(Mrot(-pa), [y - yc, x - xc])
            if major == 0 or minor == 0:
                r = np.hypot(*z)
                idx = np.unravel_index(np.argmin(r), np.shape(r))
                prof[i] = [d[idx] for d in data]
            else:
                r = np.hypot(*dot2d(Mfac(2 / major, 2 / minor), z))
                prof[i] = [calc(d[r <= 1]) for d in data]
        if flux:
            if None in self.beam or None in [self.dx, self.dy]:
                raise ValueError(
                    'flux=True requires a complete beam and x/y pixel sizes.')
            else:
                Omega = np.pi * self.beam[0] * self.beam[1] / 4. / np.log(2.)
                prof *= np.abs(self.dx * self.dy) / Omega
        gfitres = self._gfit_profile(prof, gaussfit)
        return self.v, prof, gfitres

    def rotate(self, pa: float | np.floating = 0, **kwargs: Any) -> None:
        """Counter clockwise rotation with respect to the center.

        Args:
            pa (float, optional): Position angle in the unit of degree.
                Defaults to 0.
        """
        yxnew = dot2d(Mrot(-pa), np.meshgrid(self.y, self.x, indexing='ij'))
        self.data = RGIxy(self.y, self.x, self.data, yxnew, **kwargs)
        if self.beam[2] is not None:
            if isinstance(self.beam, tuple):
                self.beam = list(self.beam)
            self.beam[2] = _normalize_float(self.beam[2] + pa)

    def slice(self, length: float | np.floating = 0,
              pa: float | np.floating = 0,
              dx: float | np.floating | None = None, **kwargs: Any
              ) -> tuple[np.ndarray, np.ndarray]:
        """Get 1D slice with given a length and a position-angle.

        Args:
            length (float, optional): Slice line length. Defaults to 0.
            pa (float, optional): Position angle in the unit of degree.
                Defaults to 0.
            dx (float, optional): Grid increment. Defaults to None.

        Returns:
            tuple: (positions, data). Data have shape (npositions,) for
            images or (nchannels, npositions) for cubes, including when
            length=0 gives one position.
        """
        if self.data.ndim not in (2, 3):
            raise ValueError('slice() requires a 2D image or 3D cube.')
        if dx is None and self.dx is not None:
            dx = np.abs(self.dx)
        if dx is None:
            raise ValueError('slice() requires dx or an x grid with pixel size.')
        if not np.isfinite(dx) or dx <= 0:
            raise ValueError('dx must be finite and positive.')
        if not np.isfinite(length) or length < 0:
            raise ValueError('length must be finite and nonnegative.')

        n = int(np.ceil(length / 2 / dx))
        r = np.linspace(-n, n, 2 * n + 1) * dx
        pa_rad = np.radians(pa)
        yg, xg = r * np.cos(pa_rad), r * np.sin(pa_rad)
        z = RGIxy(self.y, self.x, self.data, (yg, xg), **kwargs)
        z = np.asarray(z).reshape(self.data.shape[:-2] + (len(r),))
        return r, z

    def todict(self) -> dict[str, Any]:
        """Output the attributes as a dictionary that can be input to
        PlotAstroData.

        Returns:
            dict: Output that can be input to PlotAstroData. Arrays and
            lists are shared references, not copies.
        """
        d = {'data': self.data, 'x': self.x, 'y': self.y, 'v': self.v,
             'fitsimage': self.fitsimage, 'beam': self.beam, 'Tb': self.Tb,
             'restfreq': self.restfreq, 'cfactor': self.cfactor,
             'sigma': self.sigma, 'center': self.center, 'pv': self.pv,
             'bunit': self.bunit}
        return d

    def _put_header(self, h: Header | dict[str, Any], t: Literal['x', 'y', 'v'],
                    crpix: int | np.integer, crval: _RealScalar,
                    cdelt: _RealScalar | None = None) -> None:
        """Write axis values in the units already selected by the caller.

        FITS numeric header cards use Python floats: conversion here is an
        explicit serialization boundary and may narrow extended precision.
        """
        axis = {'x': 1, 'y': 2, 'v': 2 if self.pv else 3}[t]
        if cdelt is None:
            cdelt = getattr(self, f'd{t}')
        if cdelt is None:
            raise ValueError(f'Cannot write the {t} axis without pixel spacing.')
        h[f'NAXIS{axis}'] = len(getattr(self, t))
        h[f'CRPIX{axis}'] = int(crpix)
        h[f'CRVAL{axis}'] = float(crval)
        h[f'CDELT{axis}'] = float(cdelt)

    def _get_cvdv_in_freq(self, ck: int | np.integer
                         ) -> tuple[float | np.longdouble,
                                    float | np.longdouble]:
        """Convert velocity reference and spacing, preserving scalar precision."""
        if self.v is None or self.dv is None:
            raise ValueError('Writing a velocity axis requires coordinates'
                             + ' and spacing.')
        cv, dv = self.v[ck], self.dv
        if self.restfreq is None or self.restfreq == 0:
            s = 'No valid restfreq. The velocity axis is saved as is.'
            warnings.warn(s, UserWarning)
        else:
            if not np.isfinite(self.restfreq) or self.restfreq < 0:
                raise ValueError('restfreq must be finite and positive,'
                                 + ' or None/0.')
            cv = (1 - cv / cu.c_kms) * self.restfreq
            dv = -dv / cu.c_kms * self.restfreq
        return (_normalize_float(cv) if isinstance(cv, np.floating)
                else float(cv),
                _normalize_float(dv) if isinstance(dv, np.floating)
                else float(dv))

    @_need_multipixels
    def writetofits(self, fitsimage: str | PathLike[str] = 'out.fits',
                    header: Header | dict[str, Any] | None = None) -> None:
        """Write a single image or cube, overwriting an existing file.

        Spatial axes use degrees, undoing any distance scaling applied by
        AstroFrame.read. Known sky centers use celestial coordinates;
        otherwise spatial axes are linear angular offsets. PV spatial axes
        are also angular offsets. Spectral axes use Hz
        when restfreq is known, otherwise radio velocity in km/s.

        header optionally overrides generated cards. Numeric FITS cards are
        serialized as Python floats, which can narrow extended precision.
        Missing required grid spacing raises ValueError.
        """
        if (not isinstance(self.data, np.ndarray)
                or self.data.ndim not in (2, 3)):
            raise ValueError('writetofits requires a single 2D image'
                             + ' or 3D cube.')
        if not np.isfinite(self.dist) or self.dist <= 0:
            raise ValueError('dist must be finite and positive.')
        # Retain non-coordinate metadata; rebuild WCS for processed data.
        h = dict(self.fitsheader.items()) if self.fitsheader is not None else {}
        for key in list(h):
            if (re.match(r'^(NAXIS|CTYPE|CUNIT|CRPIX|CRVAL|CDELT|CROTA)\d*$', key)
                    or re.match(r'^(PC|CD|PV|PS)\d+_\d+$', key)
                    or key in ('WCSAXES', 'LONPOLE', 'LATPOLE')):
                del h[key]
        ci = nearest_index(self.x)
        celestial = self.center is not None and not self.pv
        if celestial:
            cj = nearest_index(self.y)
            coord = xy2coord([self.x[ci] / self.dist / 3600,
                              self.y[cj] / self.dist / 3600], self.center)
            _, frame, _ = _getframe(coord, as_frame=True)
            cx, cy = frame.ra.degree, frame.dec.degree
            h.update(CTYPE1='RA---SIN', CTYPE2='DEC--SIN',
                     CUNIT1='deg', CUNIT2='deg')
            h.pop('EQUINOX', None)
            h['RADESYS'] = frame.name.upper()
            if hasattr(frame, 'equinox'):
                h['EQUINOX'] = float(frame.equinox.jyear)
            self._put_header(h, 'x', ci + 1, cx, self.dx / self.dist / 3600)
            self._put_header(h, 'y', cj + 1, cy, self.dy / self.dist / 3600)
        else:
            h.update(CTYPE1='LINEAR', CUNIT1='deg')
            self._put_header(h, 'x', ci + 1, self.x[ci] / self.dist / 3600,
                             self.dx / self.dist / 3600)
            if not self.pv:
                cj = nearest_index(self.y)
                h.update(CTYPE2='LINEAR', CUNIT2='deg')
                self._put_header(h, 'y', cj + 1, self.y[cj] / self.dist / 3600,
                                 self.dy / self.dist / 3600)
        if self.pv or self.data.ndim == 3:
            if self.v is None:
                raise ValueError('A cube or PV image requires'
                                 + ' velocity coordinates.')
            ck = nearest_index(self.v)
            cv, dv = self._get_cvdv_in_freq(ck)
            axis = 2 if self.pv else 3
            frequency = self.restfreq is not None and self.restfreq != 0
            h[f'CTYPE{axis}'] = 'FREQ' if frequency else 'VRAD'
            h[f'CUNIT{axis}'] = 'Hz' if frequency else 'km/s'
            self._put_header(h, 'v', ck + 1, cv, dv)
        beam = self.beam_org if self.pv else self.beam
        if beam is not None and None not in beam:
            h['BMAJ'] = float(beam[0] / self.dist / 3600)
            h['BMIN'] = float(beam[1] / self.dist / 3600)
            h['BPA'] = float(beam[2])
        elif self.pv and not all(k in h for k in ('BMAJ', 'BMIN', 'BPA')):
            s = 'Original sky beam unavailable; omitting FITS beam metadata.'
            warnings.warn(s, UserWarning)
            for key in ('BMAJ', 'BMIN', 'BPA'):
                h.pop(key, None)
        if self.bunit is not None:
            h['BUNIT'] = self.bunit
        if self.restfreq is not None and self.restfreq > 0:
            h['RESTFRQ'] = float(self.restfreq)
        h.update(header or {})
        data2fits(d=self.data, h=h, fitsimage=str(fitsimage))


def _as_list(value: Any, n: int, isbeam: bool = False) -> Any:
    if isbeam:
        return [value] * n if np.ndim(value) == 1 else value
    else:
        return value if isinstance(value, list) else [value] * n


def _scalar_if_single(value: Any, n: int) -> Any:
    return value[0] if n == 1 else value


def _get_gridsep(axis: np.ndarray | None
                 ) -> int | float | np.longdouble | None:
    """Return grid spacing as a Python scalar, preserving extended precision."""
    if axis is None or len(axis) <= 1:
        return None
    spacing = axis[1] - axis[0]
    return (int(spacing)
            if isinstance(spacing, np.integer)
            else _normalize_float(spacing))


ASTRODATA_ARGS = ['fitsimage', 'data', 'Tb', 'sigma', 'center', 'restfreq',
                  'cfactor', 'bunit', 'fitsimage_org', 'sigma_org',
                  'beam_org', 'fitsheader', 'pv', 'pvpa']


def _validate_frame_scalar(value: Any, handler: Callable
                           ) -> float | np.longdouble:
    """Preserve long doubles before Pydantic can coerce them to float."""
    if isinstance(value, np.longdouble):
        return value
    return handler(value)


_FrameScalar = Annotated[float | np.longdouble,
                         WrapValidator(_validate_frame_scalar)]
_PositionPair = tuple[_RealScalar, _RealScalar] | list[_RealScalar] | np.ndarray
_Position = str | _PositionPair
_Grid = list[np.ndarray | None]


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class AstroFrame():
    """Parameter set to limit and reshape the data in the AstroData
    format. Ordinary numeric inputs become Python floats; extended-precision
    NumPy floating scalars retain their precision.

    Args:
        vmin (float, optional): Velocity at the upper left. Defaults to
            -1e20.
        vmax (float, optional): Velocity at the lower bottom. Defaults
            to 1e20.
        vsys (float, optional): Each channel shows v-vsys. Defaults to
            0..
        center (str, optional): Central coordinate like '12h34m56.7s
            12d34m56.7s'. Defaults to None.
        fitsimage (str or os.PathLike, optional): Fits to get center.
            Defaults to None.
        rmax (float, optional): The x range is [-rmax, rmax]. The y
            range is [-rmax, rmax]. Defaults to 1e10.
        xmax (float, optional): The x range is [xmin, xmax]. Defaults to
            None.
        xmin (float, optional): The x range is [xmin, xmax]. Defaults to
            None.
        ymax (float, optional): The y range is [ymin, ymax]. Defaults to
            None.
        ymin (float, optional): The y range is [ymin, ymax]. Defaults to
            None.
        dist (float, optional): Change x and y in arcsec to au. Defaults
            to 1..
        xoff (float, optional): Map center relative to the center.
            Defaults to 0.
        yoff (float, optional): Map center relative to the center.
            Defaults to 0.
        xflip (bool, optional): True means left is positive x. Defaults
            to True.
        yflip (bool, optional): True means bottom is positive y.
            Defaults to False.
        swapxy (bool, optional): True means x and y are swapped.
            Defaults to False.
        pv (bool, optional): Mode for PV diagram. Defaults to False.
        quadrants (str, optional): '13' or '24'. Quadrants to take mean.
            None means not taking mean. Defaults to None.
    """
    rmax: _FrameScalar = 1e10
    xmax: _FrameScalar | None = None
    xmin: _FrameScalar | None = None
    ymax: _FrameScalar | None = None
    ymin: _FrameScalar | None = None
    dist: _FrameScalar = 1.0
    center: str | None = None
    fitsimage: str | PathLike[str] | None = None
    xoff: _FrameScalar = 0.0
    yoff: _FrameScalar = 0.0
    vsys: _FrameScalar = 0.0
    vmin: _FrameScalar = -1e20
    vmax: _FrameScalar = 1e20
    xflip: bool = True
    yflip: bool = False
    swapxy: bool = False
    pv: bool = False
    quadrants: Literal['13', '24'] | None = None

    xskip: int = field(init=False, default=1, repr=False, compare=False)
    yskip: int = field(init=False, default=1, repr=False, compare=False)
    xdir: int = field(init=False, repr=False, compare=False)
    ydir: int = field(init=False, repr=False, compare=False)
    xlim: list[_FrameScalar] = field(init=False, repr=False, compare=False)
    ylim: list[_FrameScalar] = field(init=False, repr=False, compare=False)
    vlim: list[_FrameScalar] = field(init=False, repr=False, compare=False)
    Xlim: list[_FrameScalar] = field(init=False, repr=False, compare=False)
    Ylim: list[_FrameScalar] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.xdir = -1 if self.xflip else 1
        self.ydir = -1 if self.yflip else 1
        self.xmin = -self.rmax if self.xmin is None else self.xmin
        self.xmax = self.rmax if self.xmax is None else self.xmax
        self.ymin = -self.rmax if self.ymin is None else self.ymin
        self.ymax = self.rmax if self.ymax is None else self.ymax
        if self.xflip:
            self.xmin, self.xmax = self.xmax, self.xmin
        if self.yflip:
            self.ymin, self.ymax = self.ymax, self.ymin
        xlim = [self.xoff + self.xmin, self.xoff + self.xmax]
        ylim = [self.yoff + self.ymin, self.yoff + self.ymax]
        vlim = [self.vmin, self.vmax]
        if self.pv:
            xlim = sorted(xlim)
            if not self.xflip:
                xlim.reverse()
        self.xlim, self.ylim, self.vlim = xlim, ylim, vlim
        _x, _y = (xlim, vlim) if self.pv else (xlim, ylim)
        self.Xlim, self.Ylim = (_y, _x) if self.swapxy else (_x, _y)
        if self.quadrants is not None:
            self.Xlim = [0, self.rmax]
            self.Ylim = [0, min(self.vmax, -self.vmin)]
        if self.fitsimage is not None and self.center is None:
            self.center = FitsData(self.fitsimage).get_center()

    def pos2xy(self, poslist: _Position | list[_Position]
               | tuple[_Position, ...]) -> np.ndarray:
        """Convert sky coordinates or relative numeric pairs to coordinates.

        Args:
            poslist: One sky-coordinate string or numeric pair, or a list,
                tuple, or array of positions. Mixed strings and pairs are
                supported. Sky-coordinate strings require a center.

        Returns:
            np.ndarray: Coordinates with shape (2, N), including (2, 1)
            for one position and (2, 0) for an empty collection.

        Raises:
            ValueError: If a numeric position is not a pair of real scalars,
                or sky coordinates are supplied without a center.
        """
        def is_pair(value: Any) -> bool:
            return (isinstance(value, (list, tuple, np.ndarray))
                    and not (isinstance(value, np.ndarray) and value.ndim == 0)
                    and len(value) == 2
                    and all(isinstance(v, (numbers.Real,
                                           np.integer,
                                           np.floating))
                            for v in value))

        if isinstance(poslist, str):
            poslist = [poslist]
        elif isinstance(poslist, np.ndarray) and poslist.ndim == 0:
            raise ValueError('Each position must be a string or a numeric pair.')
        elif isinstance(poslist, (list, tuple, np.ndarray)):
            # Test elements without converting a mixed collection to an array.
            if (len(poslist) == 2
                    and all(isinstance(v, (numbers.Real,
                                           np.integer,
                                           np.floating)) for v in poslist)):
                poslist = [poslist]
        else:
            raise ValueError('Each position must be a string or a numeric pair.')
        for position in poslist:
            if not isinstance(position, str) and not is_pair(position):
                raise ValueError('Each numeric position must contain'
                                 + ' two real scalars.')
        if self.center is None and any(isinstance(p, str) for p in poslist):
            clsname = type(self).__name__
            raise ValueError(f'{clsname}.pos2xy() requires "center"'
                             + ' when poslist contains sky-coordinate strings.'
                             + ' Set "center" explicitly or "fitsimage"'
                             + ' from which the center can be determined.')
        x, y = [None] * len(poslist), [None] * len(poslist)
        for i, p in enumerate(poslist):
            if isinstance(p, str):
                x[i], y[i] = coord2xy(p, self.center) * 3600.
            else:
                x[i], y[i] = rel2abs(*p, self.Xlim, self.Ylim)
        return np.array([x, y])

    def _get_restfreq(self, header: Header | dict[str, Any]
                      ) -> float | np.longdouble | None:
        """Extract rest frequency from FITS header."""
        if 'RESTFRQ' in header:
            return _normalize_float(header['RESTFRQ'])
        if 'RESTFREQ' in header:
            return _normalize_float(header['RESTFREQ'])
        if 'NAXIS3' in header and header['NAXIS3'] == 1 and not self.pv:
            return _normalize_float(header['CRVAL3'])
        return None

    def _read_fitsimage(self, d: AstroData, i: int, grid: _Grid) -> _Grid:
        """Read FITS-derived values into d and return the FITS grid."""
        if d.fitsimage[i] is None:
            return grid

        fd = FitsData(d.fitsimage[i])
        if d.fitsheader[i] is None:
            d.fitsheader[i] = fd.get_header()
        if d.center[i] is None and not self.pv:
            d.center[i] = fd.get_center()
        if d.restfreq[i] is None:
            d.restfreq[i] = self._get_restfreq(d.fitsheader[i])
        d.data[i] = fd.get_data()
        grid = fd.get_grid(center=d.center[i], dist=self.dist,
                           restfreq=d.restfreq[i], vsys=self.vsys,
                           pv=self.pv)
        if fd.wcsrot:
            d.center[i] = fd.get_center()
        d.beam[i] = fd.get_beam(dist=self.dist)
        d.bunit[i] = fd.get_header('BUNIT')
        return list(grid)

    def _shift_center(self, d: AstroData, i: int, grid: _Grid) -> _Grid:
        corg = d.center[i]
        cnew = self.center
        if self.pv or cnew is None or corg is None or corg == cnew:
            return grid

        cx, cy = coord2xy(corg, cnew) * 3600
        grid[0] = grid[0] + cx  # Don't use += cx.
        grid[1] = grid[1] + cy  # Don't use += cy.
        d.center[i] = cnew
        return grid

    def _ascending_v(self, d: AstroData, i: int,
                     v: np.ndarray | None) -> None:
        if v is not None and len(v) > 1 and v[1] < v[0]:
            d.data[i], v = d.data[i][::-1], v[::-1]
            print('Velocity has been inverted.')
        d.v = v

    def _xyskip(self, d: AstroData, i: int,
                x: np.ndarray, y: np.ndarray) -> None:
        d.x = x[::self.xskip]
        d.y = y[::self.yskip]
        data = np.moveaxis(d.data[i], [-2, -1], [0, 1])
        data = data[::self.yskip, ::self.xskip]
        d.data[i] = np.moveaxis(data, [0, 1], [-2, -1])

    def _validate_data_grid(self, data: np.ndarray, grid: _Grid,
                            dataset: int) -> None:
        """Validate array axes before trimming or spatial subsampling.
        """
        shape = np.shape(np.squeeze(data))
        if len(shape) not in [2, 3]:
            raise ValueError(
                f'Dataset {dataset} must be 2D or 3D; got shape {shape}.')

        x, y, v = grid
        required = [('x', x, shape[-1])]
        if self.pv:
            required.append(('v', v, shape[-2]))
        else:
            required.append(('y', y, shape[-2]))
            if len(shape) == 3:
                required.append(('v', v, shape[-3]))
        for name, axis, expected in required:
            if axis is None:
                raise ValueError(
                    f'Dataset {dataset} requires a {name} coordinate array.')
            if np.ndim(axis) != 1:
                raise ValueError(
                    f'Dataset {dataset} {name} must be a 1D array.')
            if len(axis) != expected:
                raise ValueError(
                    f'Dataset {dataset} {name} has length {len(axis)}, but '
                    f'the corresponding data axis has length {expected}.')

    def _trim_skip(self, d: AstroData, i: int, grid: _Grid) -> None:
        d.data[i], grid = trim(data=d.data[i],
                               x=grid[0], y=grid[1], v=grid[2],
                               xlim=self.xlim, ylim=self.ylim,
                               vlim=self.vlim, pv=self.pv)
        self._ascending_v(d, i, v=grid[2])
        grid = [grid[0], d.v] if self.pv else [grid[0], grid[1]]
        if self.swapxy:
            grid.reverse()
            d.data[i] = np.swapaxes(d.data[i], -2, -1)
        self._xyskip(d, i, x=grid[0], y=grid[1])
        if self.pv:
            d.v = d.y
        for axis in ['x', 'y', 'v']:
            setattr(d, f'd{axis}', _get_gridsep(getattr(d, axis)))

    def _convert_to_Tb(self, d: AstroData, i: int) -> None:
        """Convert Jy/beam data to brightness temperature if requested.
        """
        if not d.Tb[i]:
            return

        header = {'RESTFREQ': d.restfreq[i]}
        if d.beam[i][0] is not None and d.beam[i][1] is not None:
            header['BMAJ'] = d.beam[i][0] / 3600 / self.dist
            header['BMIN'] = d.beam[i][1] / 3600 / self.dist
        else:
            dx = d.dy if self.swapxy else d.dx
            if dx is None:
                raise ValueError('Brightness-temperature conversion requires'
                                 + ' beam sizes or at least two spatial pixels'
                                 + ' to determine pixel size.')
            header.update(CDELT1=dx / 3600, CUNIT1='deg')
        factor = Jy2K(header=header)
        d.data[i] = d.data[i] * factor
        if d.sigma[i] is not None:
            d.sigma[i] = _normalize_float(d.sigma[i] * factor)

    def _set_pv_beam(self, d: AstroData, i: int) -> None:
        """Set effective PV beam."""
        if not self.pv or d.pv[i] or None in d.beam[i]:
            return

        bmaj, bmin, bpa = d.beam_org[i] = d.beam[i]
        if d.pvpa[i] is None:
            d.pvpa[i] = _normalize_float(bpa)
            print('pvpa is not specified. pvpa=bpa is assumed.')
        angle = np.radians(bpa - d.pvpa[i])
        beam_incut = 1 / np.hypot(np.cos(angle) / bmaj, np.sin(angle) / bmin)
        d.beam[i] = np.array([np.abs(d.dv), beam_incut, 0])

    def _read_one(self, d: AstroData, i: int, grid: _Grid) -> None:
        if d.center[i] == 'common':
            d.center[i] = self.center
        d.sigma_org[i] = d.sigma[i]
        grid = self._read_fitsimage(d, i, grid=grid)
        if d.data[i] is not None:
            self._validate_data_grid(d.data[i], grid, i)
            d.sigma[i] = estimate_rms(d.data[i], d.sigma[i])
            grid = self._shift_center(d, i, grid)
            self._trim_skip(d, i, grid)
            if self.quadrants is not None:
                d.data[i], d.x, d.y \
                    = quadrantmean(d.data[i], d.x, d.y, self.quadrants)
            d.data[i] = d.data[i] * d.cfactor[i]
            if d.sigma[i] is not None:
                d.sigma[i] = _normalize_float(d.sigma[i] * d.cfactor[i])
            self._convert_to_Tb(d, i)
            self._set_pv_beam(d, i)
            d.pv[i] = self.pv
        d.Tb[i] = False
        d.cfactor[i] = 1
        d.fitsimage_org[i] = d.fitsimage[i]
        d.fitsimage[i] = None

    def read(self, d: AstroData, xskip: int | np.integer = 1,
             yskip: int | np.integer = 1) -> None:
        """Get data, grid, sigma, beam, and bunit from AstroData, which
        is a part of the input of add_color, add_contour, add_segment,
        and add_rgb.

        This method modifies ``d`` in place. During the read, the input
        fields are normalized to per-dataset lists, FITS-derived values
        are filled, trimming and coordinate-frame changes are applied,
        and bookkeeping fields such as ``fitsimage``, ``fitsimage_org``,
        ``Tb``, ``cfactor``, ``sigma``, and ``dist`` are updated. Multiple datasets
        must have matching processed coordinate grids (rtol=1e-7, atol=0).
        A mismatch raises ValueError; this operation mutates d in place.

        Args:
            d (AstroData): Dataclass for the add_* input.
            xskip, yskip (int or np.integer): Spatial pixel skip. Defaults to 1.
        """
        if (not isinstance(xskip, (int, np.integer)) or xskip < 1
                or not isinstance(yskip, (int, np.integer)) or yskip < 1):
            raise ValueError('xskip and yskip must be positive integers.')
        d.dist = self.dist
        self.xskip = int(xskip)
        self.yskip = int(yskip)
        for name in ASTRODATA_ARGS:
            setattr(d, name, _as_list(getattr(d, name), d.n))
        d.beam = _as_list(d.beam, d.n, isbeam=True)
        for name in ASTRODATA_ARGS + ['beam']:
            value = getattr(d, name)
            if len(value) != d.n:
                raise ValueError(
                    f'{name} must contain one value or {d.n} values; '
                    f'got {len(value)}.')
        grid = [d.x, d.y, d.v]
        shared_grid = None
        for i in range(d.n):
            self._read_one(d, i, grid.copy())
            if d.data[i] is None:
                continue
            current = [d.x, d.y, d.v]
            if shared_grid is None:
                shared_grid = [None if a is None else a.copy() for a in current]
            else:
                for name, expected, actual in zip(('x', 'y', 'v'),
                                                  shared_grid,
                                                  current):
                    same = expected is None and actual is None
                    if expected is not None and actual is not None:
                        same = (expected.shape == actual.shape
                                and np.allclose(expected, actual,
                                                rtol=1e-7, atol=0))
                    if not same:
                        raise ValueError(f'Dataset {i} has a different'
                                         + f' processed {name} grid; datasets'
                                         + ' in one AstroData must share'
                                         + ' coordinate grids.')
        for name in ASTRODATA_ARGS + ['beam']:
            setattr(d, name, _scalar_if_single(getattr(d, name), d.n))
