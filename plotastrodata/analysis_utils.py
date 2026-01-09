import numpy as np
from dataclasses import dataclass
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import curve_fit
from scipy.signal import convolve

from plotastrodata.coord_utils import coord2xy, xy2coord, rel2abs
from plotastrodata.matrix_utils import Mfac, Mrot, dot2d
from plotastrodata.other_utils import (estimate_rms, trim,
                                       gaussian2d, isdeg,
                                       RGIxy, RGIxyv, to4dim)
from plotastrodata.fits_utils import FitsData, data2fits, Jy2K
from plotastrodata import const_utils as cu
from plotastrodata.fitting_utils import EmceeCorner


def quadrantmean(data: np.ndarray, x: np.ndarray, y: np.ndarray,
                 quadrants: str = '13'
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
        print('data must be 2D.')
        return

    dx, dy = x[1] - x[0], y[1] - y[0]
    nx = int(np.floor(max(np.abs(x[0]), np.abs(x[-1])) / dx))
    ny = int(np.floor(max(np.abs(y[0]), np.abs(y[-1])) / dy))
    xnew = np.linspace(-nx * dx, nx * dx, 2 * nx + 1)
    ynew = np.linspace(-ny * dy, ny * dy, 2 * ny + 1)
    Xnew, Ynew = np.meshgrid(xnew, ynew)
    if quadrants in ['13', '24']:
        s = 1 if quadrants == '13' else -1
        f = RGI((y, s * x), data, bounds_error=False, fill_value=np.nan)
        datanew = f((Ynew, Xnew))
    else:
        print('quadrants must be \'13\' or \'24\'.')
    datanew = (datanew + datanew[::-1, ::-1]) / 2.
    return datanew[ny:, nx:], xnew[nx:], ynew[ny:]


def filled2d(data: np.ndarray, x: np.ndarray, y: np.ndarray, n: int = 1,
             **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill 2D data, 1D x, and 1D y by a factor of n using RGI.

    Args:
        data (np.ndarray): 2D or 3D array.
        x (np.ndarray): 1D array.
        y (np.ndarray): 1D array.
        n (int, optional): How many times more the new grid is. Defaults to 1.

    Returns:
        tuple: The interpolated (data, x, y).
    """
    xnew = np.linspace(x[0], x[-1], n * (len(x) - 1) + 1)
    ynew = np.linspace(y[0], y[-1], n * (len(y) - 1) + 1)
    d = RGIxy(y, x, data, np.meshgrid(ynew, xnew, indexing='ij'),
              **kwargs)
    return d, xnew, ynew


def _need_multipixels(method):
    def wrapper(self, *args, **kwargs):
        singlepixel = self.dx is None or self.dy is None
        if singlepixel:
            print('No pixel size.')
            return
        return method(self, *args, **kwargs)
    return wrapper


@dataclass
class AstroData():
    """Data to be processed and parameters for processing the data.

    Args:
        data (np.ndarray, optional): 2D or 3D array. Defaults to None.
        x (np.ndarray, optional): 1D array. Defaults to None.
        y (np.ndarray, optional): 1D array. Defaults to None.
        v (np.ndarray, optional): 1D array. Defaults to None.
        beam (np.ndarray, optional): [bmaj, bmin, bpa]. Defaults ot [None, None, None].
        fitsimage (str, optional): Input fits name. Defaults to None.
        Tb (bool, optional): True means the mapped data are brightness T. Defaults to False.
        sigma (float or str, optional): Noise level or method for measuring it. Defaults to 'hist'.
        center (str, optional): Text coordinates. 'common' means initialized value. Defaults to 'common'.
        restfreq (float, optional): Used for velocity and brightness T. Defaults to None.
        cfactor (float, optional): Output data times cfactor. Defaults to 1.
        pvpa (float, optional): Position angle of the PV cut. Defaults to None.
    """
    data: np.ndarray | None = None
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    v: np.ndarray | None = None
    beam: np.ndarray | tuple[None] = (None, None, None)
    fitsimage: str | None = None
    Tb: bool = False
    sigma: str = 'hist'
    center: str = 'common'
    restfreq: float | None = None
    cfactor: float = 1
    pvpa: float | None = None
    pv: bool = False

    def __post_init__(self):
        n = 0
        if self.fitsimage is not None:
            if type(self.fitsimage) is not list:
                n = 1
            elif any(a is not None for a in self.fitsimage):
                n = len(self.fitsimage)
            else:
                n = 0
            if n > 0:
                self.data = None
        if self.data is not None:
            if type(self.data) is not list:
                n = 1
            elif any(a is not None for a in self.data):
                n = len(self.data)
            else:
                n = 0
        if n == 0:
            print('Either data or fitsimage must be given.')
        self.n = n
        self.bunit = ''
        self.fitsimage_org = None
        self.sigma_org = None
        self.beam_org = None
        self.fitsheader = None

    def binning(self, width: list[int, int, int] = [1, 1, 1]):
        """Binning up neighboring pixels in the v, y, and x domain.

        Args:
            width (list, optional): Number of channels, y-pixels, and x-pixels for binning. Defaults to [1, 1, 1].
        """
        w = [1] * (4 - len(width)) + list(width)
        d = to4dim(self.data)
        size = np.array(np.shape(d))
        w = np.array(w, dtype=int)
        if np.any(w > size):
            w = np.minimum(w, size)
            ws = ', '.join([f'{s:d}' for s in w[1:]])
            print(f'width was changed to [{ws}].')
        newsize = size // w
        grid = [None, self.v, self.y, self.x]
        dgrid = [None, self.dv, self.dy, self.dx]
        for n in range(1, 4):
            if w[n] == 1 or grid[n] is None:
                continue
            if dgrid[n] is None:
                s = ['v', 'y', 'x'][n - 1]
                print(f'Skip binning in the {s}-axis because d{s} is None.')
                continue
            size[n] = newsize[n]
            olddata = np.moveaxis(d, n, 0)
            newdata = np.moveaxis(np.zeros(size), n, 0)
            t = np.zeros(newsize[n])
            for i in range(w[n]):
                i_stop = i + newsize[n] * w[n]
                i_step = w[n]
                t += grid[n][i:i_stop:i_step]
                newdata += olddata[i:i_stop:i_step]
            grid[n] = t / w[n]
            dgrid[n] = dgrid[n] * w[n]
            d = np.moveaxis(newdata, 0, n) / w[n]
        self.data = np.squeeze(d)
        _, self.v, self.y, self.x = grid
        _, self.dv, self.dy, self.dx = dgrid

    def centering(self, includexy: bool = True, includev: bool = False,
                  **kwargs):
        """Spatial regridding to set the center at (x,y,v)=(0,0,0).

        Args:
            includexy (bool, optional): Centering in the x and y directions at each channel. Defaults to True.
            includev (bool, optional): Centering in the v direction at each position. Defaults to False.
        """
        if includexy:
            xnew = self.x - self.x[np.argmin(np.abs(self.x))]
            ynew = self.y - self.y[np.argmin(np.abs(self.y))]
        if includev:
            vnew = self.v - self.v[np.argmin(np.abs(self.v))]
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
            a = np.empty((ny, nx, nv))
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
    def circularbeam(self):
        """Make the beam circular by convolving with 1D Gaussian
        """
        if None in self.beam:
            print('No beam.')
            return

        bmaj, bmin, bpa = self.beam
        self.rotate(-bpa)
        nx = len(self.x) if len(self.x) % 2 == 1 else len(self.x) - 1
        ny = len(self.y) if len(self.y) % 2 == 1 else len(self.y) - 1
        y = np.linspace(-(ny-1) / 2, (ny-1) / 2, ny) * np.abs(self.dy)
        g1 = np.exp(-4*np.log(2) * y**2 / (bmaj**2 - bmin**2))
        e = np.sqrt(1 - bmin**2 / bmaj**2)
        g1 /= np.sqrt(np.pi / 4 / np.log(2) * bmin * e)
        g = np.zeros((ny, nx))
        g[:, (nx - 1) // 2] = g1
        d = self.data.copy()
        d[np.isnan(d)] = 0
        d = to4dim(d)
        d = [[convolve(c, g, mode='same') for c in cc] for cc in d]
        self.data = np.squeeze(d)
        self.rotate(bpa)
        self.beam[1] = self.beam[0]
        self.beam[2] = 0

    def deproject(self, pa: float = 0, incl: float = 0, **kwargs):
        """Exapnd by a factor of 1/cos(incl) in the direction of pa+90 deg.

        Args:
            pa (float, optional): Position angle in the unit of degree. Defaults to 0.
            incl (float, optional): Inclination angle in the unit of degree. Defaults to 0.
        """
        ci = np.cos(np.radians(incl))
        A = np.linalg.multi_dot([Mrot(pa), Mfac(1, ci), Mrot(-pa)])
        yxnew = dot2d(A, np.meshgrid(self.y, self.x, indexing='ij'))
        self.data = RGIxy(self.y, self.x, self.data, yxnew, **kwargs)
        if None not in self.beam:
            bmaj, bmin, bpa = self.beam
            a, b = np.linalg.multi_dot([Mfac(1/bmaj, 1/bmin), Mrot(pa-bpa),
                                        Mfac(1, ci), Mrot(-pa)]).T
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

    @_need_multipixels
    def fit2d(self, model: object, bounds: np.ndarray,
              progressbar: bool = False,
              kwargs_fit: dict = {}, kwargs_plotcorner: dict = {},
              chan: int | None = None):
        """Fit a given 2D model function to self.data.

        Args:
            model (function): The model function in the form of f(par, x, y).
            bounds (np.ndarray): bounds for fitting_utils.EmceeCorner.
            progressbar (bool, optional): progressbar for fitting_utils.EmceeCorner. Defaults to False.
            kwargs_fit (dict, optional): Arguments for fitting_utils.EmceeCorner.fit.
            kwargs_plotcorner (dict, optional): Arguments for fitting_utils.EmceeCorner.plotcorner.
            chan (int, optional): The channel number where the 2D model is fitted. Defaults to None.

        Returns:
            dict: The parameter sets (popt, plow, pmid, and phigh), the best 2D model array (model), and the residual from the model (residual).
        """
        d = self.data if chan is None else self.data[chan]
        x, y = np.meshgrid(self.x, self.y)
        if None not in self.beam:
            Omega = np.pi * self.beam[0] * self.beam[1] / 4 / np.log(2)
            pixelperbeam = Omega / np.abs(self.dx * self.dy)
        else:
            pixelperbeam = 1.

        def logl(p):
            rss = np.nansum((model(p, x, y) - d)**2)
            return -0.5 * rss / self.sigma**2 / pixelperbeam

        mcmc = EmceeCorner(bounds=bounds, logl=logl, progressbar=progressbar)
        kwargs_fit0 = {}
        kwargs_fit0.update(kwargs_fit)
        mcmc.fit(**kwargs_fit0)
        kwargs_plotcorner0 = {'show': False, 'savefig': None}
        kwargs_plotcorner0.update(kwargs_plotcorner)
        kw_pl = kwargs_plotcorner0
        if kw_pl['show'] or kw_pl['savefig'] is not None:
            mcmc.plotcorner(**kw_pl)
        popt = mcmc.popt
        plow = mcmc.plow
        pmid = mcmc.pmid
        phigh = mcmc.phigh
        modelopt = model(popt, x, y)
        residual = d - modelopt
        return {'popt': popt, 'plow': plow, 'pmid': pmid, 'phigh': phigh,
                'model': modelopt, 'residual': residual}

    @_need_multipixels
    def gaussfit2d(self, chan: int | None = None) -> dict:
        """Fit a 2D Gaussian function to self.data.

        Args:
            chan (int): The channel number where the 2D Gaussian is fitted. Defaults to None.

        Returns:
            dict: The best parameter set (popt), the covariance set (pcov), the best 2D Gaussian array (model), the residual from the model (residual), and the coordinates of the best-fit center (center).
        """
        d = self.data if chan is None else self.data[chan]
        x = self.x
        y = self.y
        ds = np.min([np.abs(self.dx), np.abs(self.dy)])
        p0 = (np.max(d), np.median(x), np.median(y), 5 * ds, 5 * ds, 0)
        amax = np.max(np.abs(d))
        xmin = np.min(x)
        xmax = np.max(x)
        ymin = np.min(y)
        ymax = np.max(y)
        smax = np.max([xmax - xmin, ymax - ymin])
        bounds = [[-amax, xmin, ymin, ds, ds, -90],
                  [amax, xmax, ymax, smax, smax, 90]]
        x, y = np.meshgrid(x, y)
        popt, pcov = curve_fit(gaussian2d,
                               (x.ravel(), y.ravel()), d.ravel(),
                               p0=p0, bounds=bounds)
        model = gaussian2d((x, y), *popt)
        residual = d - model
        if (center := self.center) is not None:
            xy = popt[1:3] / 3600
            newcenter = xy2coord(xy, coordorg=center)
        else:
            newcenter = None
        return {'popt': popt, 'pcov': pcov,
                'model': model, 'residual': residual,
                'center': newcenter}

    def histogram(self, **kwargs) -> tuple:
        """Output histogram of self.data using numpy.histogram. This method can take the arguments of numpy.histogram.

        Returns:
            tuple: (bins, histogram)
        """
        hist, hbin = np.histogram(self.data, **kwargs)
        hbin = (hbin[:-1] + hbin[1:]) / 2
        return hbin, hist

    def mask(self, dataformask: np.ndarray | None = None,
             includepix: list[float, float] = [],
             excludepix: list[float, float] = []):
        """Mask self.data using a 2D or 3D array of dataformask.

        Args:
            dataformask (np.ndarray, optional): 2D or 3D array is used for specifying the mask.
            includepix (list, optional): Data in this range survivies. Defaults to [].
            excludepix (list, optional): Data in this range is masked. Defaults to [].
        """
        if dataformask is None:
            dataformask = self.data
        if np.ndim(self.data) > np.ndim(dataformask):
            print('The mask is broadcasted.')
            mask = np.full(np.shape(self.data), dataformask)
        else:
            mask = dataformask
        if np.shape(self.data) != np.shape(mask):
            print('The dataformask has a different shape.')
            return

        if len(includepix) == 2:
            self.data[(mask < includepix[0]) + (includepix[1] < mask)] = np.nan
        if len(excludepix) == 2:
            self.data[(excludepix[0] < mask) * (mask < excludepix[1])] = np.nan

    def profile(self, coords: list[str] = [],
                xlist: list[float] = [], ylist: list[float] = [],
                ellipse: list[float, float, float] | None = None,
                ninterp: int = 1,
                flux: bool = False, gaussfit: bool = False
                ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Get a list of line profiles at given spatial coordinates.

        Args:
            coords (list, optional): Text coordinates. Defaults to [].
            xlist (list, optional): Offset from center. Defaults to [].
            ylist (list, optional): Offset from center. Defaults to [].
            ellipse (list, optional): [major, minor, pa]. For average. Defaults to None.
            ninterp (int, optional): Number of points for interpolation. Defaults to 1.
            flux (bool, optional): Jy/beam to Jy. Defaults to False.
            gaussfit (bool, optional): Fit the profiles. Defaults to False.

        Returns:
            tuple: (v, list of profiles, result of Gaussian fit)
        """
        if np.ndim(self.data) != 3 or self.v is None:
            print('Data must be 3D with the v, y, and x axes.')
            return

        if len(coords) > 0:
            xlist, ylist = coord2xy(coords, self.center) * 3600.
        nprof = len(xlist)
        v = self.v
        dv = self.dv
        data, xf, yf = filled2d(self.data, self.x, self.y, ninterp)
        x, y = np.meshgrid(xf, yf)
        prof = np.empty((nprof, len(v)))
        if ellipse is None:
            ellipse = [[0, 0, 0]] * nprof
        for i, (xc, yc, e) in enumerate(zip(xlist, ylist, ellipse)):
            major, minor, pa = e
            z = dot2d(Mrot(-pa), [y - yc, x - xc])
            if major == 0 or minor == 0:
                r = np.hypot(*z)
                idx = np.unravel_index(np.argmin(r), np.shape(r))
                prof[i] = [d[idx] for d in data]
            else:
                r = np.hypot(*dot2d(Mfac(2/major, 2/minor), z))
                if flux:
                    prof[i] = [np.sum(d[r <= 1]) for d in data]
                else:
                    prof[i] = [np.mean(d[r <= 1]) for d in data]
        if flux:
            if None in self.beam or None in [self.dx, self.dy]:
                print('None in beam, dx, or dy. Flux is not converted.')
            else:
                Omega = np.pi * self.beam[0] * self.beam[1] / 4. / np.log(2.)
                prof *= np.abs(self.dx * self.dy) / Omega
        gfitres = {}
        if gaussfit:
            xmin, xmax = np.min(v), np.max(v)
            ymin, ymax = np.min(prof), np.max(prof)
            bounds = [[ymin, xmin, np.abs(dv)], [ymax, xmax, xmax - xmin]]

            def gauss(x, p, c, w):
                return p * np.exp(-4. * np.log(2.) * ((x - c) / w)**2)

            nprof = len(prof)
            best, error = [None] * nprof, [None] * nprof
            for i in range(nprof):
                popt, pcov = curve_fit(gauss, v, prof[i], bounds=bounds)
                perr = np.sqrt(np.diag(pcov))
                print('Gauss (peak, center, FWHM):', popt)
                print('Gauss uncertainties:', perr)
                best[i], error[i] = popt, perr
            gfitres = {'best': best, 'error': error}
        return v, prof, gfitres

    def rotate(self, pa: float = 0, **kwargs):
        """Counter clockwise rotation with respect to the center.

        Args:
            pa (float, optional): Position angle in the unit of degree. Defaults to 0.
        """
        yxnew = dot2d(Mrot(-pa), np.meshgrid(self.y, self.x, indexing='ij'))
        self.data = RGIxy(self.y, self.x, self.data, yxnew, **kwargs)
        if self.beam[2] is not None:
            self.beam[2] = self.beam[2] + pa

    def slice(self, length: float = 0, pa: float = 0,
              dx: float | None = None, **kwargs) -> np.ndarray:
        """Get 1D slice with given a length and a position-angle.

        Args:
            length (float, optional): Slice line length. Defaults to 0.
            pa (float, optional): Position angle in the unit of degree. Defaults to 0.
            dx (float, optional): Grid increment. Defaults to None.

        Returns:
            np.ndarray: [x, data]. If self.data is 3D, the output data are in the shape of (len(v), len(x)).
        """
        if dx is None and self.dx is not None:
            dx = np.abs(self.dx)
        if dx is None:
            print('dx was not found. Please input dx.')
            return

        n = int(np.ceil(length / 2 / dx))
        r = np.linspace(-n, n, 2 * n + 1) * dx
        pa_rad = np.radians(pa)
        yg, xg = r * np.cos(pa_rad), r * np.sin(pa_rad)
        z = RGIxy(self.y, self.x, self.data, (yg, xg), **kwargs)
        return np.array([r, z])

    def todict(self) -> dict:
        """Output the attributes as a dictionary that can be input to PlotAstroData.

        Returns:
            dict: Output that can be input to PlotAstroData.
        """
        d = {'data': self.data, 'x': self.x, 'y': self.y, 'v': self.v,
             'fitsimage': self.fitsimage, 'beam': self.beam, 'Tb': self.Tb,
             'restfreq': self.restfreq, 'cfactor': self.cfactor,
             'sigma': self.sigma, 'center': self.center, 'pv': self.pv}
        return d

    @_need_multipixels
    def writetofits(self, fitsimage: str = 'out.fits', header: dict = {}):
        """Write out the AstroData to a FITS file.

        Args:
            fitsimage (str, optional): Output FITS file name. Defaults to 'out.fits'.
            header (dict, optional): Header dictionary. Defaults to {}.
        """
        fhd = self.fitsheader
        h = {}
        nocent = self.pv or self.center is None
        cx, cy = (0, 0) if nocent else coord2xy(self.center)
        h['NAXIS1'] = len(self.x)
        h['CRPIX1'] = np.argmin(np.abs(self.x)) + 1
        h['CRVAL1'] = cx
        h['CDELT1'] = self.dx
        if fhd is not None and isdeg(fhd['CUNIT1']):
            h['CDELT1'] = h['CDELT1'] / 3600
        vaxis = '2' if self.pv else '3'
        h[f'NAXIS{vaxis}'] = len(self.v)
        k_vmin = np.argmin(np.abs(self.v))
        h[f'CRPIX{vaxis}'] = k_vmin + 1
        h[f'CRVAL{vaxis}'] = (1 - self.v[k_vmin]/cu.c_kms) * self.restfreq
        h[f'CDELT{vaxis}'] = -self.dv / cu.c_kms * self.restfreq
        if not self.pv:
            h['NAXIS2'] = len(self.y)
            h['CRPIX2'] = np.argmin(np.abs(self.y)) + 1
            h['CRVAL2'] = cy
            h['CDELT2'] = self.dy
            if fhd is not None and isdeg(fhd['CUNIT2']):
                h['CDELT2'] = h['CDELT2'] / 3600
        if None not in self.beam:
            if self.pv:
                h['BMAJ'] = self.beam_org[0] / 3600
                h['BMIN'] = self.beam_org[1] / 3600
                h['BPA'] = self.beam_org[2]
            else:
                h['BMAJ'] = self.beam[0] / 3600
                h['BMIN'] = self.beam[1] / 3600
                h['BPA'] = self.beam[2]
        h.update(header)
        data2fits(d=self.data, h=h, templatefits=self.fitsimage_org,
                  fitsimage=fitsimage)


@dataclass
class AstroFrame():
    """Parameter set to limit and reshape the data in the AstroData format.

    Args:
        vmin (float, optional): Velocity at the upper left. Defaults to -1e10.
        vmax (float, optional): Velocity at the lower bottom. Defaults to 1e10.
        vsys (float, optional): Each channel shows v-vsys. Defaults to 0..
        center (str, optional): Central coordinate like '12h34m56.7s 12d34m56.7s'. Defaults to None.
        fitsimage (str, optional): Fits to get center. Defaults to None.
        rmax (float, optional): The x range is [-rmax, rmax]. The y range is [-rmax, ramx]. Defaults to 1e10.
        xmax (float, optional): The x range is [xmin, xmax]. Defaults to None.
        xmin (float, optional): The x range is [xmin, xmax]. Defaults to None.
        ymax (float, optional): The y range is [ymin, ymax]. Defaults to None.
        ymin (float, optional): The y range is [ymin, ymax]. Defaults to None.
        dist (float, optional): Change x and y in arcsec to au. Defaults to 1..
        xoff (float, optional): Map center relative to the center. Defaults to 0.
        yoff (float, optional): Map center relative to the center. Defaults to 0.
        xflip (bool, optional): True means left is positive x. Defaults to True.
        yflip (bool, optional): True means bottom is positive y. Defaults to False.
        swapxy (bool, optional): True means x and y are swapped. Defaults to False.
        pv (bool, optional): Mode for PV diagram. Defaults to False.
        quadrants (str, optional): '13' or '24'. Quadrants to take mean. None means not taking mean. Defaults to None.
    """
    rmax: float = 1e10
    xmax: float | None = None
    xmin: float | None = None
    ymax: float | None = None
    ymin: float | None = None
    dist: float = 1
    center: str | None = None
    fitsimage: str | None = None
    xoff: float = 0
    yoff: float = 0
    vsys: float = 0
    vmin: float = -1e20
    vmax: float = 1e20
    xflip: bool = True
    yflip: bool = False
    swapxy: bool = False
    pv: bool = False
    quadrants: str | None = None

    def __post_init__(self):
        self.xdir = -1 if self.xflip else 1
        self.ydir = -1 if self.yflip else 1
        if self.xmax is None:
            self.xmax = self.rmax
        if self.xmin is None:
            self.xmin = -self.rmax
        if self.ymax is None:
            self.ymax = self.rmax
        if self.ymin is None:
            self.ymin = -self.rmax
        if self.xdir == -1:
            self.xmin, self.xmax = self.xmax, self.xmin
        if self.ydir == -1:
            self.ymin, self.ymax = self.ymax, self.ymin
        xlim = [self.xoff + self.xmin, self.xoff + self.xmax]
        ylim = [self.yoff + self.ymin, self.yoff + self.ymax]
        vlim = [self.vmin, self.vmax]
        if self.pv:
            xlim = np.sort(xlim)
            if not self.xflip:
                xlim = xlim[::-1]
        self.xlim = xlim
        self.ylim = ylim
        self.vlim = vlim
        if self.pv:
            self.Xlim = vlim if self.swapxy else xlim
            self.Ylim = xlim if self.swapxy else vlim
        else:
            self.Xlim = ylim if self.swapxy else xlim
            self.Ylim = xlim if self.swapxy else ylim
        if self.quadrants is not None:
            self.Xlim = [0, self.rmax]
            self.Ylim = [0, min(self.vmax, -self.vmin)]
        if self.fitsimage is not None and self.center is None:
            self.center = FitsData(self.fitsimage).get_center()

    def pos2xy(self, poslist: list[str | list[float, float]] = []
               ) -> np.ndarray:
        """Text or relative to absolute coordinates.

         Args:
            poslist (list, optional): Text coordinates or relative coordinates. Defaults to [].

         Returns:
            np.ndarray: absolute coordinates.
         """
        onexy = np.shape(poslist) == (2,) and type(poslist[0]) is not str
        if np.shape(poslist) == () or onexy:
            poslist = [poslist]
        x, y = [None] * len(poslist), [None] * len(poslist)
        for i, p in enumerate(poslist):
            if type(p) is str:
                x[i], y[i] = coord2xy(p, self.center) * 3600.
            else:
                x[i], y[i] = rel2abs(*p, self.Xlim, self.Ylim)
        return np.array([x, y])

    def read(self, d: AstroData, xskip: int = 1, yskip: int = 1):
        """Get data, grid, sigma, beam, and bunit from AstroData, which is a part of the input of add_color, add_contour, add_segment, and add_rgb.

        Args:
            d (AstroData): Dataclass for the add_* input.
            xskip, yskip (int): Spatial pixel skip. Defaults to 1.
        """
        if type(d.fitsimage) is not list:
            d.fitsimage = [d.fitsimage] * d.n
        if type(d.data) is not list:
            d.data = [d.data] * d.n
        if np.ndim(d.beam) == 1:
            d.beam = [d.beam] * d.n
        if type(d.Tb) is not list:
            d.Tb = [d.Tb] * d.n
        if type(d.sigma) is not list:
            d.sigma = [d.sigma] * d.n
        if type(d.center) is not list:
            d.center = [d.center] * d.n
        if type(d.restfreq) is not list:
            d.restfreq = [d.restfreq] * d.n
        if type(d.cfactor) is not list:
            d.cfactor = [d.cfactor] * d.n
        if type(d.bunit) is not list:
            d.bunit = [d.bunit] * d.n
        if type(d.fitsimage_org) is not list:
            d.fitsimage_org = [d.fitsimage_org] * d.n
        if type(d.sigma_org) is not list:
            d.sigma_org = [d.sigma_org] * d.n
        if type(d.beam_org) is not list:
            d.beam_org = [d.beam_org] * d.n
        if type(d.fitsheader) is not list:
            d.fitsheader = [d.fitsheader] * d.n
        if type(d.pv) is not list:
            d.pv = [d.pv] * d.n
        if type(d.pvpa) is not list:
            d.pvpa = [d.pvpa] * d.n
        grid0 = [d.x, d.y, d.v]
        for i in range(d.n):
            if d.center[i] == 'common':
                d.center[i] = self.center
            grid = grid0.copy()
            if d.fitsimage[i] is not None:
                fd = FitsData(d.fitsimage[i])
                if d.fitsheader[i] is None:
                    d.fitsheader[i] = fd.get_header()
                if d.center[i] is None and not self.pv:
                    d.center[i] = fd.get_center()
                if d.restfreq[i] is None:
                    h = d.fitsheader[i]
                    if 'NAXIS3' in h and h['NAXIS3'] == 1 and not self.pv:
                        d.restfreq[i] = h['CRVAL3']
                    elif 'RESTFRQ' in h:
                        d.restfreq[i] = h['RESTFRQ']
                    elif 'RESTFREQ' in h:
                        d.restfreq[i] = h['RESTFREQ']
                d.data[i] = fd.get_data()
                grid = fd.get_grid(center=d.center[i], dist=self.dist,
                                   restfreq=d.restfreq[i], vsys=self.vsys,
                                   pv=self.pv)
                if fd.wcsrot:
                    d.center[i] = fd.get_center()  # for WCS rotation
                d.beam[i] = fd.get_beam(dist=self.dist)
                d.bunit[i] = fd.get_header('BUNIT')
            if d.data[i] is not None:
                d.sigma_org[i] = d.sigma[i]
                d.sigma[i] = estimate_rms(d.data[i], d.sigma[i])
                diffcent = (not self.pv
                            and self.center is not None
                            and d.center[i] is not None
                            and d.center[i] != self.center)
                if diffcent:
                    cx, cy = coord2xy(d.center[i], self.center) * 3600
                    grid[0] = grid[0] + cx  # Don't use += cx.
                    grid[1] = grid[1] + cy  # Don't use += cy.
                    d.center[i] = self.center
                d.data[i], grid = trim(data=d.data[i],
                                       x=grid[0], y=grid[1], v=grid[2],
                                       xlim=self.xlim, ylim=self.ylim,
                                       vlim=self.vlim, pv=self.pv)
                v = grid[2]
                has_v = v is not None and len(v) > 1
                if has_v and v[1] < v[0]:
                    d.data[i], v = d.data[i][::-1], v[::-1]
                    print('Velocity has been inverted.')
                d.v = v
                d.dv = v[1] - v[0] if has_v else None
                grid = grid[:3:2] if self.pv else grid[:2]
                if self.swapxy:
                    grid = [grid[1], grid[0]]
                    d.data[i] = np.moveaxis(d.data[i], 1, 0)
                grid[0] = grid[0][::xskip]
                grid[1] = grid[1][::yskip]
                a = d.data[i]
                a = np.moveaxis(a, [-2, -1], [0, 1])
                a = a[::yskip, ::xskip]
                a = np.moveaxis(a, [0, 1], [-2, -1])
                d.data[i] = a
                x, y = d.x, d.y = grid
                has_x = x is not None and len(x) > 1
                d.dx = x[1] - x[0] if has_x else None
                has_y = y is not None and len(y) > 1
                d.dy = y[1] - y[0] if has_y else None
                if self.quadrants is not None:
                    d.data[i], d.x, d.y \
                        = quadrantmean(d.data[i], d.x, d.y, self.quadrants)
                d.data[i] = d.data[i] * d.cfactor[i]
                if d.sigma[i] is not None:
                    d.sigma[i] = d.sigma[i] * d.cfactor[i]
                if d.Tb[i]:
                    dx = d.dy if self.swapxy else d.dx
                    header = {'CDELT1': dx / 3600,
                              'CUNIT1': 'DEG',
                              'RESTFREQ': d.restfreq[i]}
                    if None not in d.beam[i]:
                        header['BMAJ'] = d.beam[i][0] / 3600 / self.dist
                        header['BMIN'] = d.beam[i][1] / 3600 / self.dist
                    d.data[i] = d.data[i] * Jy2K(header=header)
                    d.sigma[i] = d.sigma[i] * Jy2K(header=header)
                if self.pv and not d.pv[i] and None not in d.beam[i]:
                    bmaj, bmin, bpa = d.beam_org[i] = d.beam[i]
                    if d.pvpa[i] is None:
                        d.pvpa[i] = bpa
                        print('pvpa is not specified. pvpa=bpa is assumed.')
                    p = np.radians(bpa - d.pvpa[i])
                    b = 1 / np.hypot(np.cos(p) / bmaj, np.sin(p) / bmin)
                    d.beam[i] = np.array([np.abs(d.dv), b, 0])
                d.pv[i] = self.pv
            d.Tb[i] = False
            d.cfactor[i] = 1
            if d.fitsimage[i] is not None:
                d.fitsimage_org[i] = d.fitsimage[i]
            d.fitsimage[i] = None
        if d.n == 1:
            d.data = d.data[0]
            d.beam = d.beam[0]
            d.fitsimage = d.fitsimage[0]
            d.Tb = d.Tb[0]
            d.sigma = d.sigma[0]
            d.center = d.center[0]
            d.restfreq = d.restfreq[0]
            d.cfactor = d.cfactor[0]
            d.bunit = d.bunit[0]
            d.fitsimage_org = d.fitsimage_org[0]
            d.sigma_org = d.sigma_org[0]
            d.beam_org = d.beam_org[0]
            d.fitsheader = d.fitsheader[0]
            d.pv = d.pv[0]
            d.pvpa = d.pvpa[0]
