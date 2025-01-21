import numpy as np
from dataclasses import dataclass
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import curve_fit
from scipy.signal import convolve

from plotastrodata.other_utils import (coord2xy, rel2abs, estimate_rms, trim,
                                       Mfac, Mrot, dot2d, gaussian2d)
from plotastrodata.fits_utils import FitsData, data2fits, Jy2K
from plotastrodata import const_utils as cu


def to4dim(data: np.ndarray) -> np.ndarray:
    """Change a 2D, 3D, or 4D array to a 4D array.

    Args:
        data (np.ndarray): Input data. 2D, 3D, or 4D.

    Returns:
        np.ndarray: Output 4D array.
    """
    if np.ndim(data) == 2:
        d = np.array([[data]])
    elif np.ndim(data) == 3:
        d = np.array([data])
    else:
        d = np.array(data)
    return d


def quadrantmean(data: np.ndarray, x: np.ndarray, y: np.ndarray,
                 quadrants: str = '13') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        return -1

    dx, dy = x[1] - x[0], y[1] - y[0]
    nx = int(np.floor(max(np.abs(x[0]), np.abs(x[-1])) / dx))
    ny = int(np.floor(max(np.abs(y[0]), np.abs(y[-1])) / dy))
    xnew = np.linspace(-nx * dx, nx * dx, 2 * nx + 1)
    ynew = np.linspace(-ny * dy, ny * dy, 2 * ny + 1)
    Xnew, Ynew = np.meshgrid(x, y)
    if quadrants == '13':
        f = RGI((y, x), data, bounds_error=False, fill_value=np.nan)
        datanew = f((Ynew, Xnew))
    elif quadrants == '24':
        f = RGI((y, -x), data, bounds_error=False, fill_value=np.nan)
        datanew = f((Ynew, Xnew))
    else:
        print('quadrants must be \'13\' or \'24\'.')
    datanew = (datanew + datanew[::-1, ::-1]) / 2.
    return datanew[ny:, nx:], xnew[nx:], ynew[ny:]


def RGIxy(y: np.ndarray, x: np.ndarray, data: np.ndarray,
          yxnew: tuple[np.ndarray, np.ndarray] | None = None
          ) -> object | np.ndarray:
    """RGI for x and y at each channel.

    Args:
        y (np.ndarray): 1D array. Second coordinate.
        x (np.ndarray): 1D array. First coordinate.
        data (np.ndarray): 2D, 3D, or 4D array.
        yxnew (tuple, optional): (ynew, xnew), where ynew and xnew are 1D or 2D arrays. Defaults to None.

    Returns:
        np.ndarray: The RGI function or the interpolated array.
    """
    if not np.ndim(data) in [2, 3, 4]:
        print('data must be 2D, 3D, or 4D.')
        return -1

    c4d = to4dim(data)
    c4d[np.isnan(c4d)] = 0
    f = [[RGI((y, x), c2d, bounds_error=False, fill_value=np.nan)
          for c2d in c3d] for c3d in c4d]
    if yxnew is None:
        if len(f) == 1:
            f = f[0]
        if len(f) == 1:
            f = f[0]
        return f
    else:
        return np.squeeze([[f2d(tuple(yxnew)) for f2d in f3d] for f3d in f])


def RGIxyv(v: np.ndarray, y: np.ndarray, x: np.ndarray, data: np.ndarray,
           vyxnew: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
           ) -> object | np.ndarray:
    """RGI in the x-y-v space.

    Args:
        v (np.ndarray): 1D array. Third coordinate.
        y (np.ndarray): 1D array. Second coordinate.
        x (np.ndarray): 1D array. First coordinate.
        data (np.ndarray): 3D or 4D array.
        vyxnew (tuple, optional): (vnew, ynew, xnew), where vnew, ynew, and xnew are 1D or 2D arrays. Defaults to None.

    Returns:
        np.ndarray: The RGI function or the interpolated array.
    """
    if not np.ndim(data) in [3, 4]:
        print('data must be 3D or 4D.')
        return -1

    c4d = to4dim(data)
    c4d[np.isnan(c4d)] = 0
    f = [RGI((v, y, x), c3d, bounds_error=False, fill_value=np.nan) for c3d in c4d]
    if vyxnew is None:
        if len(f) == 1:
            f = f[0]
        return f
    else:
        return np.squeeze([f3d(tuple(vyxnew)) for f3d in f])


def filled2d(data: np.ndarray, x: np.ndarray, y: np.ndarray, n: int = 1
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    d = RGIxy(y, x, data, np.meshgrid(ynew, xnew, indexing='ij'))
    return d, xnew, ynew


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
        self.fitsheader = None
        self.pv = False

    def binning(self, width: list[int, int, int] = [1, 1, 1]):
        """Binning up neighboring pixels in the v, y, and x domain.

        Args:
            width (list, optional): Number of channels, y-pixels, and x-pixels for binning. Defaults to [1, 1, 1].
        """
        width = [1] * (4 - len(width)) + width
        d = to4dim(self.data)
        size = list(np.shape(d))
        newsize = size // np.array(width, dtype=int)
        grid = [None, self.v, self.y, self.x]
        for n in range(4):
            if width[n] == 1:
                continue
            size[n] = newsize[n]
            olddata = np.moveaxis(d, n, 0)
            newdata = np.moveaxis(np.zeros(size), n, 0)
            t = np.zeros(newsize[n])
            for i in range(width[n]):
                t += grid[n][i:i + newsize[n]*width[n]:width[n]]
                newdata += olddata[i:i + newsize[n]*width[n]:width[n]]
            grid[n] = t / width[n]
            d = np.moveaxis(newdata, 0, n) / width[n]
        self.data = np.squeeze(d)
        _, self.v, self.y, self.x = grid

    def centering(self, includexy: bool = True, includev: bool = False):
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
                               np.meshgrid(vnew, ynew, xnew, indexing='ij'))
            self.v, self.y, self.x = vnew, ynew, xnew
        elif includexy:
            self.data = RGIxy(self.y, self.x, self.data,
                              np.meshgrid(ynew, xnew, indexing='ij'))
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

    def circularbeam(self):
        """Make the beam circular by convolving with 1D Gaussian
        """
        if None in self.beam:
            print('No beam.')
            return False

        bmaj, bmin, bpa = self.beam
        self.rotate(-bpa)
        nx = len(self.x) if len(self.x) % 2 == 1 else len(self.x) - 1
        ny = len(self.y) if len(self.y) % 2 == 1 else len(self.y) - 1
        y = np.linspace(-(ny-1) / 2, (ny-1) / 2, ny) * (self.y[1]-self.y[0])
        g1 = np.exp(-4*np.log(2) * y**2 / (bmaj**2 - bmin**2))
        g1 /= np.sqrt(np.pi/4/np.log(2) * bmin * np.sqrt(1 - bmin**2/bmaj**2))
        g = np.zeros((ny, nx))
        g[:, (nx - 1) // 2] = g1
        d = to4dim(self.data)
        d = [[convolve(c, g, mode='same') for c in cc] for cc in d]
        self.data = np.squeeze(d)
        self.rotate(bpa)
        self.beam[1] = self.beam[0]
        self.beam[2] = 0

    def deproject(self, pa: float = 0, incl: float = 0):
        """Exapnd by a factor of 1/cos(incl) in the direction of pa+90 deg.

        Args:
            pa (float, optional): Position angle in the unit of degree. Defaults to 0.
            incl (float, optional): Inclination angle in the unit of degree. Defaults to 0.
        """
        ci = np.cos(np.radians(incl))
        A = np.linalg.multi_dot([Mrot(pa), Mfac(1, ci), Mrot(-pa)])
        yxnew = dot2d(A, np.meshgrid(self.y, self.x, indexing='ij'))
        self.data = RGIxy(self.y, self.x, self.data, yxnew)
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

    def histogram(self, **kwargs) -> tuple:
        """Output histogram of self.data using numpy.histogram. This method can take the arguments of numpy.histogram.

        Returns:
            tuple: (bins, histogram)
        """
        hist, hbin = np.histogram(self.data, **kwargs)
        hbin = (hbin[:-1] + hbin[1:]) / 2
        return hbin, hist

    def gaussfit2d(self, chan: int = None):
        """Fit a 2D Gaussian function to self.data.

        Args:
            chan (int): The channel number where the 2D Gaussian is fitted. Defaults to None.

        Returns:
            dict: The best parameter set (popt), the covariance set (pcov), the best 2D Gaussian array (model), and the residual from the model (residual).
        """
        d = self.data if chan is None else self.data[chan]
        x = self.x
        y = self.y
        p0 = (np.max(d), np.median(x), np.median(y), 1, 1, 0)
        amax = np.max(np.abs(d))
        xmin = np.min(x)
        xmax = np.max(x)
        ymin = np.min(y)
        ymax = np.max(y)
        bounds = [[-amax, xmin, ymin, np.abs(x[1] - x[0]), np.abs(y[1] - y[0]), -90],
                  [amax, xmax, ymax, xmax - xmin, ymax - ymin, 90]]
        x, y = np.meshgrid(x, y)
        popt, pcov = curve_fit(gaussian2d, (x.ravel(), y.ravel()), d.ravel(),
                               p0=p0, bounds=bounds)
        model = gaussian2d((x, y), *popt)
        residual = d - model
        return {'popt': popt, 'pcov': pcov, 'model': model, 'residual': residual}

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
            return False

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
            return False

        if len(coords) > 0:
            xlist, ylist = coord2xy(coords, self.center) * 3600.
        nprof = len(xlist)
        v = self.v
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
        if flux and (None not in self.beam):
            Omega = np.pi * self.beam[0] * self.beam[1] / 4. / np.log(2.)
            dxdy = np.abs((yf[1]-yf[0]) * (xf[1]-xf[0]))
            prof *= dxdy / Omega
        gfitres = {}
        if gaussfit:
            xmin, xmax = np.min(v), np.max(v)
            ymin, ymax = np.min(prof), np.max(prof)
            bounds = [[ymin, xmin, v[1] - v[0]], [ymax, xmax, xmax - xmin]]

            def gauss(x, p, c, w):
                return p * np.exp(-4. * np.log(2.) * ((x - c) / w)**2)

            nprof = len(prof)
            best, error = [None] * nprof, [None] * nprof
            for i in range(nprof):
                popt, pcov = curve_fit(gauss, v, prof[i], bounds=bounds)
                e = np.sqrt(np.diag(pcov))
                print('Gauss (peak, center, FWHM):', popt)
                print('Gauss uncertainties:', e)
                best[i], error[i] = popt, e
            gfitres = {'best': best, 'error': error}
        return v, prof, gfitres

    def rotate(self, pa: float = 0):
        """Counter clockwise rotation with respect to the center.

        Args:
            pa (float, optional): Position angle in the unit of degree. Defaults to 0.
        """
        yxnew = dot2d(Mrot(-pa), np.meshgrid(self.y, self.x, indexing='ij'))
        self.data = RGIxy(self.y, self.x, self.data, yxnew)
        if self.beam[2] is not None:
            self.beam[2] = self.beam[2] + pa

    def slice(self, length: float = 0, pa: float = 0,
              dx: float | None = None) -> np.ndarray:
        """Get 1D slice with given a length and a position-angle.

        Args:
            length (float, optional): Slice line length. Defaults to 0.
            pa (float, optional): Position angle in the unit of degree. Defaults to 0.
            dx (float, optional): Grid increment. Defaults to None.

        Returns:
            np.ndarray: [x, data]. If self.data is 3D, the output data are in the shape of (len(v), len(x)).
        """
        if dx is None:
            dx = np.abs(self.x[1] - self.x[0])
        n = int(np.ceil(length / 2 / dx))
        r = np.linspace(-n, n, 2 * n + 1) * dx
        pa_rad = np.radians(pa)
        yg, xg = r * np.cos(pa_rad), r * np.sin(pa_rad)
        z = RGIxy(self.y, self.x, self.data, (yg, xg))
        return np.array([r, z])

    def todict(self) -> dict:
        """Output the attributes as a dictionary that can be input to PlotAstroData.

        Returns:
            dict: Output that can be input to PlotAstroData.
        """
        d = {'data': self.data, 'x': self.x, 'y': self.y, 'v': self.v,
             'fitsimage': self.fitsimage, 'beam': self.beam, 'Tb': self.Tb,
             'restfreq': self.restfreq, 'cfactor': self.cfactor,
             'sigma': self.sigma, 'center': self.center}
        return d

    def writetofits(self, fitsimage: str = 'out.fits', header: dict = {}):
        """Write out the AstroData to a FITS file.

        Args:
            fitsimage (str, optional): Output FITS file name. Defaults to 'out.fits'.
            header (dict, optional): Header dictionary. Defaults to {}.
        """
        if self.pv:
            print('writetofits does not support PV diagram yet.')
            return

        h = {}
        cx, cy = (0, 0) if self.center is None else coord2xy(self.center)
        h['NAXIS1'] = len(self.x)
        h['CRPIX1'] = np.argmin(np.abs(self.x)) + 1
        h['CRVAL1'] = cx
        h['CDELT1'] = (self.x[1] - self.x[0]) / 3600
        h['NAXIS2'] = len(self.y)
        h['CRPIX2'] = np.argmin(np.abs(self.y)) + 1
        h['CRVAL2'] = cy
        h['CDELT2'] = (self.y[1] - self.y[0]) / 3600
        if self.v is not None:
            h['NAXIS3'] = len(self.v)
            h['CRPIX3'] = i = np.argmin(np.abs(self.v)) + 1
            h['CRVAL3'] = (1 - self.v[i]/cu.c_kms) * self.restfreq
            h['CDELT3'] = (self.v[0]-self.v[1]) / cu.c_kms * self.restfreq
        if None not in self.beam:
            h['BMAJ'] = self.beam[0] / 3600
            h['BMIN'] = self.beam[1] / 3600
            h['BPA'] = self.beam[2]
        h0 = header
        for k in h:
            if k not in h0:
                h0[k] = h[k]
        data2fits(d=self.data, h=h0, templatefits=self.fitsimage_org,
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

    def pos2xy(self, poslist: list[str | list[float, float]] = []) -> np.ndarray:
        """Text or relative to absolute coordinates.

         Args:
            poslist (list, optional): Text coordinates or relative coordinates. Defaults to [].

         Returns:
            np.ndarray: absolute coordinates.
         """
        if np.shape(poslist) == () \
            or (np.shape(poslist) == (2,)
                and type(poslist[0]) is not str):
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
        if type(d.fitsheader) is not list:
            d.fitsheader = [d.fitsheader] * d.n
        if type(d.pv) is not list:
            d.pv = [d.pv] * d.n
        grid0 = [d.x, d.y, d.v]
        for i in range(d.n):
            if d.center[i] == 'common':
                d.center[i] = self.center
            grid = grid0
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
                d.beam[i] = fd.get_beam(dist=self.dist)
                d.bunit[i] = d.fitsheader[i]['BUNIT']
            if d.data[i] is not None:
                d.pv[i] = self.pv
                d.sigma_org[i] = d.sigma[i]
                d.sigma[i] = estimate_rms(d.data[i], d.sigma[i])
                d.data[i], grid = trim(data=d.data[i],
                                       x=grid[0], y=grid[1], v=grid[2],
                                       xlim=self.xlim, ylim=self.ylim,
                                       vlim=self.vlim, pv=self.pv)
                if grid[2] is not None and grid[2][1] < grid[2][0]:
                    d.data[i], grid[2] = d.data[i][::-1], grid[2][::-1]
                    print('Inverted velocity.')
                d.v = grid[2]
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
                d.x, d.y = grid
                if self.quadrants is not None:
                    d.data[i], d.x, d.y \
                        = quadrantmean(d.data[i], d.x, d.y, self.quadrants)
                d.data[i] = d.data[i] * d.cfactor[i]
                if d.sigma[i] is not None:
                    d.sigma[i] = d.sigma[i] * d.cfactor[i]
                if d.Tb[i]:
                    dx = d.y[1] - d.y[0] if self.swapxy else d.x[1] - d.x[0]
                    header = {'CDELT1': dx / 3600,
                              'CUNIT1': 'DEG',
                              'RESTFREQ': d.restfreq[i]}
                    if None not in d.beam[i]:
                        header['BMAJ'] = d.beam[i][0] / 3600
                        header['BMIN'] = d.beam[i][1] / 3600
                    d.data[i] = d.data[i] * Jy2K(header=header)
                    d.sigma[i] = d.sigma[i] * Jy2K(header=header)
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
            d.fitsheader = d.fitsheader[0]
            d.pv = d.pv[0]
