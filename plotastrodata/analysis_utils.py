import numpy as np
from dataclasses import dataclass
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import curve_fit
from scipy.signal import convolve
from astropy import constants

from plotastrodata.other_utils import (coord2xy, rel2abs, estimate_rms, trim,
                                       Mfac, Mrot, dot2d)
from plotastrodata.fits_utils import FitsData, data2fits, Jy2K


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
                 quadrants: str ='13') -> tuple:
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
        f = RGI((y, x), data, bounds_error=False, fill_value=0)
        datanew = f((Ynew, Xnew))
    elif quadrants == '24':
        f = RGI((y, -x), data, bounds_error=False, fill_value=0)
        datanew = f((Ynew, Xnew))
    else:
        print('quadrants must be \'13\' or \'24\'.')
    datanew = (datanew + datanew[::-1, ::-1]) / 2.
    return datanew[ny:, nx:], xnew[nx:], ynew[ny:]


def sortRGI(y: np.ndarray, x: np.ndarray, data: np.ndarray,
            ynew: np.ndarray = None, xnew: np.ndarray = None
            ) -> object or np.ndarray:
    """RGI but input x and y can be decreasing.

    Args:
        y (np.ndarray): 1D array. Second coordinate.
        x (np.ndarray): 1D array. First coordinate.
        data (np.ndarray): 2D or 3D array.
        ynew (np.ndarray, optional): 1D array. Defaults to None.
        xnew (np.ndarray, optional): 1D array. Defaults to None.

    Returns:
        np.ndarray: The RGI function or the interpolated array.
    """
    if not np.ndim(data) in [2, 3, 4]:
        print('data must be 2D, 3D, or 4D.')
        return -1
    
    csort, xsort, ysort = to4dim(data), x, y
    if x[0] > x[1]:
        xsort = x[::-1]
        csort = np.moveaxis(np.moveaxis(csort, -1, 0)[::-1], 0, -1)
    if y[0] > y[1]:
        ysort = y[::-1]
        csort = np.moveaxis(np.moveaxis(csort, -2, 0)[::-1], 0, -2)
    csort[np.isnan(csort)] = 0
    f = [[RGI((ysort, xsort), c, bounds_error=False, fill_value=0)
          for c in cc] for cc in csort]
    if ynew is None or xnew is None:
        if len(f) == 1: f = f[0]
        if len(f) == 1: f = f[0]
        return f
    d = [[g((ynew.ravel(), xnew.ravel())) for g in ff] for ff in f]
    d = np.reshape(d, (*np.shape(f), *np.shape(xnew)))
    return np.squeeze(d)


def filled2d(data: np.ndarray, x: np.ndarray, y: np.ndarray,
             n: int = 1) -> tuple:
    """Fill 2D data, 1D x, and 1D y by a factor of n using RGI.

    Args:
        data (np.ndarray): 2D or 3D array.
        x (np.ndarray): 1D array.
        y (np.ndarray): 1D array.
        n (int, optional): How many times more the new grid is. Defaults to 1.

    Returns:
        tuple: The interpolated (data, x, y).
    """
    xf = np.linspace(x[0], x[-1], n * (len(x) - 1) + 1)
    yf = np.linspace(y[0], y[-1], n * (len(y) - 1) + 1)
    xnew, ynew = np.meshgrid(xf, yf)
    d = sortRGI(y, x, data, ynew, xnew)
    return d, xf, yf
    

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
        restfrq (float, optional): Used for velocity and brightness T. Defaults to None.
        cfactor (float, optional): Output data times cfactor. Defaults to 1.
    """
    data: np.ndarray = None
    x: np.ndarray = None
    y: np.ndarray = None
    v: np.ndarray = None
    beam: np.ndarray = np.array([None] * 3)
    fitsimage: str = None
    Tb: bool = False
    sigma: str = 'hist'
    center: str = 'common'
    restfrq: float = None
    cfactor: float = 1
    def __post_init__(self):
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

    def binning(self, width: list = [1, 1, 1]):
        """Binning up neighboring pixels in the v, y, and x domain.

        Args:
            width (list, optional): Number of channels, y-pixels, and x-pixels for binning. Defaults to [1, 1, 1].
        """
        width = [1] * (4 - len(width)) + width
        d = to4dim(self.data)
        size = list(np.shape(d))
        newsize = size // np.array(width, dtype=int)
        grid = [None, self.v, self.y, self.x]
        if self.y is None:
            grid[1], grid[2] = grid[2], grid[1]  # for PV diagram
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
        if self.y is None:
            grid[1], grid[2] = grid[2], grid[1]
        _, self.v, self.y, self.x = grid
            
    def centering(self):
        """Spatial regridding to set the center at (x,y)=(0,0).
        """
        x = self.x - self.x[np.argmin(np.abs(self.x))]
        y = self.y - self.y[np.argmin(np.abs(self.y))]
        xnew, ynew = np.meshgrid(x, y)
        self.data = sortRGI(self.y, self.x, self.data, ynew, xnew)
        self.y, self.x = y, x

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
        ynew, xnew = dot2d(A, np.meshgrid(self.x, self.y)[::-1])
        self.data = sortRGI(self.y, self.x, self.data, ynew, xnew)
        if None not in self.beam:
            bmaj, bmin, bpa = self.beam
            a, b = np.linalg.multi_dot([Mfac(1/bmaj, 1/bmin), Mrot(pa-bpa),
                                        Mfac(1, ci), Mrot(-pa)]).T
            alpha = (np.dot(a, a) + np.dot(b, b)) / 2
            beta = np.dot(a, b)
            gamma = (np.dot(a, a) - np.dot(b, b)) / 2
            bpa_new = np.arctan(beta / gamma) / 2 * np.degrees(1)
            if beta * bpa_new >= 0: bpa_new += 90
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

    def mask(self, dataformask: np.ndarray = None, includepix: list = [],
             excludepix: list = []):
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

    def profile(self, coords: list = [], xlist: list = [], ylist: list = [],
                ellipse: list = None, flux: bool = False,
                gaussfit: bool = False) -> tuple:
        """Get a list of line profiles at given spatial coordinates.

        Args:
            coords (list, optional): Text coordinates. Defaults to [].
            xlist (list, optional): Offset from center. Defaults to [].
            ylist (list, optional): Offset from center. Defaults to [].
            ellipse (list, optional): [major, minor, pa]. For average. Defaults to None.
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
        data, xf, yf = filled2d(self.data, self.x, self.y, 8)
        x, y = np.meshgrid(xf, yf)
        prof = np.empty((nprof, len(v)))
        if ellipse is None: ellipse = [[0, 0, 0]] * nprof
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
            gfitres = {'best':best, 'error':error}
        return v, prof, gfitres
    
    def rotate(self, pa: float = 0):
        """Counter clockwise rotation with respect to the center.

        Args:
            pa (float, optional): Position angle in the unit of degree. Defaults to 0.
        """
        ynew, xnew = dot2d(Mrot(-pa), np.meshgrid(self.x, self.y)[::-1])
        self.data = sortRGI(self.y, self.x, self.data, ynew, xnew)
        if self.beam[2] is not None:
            self.beam[2] = self.beam[2] + pa
    
    def slice(self, length: float = 0, pa: float = 0,
              dx: float = None) -> np.ndarray:
        """Get 1D slice with given a length and a position-angle.

        Args:
            length (float, optional): Slice line length. Defaults to 0.
            pa (float, optional): Position angle in the unit of degree. Defaults to 0.
            dx (float, optional): Grid increment. Defaults to None.

        Returns:
            np.ndarray: [x, data]. If self.data is 3D, the output data are in the shape of (len(v), len(x)).
        """
        if dx is None: dx = np.abs(self.x[1] - self.x[0])
        n = int(np.ceil(length / 2 / dx))
        r = np.linspace(-n, n, 2 * n + 1) * dx
        xg, yg = r * np.sin(np.radians(pa)), r * np.cos(np.radians(pa))
        z = sortRGI(self.y, self.x, self.data, yg, xg)
        return np.array([r, z])

    def todict(self) -> dict:
        """Output the attributes as a dictionary that can be input to PlotAstroData.

        Returns:
            dict: Output that can be input to PlotAstroData.
        """
        d = {'data':self.data, 'x':self.x, 'y':self.y, 'v':self.v,
             'fitsimage':self.fitsimage, 'beam':self.beam, 'Tb':self.Tb,
             'restfrq':self.restfrq, 'cfactor':self.cfactor,
             'sigma':self.sigma, 'center':self.center}
        return d
   
    def writetofits(self, fitsimage: str = 'out.fits', header: dict = {}):
        """Write out the AstroData to a FITS file.

        Args:
            fitsimage (str, optional): Output FITS file name. Defaults to 'out.fits'.
            header (dict, optional): Header dictionary. Defaults to {}.
        """
        if self.y is None:
            print('writetofits does not support PV diagram yet.')
            return False
        
        self.centering()
        cx, cy = (0, 0) if self.center is None else coord2xy(self.center)
        header['NAXIS1'] = len(self.x)
        header['CRPIX1'] = np.argmin(np.abs(self.x)) + 1
        header['CRVAL1'] = cx
        header['CDELT1'] = (self.x[1] - self.x[0]) / 3600
        header['NAXIS2'] = len(self.y)
        header['CRPIX2'] = np.argmin(np.abs(self.y)) + 1
        header['CRVAL2'] = cy
        header['CDELT2'] = (self.y[1] - self.y[0]) / 3600
        if self.v is not None:
            clight = constants.c.to('km*s**(-1)').value
            header['NAXIS3'] = len(self.v)
            header['CRPIX3'] = i = np.argmin(np.abs(self.v)) + 1
            header['CRVAL3'] = (1 - self.v[i]/clight) * self.restfrq
            header['CDELT3'] = (self.v[0]-self.v[1]) /clight*self.restfrq
        if None not in self.beam:
            header['BMAJ'] = self.beam[0] / 3600
            header['BMIN'] = self.beam[1] / 3600
            header['BPA'] = self.beam[2]
        data2fits(d=self.data, h=header, templatefits=self.fitsimage_org,
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
        rmax (float, optional): Map size is 2rmax x 2rmax. Defaults to 1e10.
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
    dist: float = 1
    center: str = None
    fitsimage: str = None
    xoff: float = 0
    yoff: float = 0
    vsys: float = 0
    vmin: float = -1e20
    vmax: float = 1e20
    xflip: bool = True
    yflip: bool = False
    swapxy: bool = False
    pv: bool = False
    quadrants: str = None
    def __post_init__(self):
        self.xdir = xdir = -1 if self.xflip else 1
        self.ydir = ydir = -1 if self.yflip else 1
        xlim = [self.xoff - xdir*self.rmax, self.xoff + xdir*self.rmax]
        ylim = [self.yoff - ydir*self.rmax, self.yoff + ydir*self.rmax]
        vlim = [self.vmin, self.vmax]
        if self.pv: xlim = np.sort(xlim)
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
            self.Ylim = [0, min(self.vmax - self.vsys, self.vsys - self.vmin)]
        if self.fitsimage is not None and self.center is None:
            self.center = FitsData(self.fitsimage).get_center()
        
    def pos2xy(self, poslist: list = []) -> np.ndarray:
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
        if type(d.fitsimage) is not list: d.fitsimage = [d.fitsimage] * d.n
        if type(d.data) is not list: d.data = [d.data] * d.n
        if np.ndim(d.beam) == 1: d.beam = [d.beam] * d.n
        if type(d.Tb) is not list: d.Tb = [d.Tb] * d.n
        if type(d.sigma) is not list: d.sigma = [d.sigma] * d.n
        if type(d.center) is not list: d.center = [d.center] * d.n
        if type(d.restfrq) is not list: d.restfrq = [d.restfrq] * d.n
        if type(d.cfactor) is not list: d.cfactor = [d.cfactor] * d.n
        if type(d.bunit) is not list: d.bunit = [d.bunit] * d.n
        if type(d.fitsimage_org) is not list: d.fitsimage_org = [d.fitsimage_org] * d.n
        if type(d.sigma_org) is not list: d.sigma_org = [d.sigma_org] * d.n
        grid0 = [d.x, d.y, d.v]
        for i in range(d.n):
            if d.center[i] == 'common': d.center[i] = self.center
            grid = grid0
            if d.fitsimage[i] is not None:
                fd = FitsData(d.fitsimage[i])
                if d.center[i] is None and not self.pv:
                    d.center[i] = fd.get_center()
                if d.restfrq[i] is None:
                    h = fd.get_header()
                    if 'RESTFRQ' in h: d.restfrq[i] = h['RESTFRQ']
                    if 'RESTFREQ' in h: d.restfrq[i] = h['RESTFREQ']
                d.data[i] = fd.get_data()
                grid = fd.get_grid(center=d.center[i], dist=self.dist,
                                   restfrq=d.restfrq[i], vsys=self.vsys,
                                   pv=self.pv)
                d.beam[i] = fd.get_beam(dist=self.dist)
                d.bunit[i] = fd.get_header('BUNIT')
            if d.data[i] is not None:
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
                    header = {'CDELT1':(d.x[1] - d.x[0]) / 3600,
                              'CUNIT1':'DEG',
                              'RESTFREQ':d.restfrq[i]}
                    if None not in d.beam[i]:
                        header['BMAJ'] = d.beam[i][0] / 3600
                        header['BMIN'] = d.beam[i][1] / 3600
                    d.data[i] = d.data[i] * Jy2K(header=header)
            d.Tb[i] = False
            d.cfactor[i] = 1
            d.fitsimage_org[i] = d.fitsimage[i]
            d.fitsimage[i] = None
        if d.n == 1:
            d.data = d.data[0]
            d.beam = d.beam[0]
            d.fitsimage = d.fitsimage[0]
            d.Tb = d.Tb[0]
            d.sigma = d.sigma[0]
            d.center = d.center[0]
            d.restfrq = d.restfrq[0]
            d.cfactor = d.cfactor[0]
            d.bunit = d.bunit[0]
            d.fitsimage_org = d.fitsimage_org[0]
