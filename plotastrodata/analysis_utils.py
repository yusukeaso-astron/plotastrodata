import numpy as np
from dataclasses import dataclass
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.optimize import curve_fit

from plotastrodata.other_utils import coord2xy, rel2abs, estimate_rms, trim
from plotastrodata.fits_utils import FitsData



def quadrantmean(c: list, x: list, y: list, quadrants: str ='13') -> tuple:
    """Take mean between 1st and 3rd (or 2nd and 4th) quadrants."""
    dx, dy = x[1] - x[0], y[1] - y[0]
    nx = int(np.floor(max(np.abs(x[0]), np.abs(x[-1])) / dx))
    ny = int(np.floor(max(np.abs(y[0]), np.abs(y[-1])) / dy))
    xnew = np.linspace(-nx * dx, nx * dx, 2 * nx + 1)
    ynew = np.linspace(-ny * dy, ny * dy, 2 * ny + 1)
    if quadrants == '13':
        cnew = RBS(y, x, c)(ynew, xnew)
    elif quadrants == '24':
        cnew = RBS(y, -x, c)(ynew, xnew)
    else:
        print('quadrants must be \'13\' or \'24\'.')
    cnew = (cnew + cnew[::-1, ::-1]) / 2.
    return cnew[ny:, nx:], xnew[nx:], ynew[ny:]


def sortRBS(y: list, x: list, data: list,
            ynew: list = None, xnew: list = None):
    """RBS but input x and y can be decreasing."""
    xsort = x if x[1] > x[0] else x[::-1]
    csort = data if x[1] > x[0] else data[:, ::-1]
    ysort = y if y[1] > y[0] else y[::-1]
    csort = csort if y[1] > y[0] else csort[::-1, :]
    f = RBS(ysort, xsort, csort)
    if ynew is None or xnew is None:
        return f
    x1d, y1d = np.ravel(xnew), np.ravel(ynew)
    d = np.reshape(np.squeeze(list(map(f, y1d, x1d))), np.shape(xnew))
    return d


def filled2d(data: list, x: list, y: list, n: list = 1) -> list:
    """Fill 2D data, 1D x, and 1D y by a factor of n using RBS."""
    if not np.ndim(data) in [2, 3]:
        print('data must be 2D or 3D.')
        return -1
    
    xf = np.linspace(x[0], x[-1], n * (len(x) - 1) + 1)
    yf = np.linspace(y[0], y[-1], n * (len(y) - 1) + 1)
    xsort = xf if xf[1] > xf[0] else xf[::-1]
    ysort = yf if yf[1] > yf[0] else yf[::-1]
    if np.ndim(data) == 2: data = [data]
    df = [None] * len(data)
    for i, d in enumerate(data):
        d = np.squeeze(sortRBS(y, x, d)(ysort, xsort))
        d = d if x[1] > x[0] else d[:, ::-1]
        d = d if y[1] > y[0] else d[::-1, :]
        df[i] = d
    return np.squeeze(df), xf, yf
    

@dataclass
class AstroData():
    """Data to be processed and parameters for processing the data.

    data (list, optional): 2D or 3D array. Defaults to None.
    x (list, optional): 1D array. Defaults to None.
    y (list, optional): 1D array. Defaults to None.
    v (list, optional): 1D array. Defaults to None.
    beam (list, optional): [bmaj, bmin, bpa]. Defaults ot [None, None, None].
    fitsimage (str, optional): Input fits name. Defaults to None.
    Tb (bool, optional):
        True means the mapped data are brightness T. Defaults to False.
    sigma (float or str, optional):
        Noise level or method for measuring it. Defaults to 'out'.
    center (str, optional):
        Text coordinates. 'common' means initialized value.
        Defaults to 'common'.
    restfrq (float, optional):
        Used for velocity and brightness T. Defaults to None.
    cfactor (float, optional):
        Output data times cfactor. Defaults to 1.
    """
    data: np.ndarray = None
    x: np.ndarray = None
    y: np.ndarray = None
    v: np.ndarray = None
    beam: np.ndarray = np.array([None] * 3)
    fitsimage: str = None
    Tb: bool = False
    sigma: str = 'out'
    center: str = 'common'
    restfrq: float = None
    cfactor: float = 1
    def __post_init__(self):
        if self.fitsimage is not None:
            n = 1 if type(self.fitsimage) is str else len(self.fitsimage)
            self.data = None
        elif self.data is not None:
            n = 1 if type(self.data) is not list else len(self.data)
        else:
            n = 0
            print('Either data or fitsimage must be given.')
        if type(self.fitsimage) is not list:
            self.fitsimage = [self.fitsimage] * n
        if type(self.data) is not list: self.data = [self.data] * n
        if np.ndim(self.beam) == 1: self.beam = [self.beam] * n
        if type(self.Tb) is not list: self.Tb = [self.Tb] * n
        if type(self.sigma) is not list: self.sigma = [self.sigma] * n
        if type(self.center) is not list: self.center = [self.center] * n
        if type(self.restfrq) is not list: self.restfrq = [self.restfrq] * n
        if type(self.cfactor) is not list: self.cfactor = [self.cfactor] * n
        self.rms = [None] * n
        self.bunit = [''] * n
        
    def slice(self, length: float = 0, pa: float = 0,
              dx: float = None) -> list:
        """Get 1D slice with given a length and a position-angle.

        Args:
            length (float, optional): Slice line length. Defaults to 0.
            pa (float, optional): Degree. Position angle. Defaults to 0.
            dx (float, optional): Grid increment. Defaults to None.

        Returns:
            list: [x, y], where x is the grid and the y is the intensity.
        """
        if np.ndim(self.data) != 2:
            print('Data must be 2D.')
            return False
        
        if dx is None: dx = np.abs(self.x[1] - self.x[0])
        n = int(np.ceil(length / 2 / dx))
        r = np.linspace(-n, n, 2 * n + 1) * dx
        xg, yg = r * np.sin(np.radians(pa)), r * np.cos(np.radians(pa))
        z = sortRBS(self.y, self.x, self.data, yg, xg)
        return np.array([r, z])
    
    def centering(self):
        X = self.x - self.x[np.argmin(np.abs(self.x))]
        Y = self.y - self.y[np.argmin(np.abs(self.y))]
        x, y = np.meshgrid(X, Y)
        d = [self.data] if np.ndim(self.data) == 2 else self.data
        self.data = np.squeeze([sortRBS(self.y, self.x, c, y, x) for c in d])
        self.y, self.x = Y, X

    def deproject(self, pa: float = 0, incl: float = 0):
        pa, ci = np.radians(pa), np.cos(np.radians(incl))
        x, y = np.meshgrid(self.x, self.y)
        z = (y + 1j * x) / np.exp(1j * pa)
        y, x = np.real(z), np.imag(z) * ci
        z = (y + 1j * x) * np.exp(1j * pa)
        y, x = np.real(z), np.imag(z)
        d = [self.data] if np.ndim(self.data) == 2 else self.data
        self.data = np.squeeze([sortRBS(self.y, self.x, c, y, x) for c in d])
        F = lambda f0, f1: np.array([[f0, 0], [0, f1]])
        R = lambda p: np.array([[np.cos(p), -np.sin(p)],
                                [np.sin(p),  np.cos(p)]])
        bmaj, bmin, bpa = self.beam[0], self.beam[1], np.radians(self.beam[2])
        A = np.linalg.multi_dot([F(1/bmaj,1/bmin), R(pa-bpa), F(1,ci), R(-pa)])
        alpha = (A[0, 0]**2 + A[1, 0]**2 + A[0, 1]**2 + A[1, 1]**2) / 2
        beta = A[0, 0]*A[0, 1] + A[1, 0]*A[1, 1]
        gamma = (A[0, 0]**2 + A[1, 0]**2 - A[0, 1]**2 - A[1, 1]**2) / 2
        bpa_new = np.arctan(beta / gamma) / 2 * np.degrees(1)
        if beta * bpa_new > 0: bpa_new += 90
        Det = np.sqrt(beta**2 + gamma**2)
        bmaj_new = 1 / np.sqrt(alpha - Det)
        bmin_new = 1 / np.sqrt(alpha + Det)
        self.beam = np.array([bmaj_new, bmin_new, bpa_new])

    def rotate(self, pa: float = 0):
        x, y = np.meshgrid(self.x, self.y)
        z = (y + 1j * x) / np.exp(1j * np.radians(pa))
        y, x = np.real(z), np.imag(z)
        d = [self.data] if np.ndim(self.data) == 2 else self.data
        self.data = np.squeeze([sortRBS(self.y, self.x, c, y, x) for c in d])
        self.beam[2] = self.beam[2] + pa
    
    def profile(self, coords: list = [], xlist: list = [], ylist: list = [],
                ellipse: list = None, flux: bool = False, width: int = 1,
                gaussfit: bool = False) -> tuple:
        """Get a list of line profiles at given spatial coordinates.

        Args:
            coords (list, optional): Text coordinates. Defaults to [].
            xlist (list, optional): Offset from center. Defaults to [].
            ylist (list, optional): Offset from center. Defaults to [].
            ellipse (list, optional):
                [major, minor, pa]. For average. Defaults to None.
            flux (bool, optional): Jy/beam to Jy. Defaults to False.
            width (int, optional): Rebinning step along v. Defaults to 1.
            gaussfit (bool, optional): Fit the profiles. Defaults to False.

        Returns:
            tuple: (v, list of profiles, result of Gaussian fit)
        """
        if np.ndim(self.data) != 3:
            print('Data must be 3D.')
            return False

        if len(coords) > 0:
            xlist, ylist = coord2xy(coords, self.center) * 3600.
        data, xf, yf = filled2d(self.data, self.x, self.y, 8)
        x, y = np.meshgrid(xf, yf)
        prof = np.empty(((nprof := len(xlist)), len(self.v)))
        if ellipse is None: ellipse = [[0, 0, 0]] * nprof
        for i, (xc, yc, e) in enumerate(zip(xlist, ylist, ellipse)):
            major, minor, pa = e
            z = ((y - yc) + 1j * (x - xc)) / np.exp(1j * np.radians(pa))
            if major == 0 or minor == 0:
                r = np.abs(z)
                idx = np.unravel_index(np.argmin(r), np.shape(r))
                prof[i] = [d[idx] for d in data]
            else:
                r = np.abs(np.real(z) / major + 1j *  (np.imag(z) / minor))
                if flux:
                    prof[i] = [np.sum(d[r <= 1]) for d in data]
                else:
                    prof[i] = [np.mean(d[r <= 1]) for d in data]
        newlen = len(self.v) // (width := int(width))
        w, q = np.zeros(newlen), np.zeros((nprof, newlen))
        for i in range(width):
            w += self.v[i:i + newlen*width:width]
            q += prof[:, i:i + newlen*width:width]
        v, prof = w / width, q / width
        if flux:
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
   

@dataclass
class AstroFrame():
    """
    vmin (float, optional):
        Velocity at the upper left. Defaults to -1e10.
    vmax (float, optional):
        Velocity at the lower bottom. Defaults to 1e10.
    vsys (float, optional):
        Each channel shows v-vsys. Defaults to 0..
    center (str, optional):
        Central coordinate like '12h34m56.7s 12d34m56.7s'.
        Defaults to None.
    fitsimage (str, optional): Fits to get center. Defaults to None.
    rmax (float, optional):
        Map size is 2rmax x 2rmax. Defaults to 1e10.
    dist (float, optional):
        Change x and y in arcsec to au. Defaults to 1..
    xoff (float, optional):
        Map center relative to the center. Defaults to 0.
    yoff (float, optional):
        Map center relative to the center. Defaults to 0.
    xflip (bool, optional):
        True means left is positive x. Defaults to True.
    yflip (bool, optional):
        True means bottom is positive y. Defaults to False.
    swapxy (bool, optional):
        True means x and y are swapped. Defaults to False.
    pv (bool, optional): Mode for PV diagram. Defaults to False.
    quadrants (str, optional): '13' or '24'. Quadrants to take mean.
        None means not taking mean. Defaults to None.
    """
    rmax: float = 1e10
    dist: float = 1
    center: str = None
    fitsimage: str = None
    xoff: float = 0
    yoff: float = 0
    vsys: float = 0
    vmin: float = -1e10
    vmax: float = 1e10
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
        if self.quadrants is not None:
            xlim = [0, self.rmax]
            vlim = [0, min(self.vmax - self.vsys, self.vsys - self.vmin)]
        if self.pv: xlim, ylim = np.sort(xlim), vlim
        if self.swapxy: xlim, ylim = ylim, xlim
        self.xlim = xlim
        self.ylim = ylim
        self.vlim = vlim
        if self.fitsimage is not None and self.center is None:
            self.center = FitsData(self.fitsimage).get_center()
        
    def pos2xy(self, poslist: list = []) -> list:
        """Text or relative to absolute coordinates.

         Args:
            poslist (list, optional):
            Text coordinates or relative coordinates. Defaults to [].

         Returns:
            list: absolute coordinates.
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
                x[i], y[i] = rel2abs(*p, self.xlim, self.ylim)
        return np.array([x, y])

    def read(self, d: AstroData, xskip: int = 1, yskip: int = 1):
        """Get data, grid, rms, beam, and bunit from AstroData,
           which is a part of the input of
           add_color, add_contour, add_segment, and add_rgb.

        Args:
            d (AstroData): Dataclass for the add_* input.
            xskip, yskip (int): Spatial pixel skip. Defaults to 1.
            cfactor (float, optional): Data times cfactor. Defaults to 1.
        """
        for i in range(n := len(d.fitsimage)):
            if d.center[i] == 'common': d.center[i] = self.center
            grid = [d.x, d.y, d.v]
            if d.fitsimage[i] is not None:
                fd = FitsData(d.fitsimage[i])
                if d.center[i] is None and not self.pv:
                    d.center[i] = fd.get_center()
                d.data[i] = fd.get_data(Tb=d.Tb[i], restfrq=d.restfrq[i])
                grid = fd.get_grid(center=d.center[i], dist=self.dist,
                                   restfrq=d.restfrq[i], vsys=self.vsys,
                                   pv=self.pv)
                d.beam[i] = fd.get_beam(dist=self.dist)
                d.bunit[i] = fd.get_header('BUNIT')
            if d.data[i] is not None:
                d.rms[i] = estimate_rms(d.data[i], d.sigma[i])
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
                    grid = grid[::-1]
                    d.data[i] = np.moveaxis(d.data[i], 1, 0)
                grid[0] = grid[0][::xskip]
                grid[1] = grid[1][::yskip]
                if np.ndim(d.data[i]) == 3:
                    d.data[i] = d.data[i][:, ::yskip, ::xskip]
                else:
                    d.data[i] = d.data[i][::yskip, ::xskip]
                d.x, d.y = grid
                if self.quadrants is not None:
                    d.data[i], d.x, d.y \
                        = quadrantmean(d.data[i], d.x, d.y, self.quadrants)
                d.data[i] = d.data[i] * d.cfactor[i]
                if d.rms[i] is not None: d.rms[i] = d.rms[i] * d.cfactor[i]
        if n == 1:
            d.data = d.data[0]
            d.beam = d.beam[0]
            d.fitsimage = d.fitsimage[0]
            d.Tb = d.Tb[0]
            d.sigma = d.sigma[0]
            d.center = d.center[0]
            d.restfrq = d.restfrq[0]
            d.cfactor = d.cfactor[0]
            d.bunit = d.bunit[0]
            d.rms = d.rms[0]
