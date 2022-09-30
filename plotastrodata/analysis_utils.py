import numpy as np
from dataclasses import dataclass
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.optimize import curve_fit

from plotastrodata.other_utils import coord2xy, rel2abs, estimate_rms, trim
from plotastrodata.fits_utils import fits2data


def quadrantmean(c: list, x: list, y: list, quadrants: str ='13') -> tuple:
    """Take mean between 1st and 3rd (or 2nd and 4th) quadrants.
    """
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

@dataclass
class AstroData():
    """Data to be processed before plotting"""
    data: np.ndarray = None
    x: np.ndarray = None
    y: np.ndarray = None
    v: np.ndarray = None
    beam: list = None
    fitsimage: str = None
    Tb: bool = False
    sigma: str = None
    center: str = None
    restfrq: float = None
    def __post_init__(self):
        if type(self.Tb) is bool:
            self.data = [self.data]
            self.beam = [self.beam]
            self.fitsimage = [self.fitsimage]
            self.Tb = [self.Tb]
            self.sigma = [self.sigma]
            self.restfrq = [self.restfrq]
        n = len(self.data)
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
        xsort = self.x if self.x[1] > self.x[0] else self.x[::-1]
        csort = self.data if self.x[1] > self.x[0] else self.data[:, ::-1]
        ysort = self.y if self.y[1] > self.y[0] else self.y[::-1]
        csort = csort if self.y[1] > self.y[0] else csort[::-1, :]
        f = RBS(ysort, xsort, csort)
        z = np.squeeze(list(map(f, yg, xg)))
        return np.array([r, z])
    
    def profile(self, coords: list = [], ellipse: list = None,
                flux: bool = False, width: int = 1,
                gaussfit: bool = False) -> tuple:
        if np.ndim(self.data) != 3:
            print('Data must be 3D.')
            return False
        
        xlist, ylist = coord2xy(coords, self.center) * 3600.
        x, y = np.meshgrid(self.x, self.y)
        prof = np.empty(((nprof := len(coords)), len(self.v)))
        if ellipse is None: ellipse = [[0, 0, 0]] * nprof
        for i, (xc, yc, e) in enumerate(zip(xlist, ylist, ellipse)):
            major, minor, pa = e
            z = ((y - yc) + 1j * (x - xc)) / np.exp(1j * np.radians(pa))
            if major == 0 or minor == 0:
                r = np.abs(z)
                idx = np.unravel_index(np.argmin(r), np.shape(r))
                prof[i] = [d[idx] for d in self.data]
            else:
                r = np.abs(np.real(z) / major + 1j *  (np.imag(z) / minor))
                if flux:
                    prof[i] = [np.sum(d[r <= 1]) for d in self.data]
                else:
                    prof[i] = [np.mean(d[r <= 1]) for d in self.data]
        newlen = len(self.v) // (width := int(width))
        w, q = np.zeros(newlen), np.zeros((nprof, newlen))
        for i in range(width):
            w += self.v[i:i + newlen*width:width]
            q += prof[:, i:i + newlen*width:width]
        v, prof = w / width, q / width
        if flux:
            Omega = np.pi * self.beam[0] * self.beam[1] / 4. / np.log(2.)
            dxdy = np.abs((self.y[1]-self.y[0]) * (self.x[1]-self.x[0]))
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
    rmax: float = 1e10
    dist: float = 1
    center: str = None
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
        
    def pos2xy(self, poslist: list = []) -> tuple:
        """Text or relative to absolute coordinates.

         Args:
            poslist (list, optional):
            Text coordinates or relative coordinates. Defaults to [].

         Returns:
            tuple: absolute coordinates.
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
        return x, y

    def read(self, d, xskip: int = 1, yskip: int = 1, cfactor: float = 1):
        """Get data, grid, rms, beam, and bunit from AstroData,
           which is a part of the input of
           add_color, add_contour, add_segment, and add_rgb.

        Args:
            d (AstroData): Dataclass for the add_* input.
            xskip, yskip (int): Spatial pixel skip. Defaults to 1.
            cfactor (float, optional): Data times cfactor. Defaults to 1.
        """
        if d.center == 'common': d.center = self.center
        for i in range(n := len(d.data)):
            grid = None
            if d.data[i] is not None:
                d.data[i], grid \
                    = trim(data=d.data[i], x=d.x, y=d.y, v=d.v,
                           xlim=self.xlim, ylim=self.ylim,
                           vlim=self.vlim, pv=self.pv)
                d.rms[i] = estimate_rms(d.data[i], d.sigma[i])
            if d.fitsimage[i] is not None:
                d.data[i], grid, d.beam[i], d.bunit[i], d.rms[i] \
                    = fits2data(fitsimage=d.fitsimage[i], Tb=d.Tb[i],
                                sigma=d.sigma[i], restfrq=d.restfrq[i],
                                center=d.center, log=False,
                                rmax=self.rmax, dist=self.dist,
                                xoff=self.xoff, yoff=self.yoff,
                                vsys=self.vsys, vmin=self.vmin,
                                vmax=self.vmax, pv=self.pv)
            if d.data[i] is not None:
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
                d.data[i] = d.data[i] * cfactor
                if d.rms[i] is not None: d.rms[i] = d.rms[i] * cfactor
        if n == 1:
            d.data = d.data[0]
            d.beam = d.beam[0]
            d.bunit = d.bunit[0]
            d.rms = d.rms[0]
