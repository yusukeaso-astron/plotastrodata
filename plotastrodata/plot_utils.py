import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.optimize import curve_fit

from plotastrodata.other_utils import (coord2xy, xy2coord, rel2abs,
                                       estimate_rms, trim, listing)
from plotastrodata.fits_utils import FitsData, fits2data


    
plt.ioff()  # force to turn off interactive mode

def set_rcparams(fontsize: int = 18, nancolor: str ='w') -> None:
    #plt.rcParams['font.family'] = 'arial'
    plt.rcParams['axes.facecolor'] = nancolor
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['legend.fontsize'] = 15
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.direction'] = 'inout'
    plt.rcParams['ytick.direction'] = 'inout'
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['xtick.major.size'] = 10
    plt.rcParams['ytick.major.size'] = 10
    plt.rcParams['xtick.minor.size'] = 6
    plt.rcParams['ytick.minor.size'] = 6
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['xtick.minor.width'] = 1.5
    plt.rcParams['ytick.minor.width'] = 1.5


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


class PlotAstroData():
    """Make a figure from 2D/3D FITS files or 2D/3D arrays.
    
    Basic rules --- For 3D data, a 1D velocity array or a FITS file
    with a velocity axis must be given to set up channels in each page.
    For 2D/3D data, the spatial center can be read from a FITS file
    or manually given.
    len(v)=1 (default) means to make a 2D figure.
    Spatial lengths are in the unit of arcsec, or au if dist (!= 1) is given.
    Angles are in the unit of degree.
    For region, line, arrow, label, and marker,
    a single input can be treated without a list, e.g., anglelist=60,
    as well as anglelist=[60].
    Each element of poslist supposes a text coordinate
    like '01h23m45.6s 01d23m45.6s' or a list of relative x and y
    like [0.2, 0.3] (0 is left or bottom, 1 is right or top).
    Parameters for original methods in matplotlib.axes.Axes can be
    used as kwargs; see the default kwargs0 for reference.
    Position-velocity diagrams (pv=True) does not yet suppot region, line,
    arrow, and segment because the units of abscissa and ordinate
    are different.
    The parameter sigma can be one of the methods of
    ['edge', 'neg', 'med', 'iter', 'out'] as well as a specific value.
    """
    def __init__(self, fitsimage: str = None,
                 v: list = [0], vskip: int = 1,
                 vmin: float = -1e10, vmax: float = 1e10,
                 vsys: float = 0., veldigit: int = 2,
                 restfrq: float = None,
                 nrows: int = 4, ncols: int = 6,
                 center: str = None, rmax: float = 1e10, dist: float = 1.,
                 xoff: float = 0, yoff: float = 0,
                 xflip: bool = True, yflip: bool = False,
                 pv: bool = False, quadrants: str = None,
                 fontsize: int = None, nancolor: str = 'w',
                 figsize: tuple = None, fig=None, ax=None) -> None:
        """Set up common parameters.

        Args:
            fitsimage (str, optional):
                Used to set up channels. Defaults to None.
            v (list, optional):
                Used to set up channels if fitsimage not given.
                Defaults to [0].
            vskip (int, optional):
                How many channels are skipped. Defaults to 1.
            vmin (float, optional):
                Velocity at the upper left. Defaults to -1e10.
            vmax (float, optional):
                Velocity at the lower bottom. Defaults to 1e10.
            vsys (float, optional):
                Each channel shows v-vsys. Defaults to 0..
            veldigit (int, optional):
                How many digits after the decimal point. Defaults to 2.
            restfrq (float, optional):
                Used for velocity and brightness T. Defaults to None.
            nrows (int, optional): Used for channel maps. Defaults to 4.
            ncols (int, optional): Used for channel maps. Defaults to 6.
            center (str, optional):
                Central coordinate like '12h34m56.7s 12d34m56.7s'.
                Defaults to None.
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
            pv (bool, optional): Mode for PV diagram. Defaults to False.
            quadrants (str, optional): '13' or '24'. Quadrants to take mean.
                None means not taking mean. Defaults to None.
            fontsize (int, optional): rc_Params['font.size'].
                None means 18 (2D) or 12 (3D). Defaults to None.
            nancolor (str, optional):
                Color for masked regions. Defaults to white.
            figsize (tuple, optional): Defaults to None.
            fig (optional): External plt.figure(). Defaults to None.
            ax (optional): External fig.add_subplot(). Defaults to None.
        """
        internalfig = fig is None
        internalax = ax is None
        if fitsimage is not None:
            fd = FitsData(fitsimage)
            _, _, v = fd.get_grid(restfrq=restfrq, vsys=vsys,
                                  vmin=vmin, vmax=vmax)
            if center is None and not pv:
                ra_deg = fd.get_header('CRVAL1')
                dec_deg = fd.get_header('CRVAL2')
                center = xy2coord([ra_deg, dec_deg])
            if v is not None and v[1] < v[0]:
                v = v[::-1]
                print('Inverted velocity.')
        if pv or v is None or len(v) == 1:
            nv = nrows = ncols = npages = nchan = 1
        else:
            nv = len(v := v[::vskip])
            npages = int(np.ceil(nv / nrows / ncols))
            nchan = npages * nrows * ncols
            v = np.r_[v, v[-1] + (np.arange(nchan-nv)+1) * (v[1] - v[0])]
        def nij2ch(n: int, i: int, j: int):
            return n*nrows*ncols + i*ncols + j
        def ch2nij(ch: int) -> list:
            n = ch // (nrows*ncols)
            i = (ch - n*nrows*ncols) // ncols
            j = ch % ncols
            return n, i, j
        if fontsize is None:
            fontsize=18 if nv == 1 else 12
        set_rcparams(fontsize=fontsize, nancolor=nancolor)
        ax = np.empty(nchan, dtype='object') if internalax else [ax]
        for ch in range(nchan):
            n, i, j = ch2nij(ch)
            if figsize is None:
                figsize = (7, 5) if nchan == 1 else (ncols*2, max(nrows*2, 3))
            if internalfig:
                fig = plt.figure(n, figsize=figsize)
            sharex = ax[nij2ch(n, i - 1, j)] if i > 0 else None
            sharey = ax[nij2ch(n, i, j - 1)] if j > 0 else None
            if internalax:
                ax[ch] = fig.add_subplot(nrows, ncols, i*ncols + j + 1,
                                         sharex=sharex, sharey=sharey)
            if nchan > 1:
                fig.subplots_adjust(hspace=0, wspace=0, right=0.87, top=0.87)
                ax[ch].text(0.9 * rmax, 0.7 * rmax,
                            rf'${v[ch]:.{veldigit:d}f}$', color='black',
                            backgroundcolor='white', zorder=20)
        self.fig = None if internalfig else fig
        self.ax = ax
        self.xdir = xdir = -1 if xflip else 1
        self.ydir = ydir = -1 if yflip else 1
        xlim = [xoff - xdir*rmax, xoff + xdir*rmax]
        ylim = [yoff - ydir*rmax, yoff + ydir*rmax]
        vlim = [vmin, vmax]
        if pv: xlim = np.sort(xlim)
        if quadrants is not None:
            xlim = [0, rmax]
            vlim = [0, min(vmax - vsys, vsys - vmin)]
        self.xlim = xlim
        self.ylim = ylim
        if pv: self.ylim = ylim = vlim
        self.rmax = rmax
        self.center = center
        self.dist = dist
        self.rowcol = nrows * ncols
        self.npages = npages
        self.allchan = np.arange(nchan)
        self.bottomleft = nij2ch(np.arange(npages), nrows - 1, 0)
        self.pv = pv
        self.quadrants = quadrants

        def pos2xy(poslist: list = []) -> tuple:
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
                    x[i], y[i] = (coord2xy(p)-coord2xy(center)) * 3600.
                else:
                    x[i], y[i] = rel2abs(*p, xlim, ylim)
            return x, y
        self.pos2xy = pos2xy

        def skipfill(c: list, skip: int = 1) -> list:
            """Skip and fill channels with nan.

            Args:
                c (list): 2D or 3D arrays.
                skip (int, optional): Spatial skip number. Defaults to 1.

            Returns:
                list: 3D arrays skipped and filled with nan.
            """
            if np.ndim(c) == 3:
                d = c[::vskip, ::skip, ::skip]
            else:
                d = c[::skip, ::skip]
                d = np.full((nv, *np.shape(d)), d)
            shape = (nchan - len(d), len(d[0]), len(d[0, 0]))
            dnan = np.full(shape, d[0] * np.nan)
            return np.concatenate((d, dnan), axis=0)
        self.skipfill = skipfill
        
        def readfits(fitsimage: str, Tb: bool = False,
                     sigma: str or float = None,
                     center: str = None, restfrq: float = None) -> tuple:
            """Use fits2data() to read a fits file.

            Args:
                fitsimage (str): Input fits name.
                Tb (bool, optional):
                    True means the output data are brightness temperature.
                    Defaults to False.
                sigma (str or float, optional):
                    Noise level or method for measuring it. Defaults to None.
                center (str, optional):
                    Text coordinates like '12h34m56.7s 12d34m56.7s.
                    Defaults to None.
                restfrq (float, optional):
                    Used for velocity and brightness T. Defaults to None.

            Returns:
                list: [data, (x, y or v), (bmaj, bmin, bpa), bunit, rms]
            """
            data, grid, beam, bunit, rms \
                = fits2data(fitsimage=fitsimage, Tb=Tb, log=False,
                            sigma=sigma, restfrq=restfrq, center=center,
                            rmax=rmax, dist=dist, xoff=xoff, yoff=yoff,
                            vsys=vsys, vmin=vmin, vmax=vmax, pv=pv)
            if grid[2] is not None and grid[2][1] < grid[2][0]:
                data, grid[2] = data[::-1], v[::-1]
                print('Inverted velocity.')
            a = [data, grid[:2], beam, bunit, rms]
            if pv: a[1] = grid[:3:2]
            return a
        self.readfits = readfits
        
        def readdata(data: list = None, x: list = None,
                     y: list = None, v: list = None) -> list:
            """Input data without a fits file.

            Args:
                data (list, optional): 2D or 3D array. Defaults to None.
                x (list, optional): 1D array. Defaults to None.
                y (list, optional): 1D array. Defaults to None.
                v (list, optional): 1D array. Defaults to None.

            Returns:
                list: [data, (x, y or v)]
            """
            dataout, grid = trim(data=data, x=x, y=y, v=v,
                                 xlim=xlim, ylim=ylim, vlim=vlim, pv=pv)
            a = [dataout, grid[:2]]
            if pv: a[1] = grid[:3:2]
            return a
        self.readdata = readdata

        
    def add_region(self, patch: str = 'ellipse', poslist: list = [],
                   majlist: list = [], minlist: list = [], palist: list = [],
                   include_chan: list = None, **kwargs) -> None:
        """Use add_patch() and Rectangle or Ellipse of matplotlib.

        Args:
            patch (str, optional):
                'ellipse' or 'rectangle'. Defaults to 'ellipse'.
            poslist (list, optional): Text or relative center. Defaults to [].
            majlist (list, optional): Ellipse major axis. Defaults to [].
            minlist (list, optional): Ellipse minor axis. Defaults to [].
            palist (list, optional):
                Position angle (north to east). Defaults to [].
            include_chan (list, optional): None means all. Defaults to None.
        """
        kwargs0 = {'facecolor':'none', 'edgecolor':'gray',
                   'linewidth':1.5, 'zorder':10}
        if include_chan is None: include_chan = self.allchan
        if not (patch in ['rectangle', 'ellipse']):
            print('Only patch=\'rectangle\' or \'ellipse\' supported. ')
            return -1
        for x, y, width, height, angle\
            in zip(*self.pos2xy(poslist), *listing(minlist, majlist, palist)):
            for ch, axnow in enumerate(self.ax):
                if not (ch in include_chan):
                    continue
                if self.fig is None:
                    plt.figure(ch // self.rowcol)
                if patch == 'rectangle':
                    a = np.radians(angle)
                    xp = x - (width*np.cos(a) + height*np.sin(a)) / 2.
                    yp = y - (-width*np.sin(a) + height*np.cos(a)) / 2.
                    p = Rectangle
                else:
                    xp, yp = x, y
                    p = Ellipse
                p = p((xp, yp), width=width, height=height,
                      angle=angle * self.xdir, **dict(kwargs0, **kwargs))
                axnow.add_patch(p)
                
    def add_beam(self, bmaj: float = 0, bmin: float = 0,
                 bpa: float = 0, beamcolor: str = 'gray',
                 poslist: list = None) -> None:
        """Use add_region().

        Args:
            bmaj (float, optional): Beam major axis. Defaults to 0.
            bmin (float, optional): Beam minor axis. Defaults to 0.
            bpa (float, optional): Beam position angle. Defaults to 0.
            beamcolor (str, optional): matplotlib color. Defaults to 'gray'.
            poslist (list, optional): text or relative. Defaults to None.
        """
        if poslist is None:
            bpos = max(0.35 * bmaj / self.rmax, 0.1)
            poslist = [[bpos, bpos]]
        self.add_region(patch='ellipse', poslist=poslist,
                        majlist=[bmaj], minlist=[bmin], palist=[bpa],
                        include_chan=self.bottomleft,
                        facecolor=beamcolor, edgecolor=None)
    
    def add_marker(self, poslist: list = [],
                   include_chan: list = None, **kwargs) -> None:
        """Use plot of matplotlib.

        Args:
            poslist (list, optional): Text or relative. Defaults to [].
            include_chan (list, optional): None means all. Defaults to None.
        """
        kwsmark0 = {'marker':'+', 'ms':10, 'mfc':'gray',
                    'mec':'gray', 'mew':2, 'alpha':1}
        if include_chan is None: include_chan = self.allchan
        for ch, axnow in enumerate(self.ax):
            if not (ch in include_chan):
                continue
            for x, y in zip(*self.pos2xy(poslist)):
                axnow.plot(x, y, **dict(kwsmark0, **kwargs), zorder=10)
            
    def add_text(self, poslist: list = [], slist: list = [],
                  include_chan: list = None, **kwargs) -> None:
        """Use text of matplotlib.

        Args:
            poslist (list, optional): Text or relative. Defaults to [].
            slist (list, optional): List of text. Defaults to [].
            include_chan (list, optional): None means all. Defaults to None.
        """
        kwargs0 = {'color':'gray', 'fontsize':15, 'ha':'center',
                   'va':'center', 'zorder':10}
        if include_chan is None: include_chan = self.allchan
        for ch, axnow in enumerate(self.ax):
            if not (ch in include_chan):
                continue
            for x, y, s in zip(*self.pos2xy(poslist), listing(slist)):
                axnow.text(x=x, y=y, s=s, **dict(kwargs0, **kwargs))

    def add_line(self, poslist: list = [], anglelist: list = [],
                 rlist: list = [], include_chan: list = None,
                 **kwargs) -> None:
        """Use plot of matplotlib.

        Args:
            poslist (list, optional): Text or relative. Defaults to [].
            anglelist (list, optional): North to east. Defaults to [].
            rlist (list, optional): List of radius. Defaults to [].
            include_chan (list, optional): None means all. Defaults to None.
        """
        kwargs0 = {'color':'gray', 'linewidth':1.5,
                   'linestyle':'-', 'zorder':10}
        if include_chan is None: include_chan = self.allchan
        for ch, axnow in enumerate(self.ax):
            if not (ch in include_chan):
                continue
            alist = np.radians(anglelist)
            for x, y, a, r \
                in zip(*self.pos2xy(poslist), *listing(alist, rlist)):
                axnow.plot([x, x + r * np.sin(a)],
                           [y, y + r * np.cos(a)],
                           **dict(kwargs0, **kwargs))

    def add_arrow(self, poslist: list = [], anglelist: list = [],
                  rlist: list = [], include_chan: list = None,
                  **kwargs) -> None:
        """Use quiver of matplotlib.

        Args:
            poslist (list, optional): Text or relative. Defaults to [].
            anglelist (list, optional): North to east. Defaults to [].
            rlist (list, optional): List of radius. Defaults to [].
            include_chan (list, optional): None means all. Defaults to None.
        """
        kwargs0 = {'color':'gray', 'width':0.012,
                   'headwidth':5, 'headlength':5, 'zorder':10}
        if include_chan is None: include_chan = self.allchan
        for ch, axnow in enumerate(self.ax):
            if not (ch in include_chan):
                continue
            alist = np.radians(anglelist)
            for x, y, a, r \
                in zip(*self.pos2xy(poslist), *listing(alist, rlist)):
                axnow.quiver(x, y, r * np.sin(a), r * np.cos(a),
                             angles='xy', scale_units='xy', scale=1,
                             **dict(kwargs0, **kwargs))
                
    def add_scalebar(self, length: float = 0, label: str = '',
                     color: str = 'gray', barpos: tuple = (0.8, 0.12),
                     fontsize: float = None, linewidth: float = 3) -> None:
        """Use text and plot of matplotlib.

        Args:
            length (float, optional): In the unit of arcsec. Defaults to 0.
            label (str, optional): Text like '100 au'. Defaults to ''.
            color (str, optional): Same for bar and label. Defaults to 'gray'.
            barpos (tuple, optional):
                Relative position. Defaults to (0.8, 0.12).
            fontsize (float, optional):
                None means 15 if one channel else 20. Defaults to None.
            linewidth (float, optional): Width of the bar. Defaults to 3.
        """
        if length == 0 or label == '':
            print('Please input length and label.')
            return -1
        if fontsize is None:
            fontsize = 20 if len(self.ax) == 1 else 15
        for ch, axnow in enumerate(self.ax):
            if not (ch in self.bottomleft):
                continue
            x, y = self.pos2xy([barpos[0], barpos[1] - 0.13])
            axnow.text(x[0], y[0], label, color=color, size=fontsize,
                       ha='center', va='top', zorder=10)
            x, y = self.pos2xy([barpos[0], barpos[1] + 0.13])
            axnow.plot([x[0] - length/2., x[0] + length/2.], [y[0], y[0]],
                       '-', linewidth=linewidth, color=color)
    
    def add_color(self, fitsimage: str = None,
                  x: list = None, y: list = None, skip: int = 1,
                  v: list = None, c: list = None,
                  center: str = 'common', restfrq: float = None,
                  Tb: bool = False, log: bool = False,
                  cfactor: float = 1, sigma: float or str = 'out',
                  show_cbar: bool = True, cblabel: str = None,
                  cbformat: float = '%.1e', cbticks: list = None,
                  cbticklabels: list = None, cblocation: str = 'right',
                  show_beam: bool = True, beamcolor: str = 'gray',
                  bmaj: float = 0., bmin: float = 0.,
                  bpa: float = 0., **kwargs) -> None:
        """Use pcolormesh of matplotlib.

        Args:
            fitsimage (str, optional): Input fits name. Defaults to None.
            x (list, optional): 1D array. Defaults to None.
            y (list, optional): 1D array. Defaults to None.
            skip (int, optional): Spatial pixel skip. Defaults to 1.
            v (list, optional): 1D array. Defaults to None.
            c (list, optional): 2D or 3D array. Defaults to None.
            center (str, optional):
                Text coordinates. 'common' means initialized value.
                Defaults to 'common'.
            restfrq (float, optional):
                Used for velocity and brightness T. Defaults to None.
            Tb (bool, optional):
                True means the mapped data are brightness T. Defaults to False.
            log (bool, optional):
                True means the mapped data are logarithmic. Defaults to False.
            cfactor (float, optional):
                Output data times cfactor. Defaults to 1.
            sigma (float or str, optional):
                Noise level or method for measuring it. Defaults to 'out'.
            show_cbar (bool, optional): Show color bar. Defaults to True.
            cblabel (str, optional): Colorbar label. Defaults to None.
            cbformat (float, optional):
                Format for ticklabels of colorbar. Defaults to '%.1e'.
            cbticks (list, optional): Ticks of colorbar. Defaults to None.
            cbticklabels (list, optional):
                Ticklabels of colorbar. Defaults to None.
            cblocation (str, optional): 'left', 'top', 'left', 'right'.
                Only for 2D images. Defaults to 'right'.
            show_beam (bool, optional): Defaults to True.
            beamcolor (str, optional): Matplotlib color. Defaults to 'gray'.
            bmaj (float, optional): Beam major axis. Defaults to 0..
            bmin (float, optional): Beam minor axis. Defaults to 0..
            bpa (float, optional): Beam position angle. Defaults to 0..
        """
        kwargs0 = {'cmap':'cubehelix', 'alpha':1, 'zorder':1}
        if center == 'common':
            center = self.center
        if c is not None:
            bunit, rms = '', estimate_rms(c, sigma)
            c, (x, y) = self.readdata(c, x, y, v)    
        if fitsimage is not None:
            c, (x, y), (bmaj, bmin, bpa), bunit, rms \
                = self.readfits(fitsimage, Tb, sigma, center, restfrq)
        if self.quadrants is not None:
            c, x, y = quadrantmean(c, x, y, self.quadrants)
        c = c * cfactor
        rms = rms * cfactor
        self.rms = rms
        if log: c = np.log10(c.clip(c[c > 0].min(), None))
        if 'vmin' in kwargs:
            if log: kwargs['vmin'] = np.log10(kwargs['vmin'])
        else:
            kwargs['vmin'] = np.log10(rms) if log else np.nanmin(c)
        if 'vmax' in kwargs:
            if log: kwargs['vmax'] = np.log10(kwargs['vmax'])
        else:
            kwargs['vmax'] = np.nanmax(c)
        c = c.clip(kwargs['vmin'], kwargs['vmax'])
        x, y = x[::skip], y[::skip]
        c = self.skipfill(c, skip)
        for axnow, cnow in zip(self.ax, c):
            p = axnow.pcolormesh(x, y, cnow, shading='nearest',
                                 **dict(kwargs0, **kwargs))
        for ch in self.bottomleft:
            if not show_cbar:
                break
            cblabel = bunit if cblabel is None else cblabel
            if self.fig is None:
                fig = plt.figure(ch // self.rowcol)
            else:
                fig = self.fig
            if len(self.ax) == 1:
                ax = self.ax[0]
                cb = fig.colorbar(p, ax=ax, label=cblabel,
                                  format=cbformat, location=cblocation)
            else:
                cax = plt.axes([0.88, 0.105, 0.015, 0.77])
                cb = fig.colorbar(p, cax=cax, label=cblabel, format=cbformat)
            cb.ax.tick_params(labelsize=14)
            font = mpl.font_manager.FontProperties(size=16)
            cb.ax.yaxis.label.set_font_properties(font)
            if cbticks is not None:
                cb.set_ticks(np.log10(cbticks) if log else cbticks)
            if cbticklabels is not None:
                cb.set_ticklabels(cbticklabels)
            elif log:
                t = cb.get_ticks()
                t = t[(kwargs['vmin'] < t) * (t < kwargs['vmax'])]
                cb.set_ticks(t)
                cb.set_ticklabels([f'{d:{cbformat[1:]}}' for d in 10**t])
        if show_beam and not self.pv:
            self.add_beam(bmaj, bmin, bpa, beamcolor)

    def add_contour(self, fitsimage: str = None,
                    x: list = None, y: list = None, skip: int = 1,
                    v: list = None, c: list = None,
                    center: str = 'common', restfrq: float = None,
                    sigma: str or float = 'out',
                    levels: list = [-12,-6,-3,3,6,12,24,48,96,192,384],
                    Tb: bool = False,
                    show_beam: bool = True, beamcolor: str = 'gray',
                    bmaj: float = 0., bmin: float = 0., bpa: float = 0.,
                    **kwargs) -> None:
        """Use contour of matplotlib.

        Args:
            fitsimage (str, optional): Input fits name. Defaults to None.
            x (list, optional): 1D array. Defaults to None.
            y (list, optional): 1D array. Defaults to None.
            skip (int, optional): Spatial pixel skip. Defaults to 1.
            v (list, optional): 1D array. Defaults to None.
            c (list, optional): 1D array. Defaults to None.
            center (str, optional):
                Text coordinate. 'common' means initalized value.
                Defaults to 'common'.
            restfrq (float, optional):
                Used for velocity and brightness T. Defaults to None.
            sigma (strorfloat, optional):
                Noise level or method for measuring it. Defaults to 'out'.
            levels (list, optional):
                Contour levels in the unit of sigma.
                Defaults to [-12,-6,-3,3,6,12,24,48,96,192,384].
            Tb (bool, optional):
                True means the mapped data are brightness T. Defaults to False.
            show_beam (bool, optional): Defaults to True.
            beamcolor (str, optional): Matplotlib color. Defaults to 'gray'.
            bmaj (float, optional): Beam major axis. Defaults to 0..
            bmin (float, optional): Beam minor axis. Defaults to 0..
            bpa (float, optional): Beam position angle. Defaults to 0..
        """
        kwargs0 = {'colors':'gray', 'linewidths':1.0, 'zorder':2}
        if center == 'common':
            center = self.center
        if c is not None:
            if np.ndim(c) == 2 and sigma == 'edge': sigma = 'out'
            rms = estimate_rms(c, sigma)
            c, (x, y) = self.readdata(c, x, y, v)
        if fitsimage is not None:
            c, (x, y), (bmaj, bmin, bpa), _, rms \
                = self.readfits(fitsimage, Tb, sigma, center, restfrq)
        self.rms = rms
        if self.quadrants is not None:
            c, x, y = quadrantmean(c, x, y, self.quadrants)
        x, y = x[::skip], y[::skip]
        c = self.skipfill(c, skip)
        for axnow, cnow in zip(self.ax, c):
            axnow.contour(x, y, cnow, np.sort(levels) * rms,
                          **dict(kwargs0, **kwargs))
        if show_beam and not self.pv:
            self.add_beam(bmaj, bmin, bpa, beamcolor)
            
    def add_segment(self, ampfits: str = None, angfits: str = None,
                    Ufits: str = None, Qfits: str = None,
                    x: list = None, y: list = None, skip: int = 1,
                    v: list = None, amp: list = None, ang: list = None,
                    stU: list = None, stQ: list = None,
                    ampfactor: float = 1., angonly: bool = False,
                    rotation: float = 0.,
                    cutoff: float = 3., sigma: str or float = 'out',
                    center: str = 'common', restfrq: float = None,
                    show_beam: bool = True, beamcolor: str = 'gray',
                    bmaj: float = 0., bmin: float = 0., bpa: float = 0.,
                    **kwargs) -> None:
        """Use quiver of matplotlib.

        Args:
            ampfits (str, optional):
                In put fits name. Length of segment. Defaults to None.
            angfits (str, optional):
                In put fits name. North to east. Defaults to None.
            Ufits (str, optional):
                In put fits name. Stokes U. Defaults to None.
            Qfits (str, optional):
                In put fits name. Stokes Q. Defaults to None.
            x (list, optional): 1D array. Defaults to None.
            y (list, optional): 1D array. Defaults to None.
            skip (int, optional): Spatial pixel skip. Defaults to 1.
            v (list, optional): 1D array. Defaults to None.
            amp (list, optional): Length of segment. Defaults to None.
            ang (list, optional): North to east. Defaults to None.
            stU (list, optional): Stokes U. Defaults to None.
            stQ (list, optional): Stokes Q. Defaults to None.
            ampfactor (float, optional):
                Length of segment is amp times ampfactor. Defaults to 1..
            angonly (bool, optional):
                True means amp=1 for all. Defaults to False.
            rotation (float, optional):
                Segment angle is ang + rotation. Defaults to 0..
            cutoff (float, optional):
                Used when amp and ang are calculated from Stokes U and Q.
                In the unit of sigma. Defaults to 3..
            sigma (str or float, optional):
                Noise level or method for measuring it. Defaults to 'out'.
            center (str, optional):
                Text coordinate. 'common' means initialized value.
                Defaults to 'common'.
            restfrq (float, optional):
                Used for velocity and brightness T. Defaults to None.
            show_beam (bool, optional): Defaults to True.
            beamcolor (str, optional): Matplotlib color. Defaults to 'gray'.
            bmaj (float, optional): Beam major axis. Defaults to 0..
            bmin (float, optional): Beam minor axis. Defaults to 0..
            bpa (float, optional): Beam position angle. Defaults to 0..
        """
        kwargs0 = {'angles':'xy', 'scale_units':'xy', 'color':'gray',
                   'pivot':'mid', 'headwidth':0, 'headlength':0,
                   'headaxislength':0, 'width':0.007, 'zorder':3}
        if center == 'common':
            center = self.center
        if amp is not None:
            amp, (x, y) = self.readdata(amp, x, y, v)
        if ampfits is not None:
            amp, (x, y), (bmaj, bmin, bpa), _, _ \
                = self.readfits(ampfits, False, None, center, restfrq)
        if ang is not None:
            ang, (x, y) = self.readdata(ang, x, y, v)
        if angfits is not None:
            ang, (x, y), (bmaj, bmin, bpa), _, _ \
                = self.readfits(angfits, False, None, center, restfrq)
        if stU is not None:
            rmsU = estimate_rms(stU, sigma)
            stU, (x, y) = self.readdata(stU, x, y, v)
        if Ufits is not None:
            stU, (x, y), (bmaj, bmin, bpa), _, rmsU \
                = self.readfits(Ufits, False, sigma, center, restfrq)
        if stQ is not None:
            rmsU = estimate_rms(stU, sigma)
            stQ, (x, y) = self.readdata(stQ, x, y, v)
        if Qfits is not None:
            stQ, (x, y), (bmaj, bmin, bpa), _, rmsQ \
                = self.readfits(Qfits, False, sigma, center, restfrq)
        if not (stU is None or stQ is None):
            rms = (rmsU + rmsQ) / 2.
            self.rms = rms
            stU[np.abs(stU) < cutoff * rms] = np.nan
            stQ[np.abs(stQ) < cutoff * rms] = np.nan
            amp = np.hypot(stU, stQ)
            ang = np.degrees(np.arctan2(stU, stQ) / 2.)
        if amp is None or angonly: amp = np.ones_like(ang)
        x, y = x[::skip], y[::skip]
        amp = self.skipfill(amp, skip)
        ang = self.skipfill(ang, skip)
        ang += rotation
        u = ampfactor * amp * np.sin(np.radians(ang))
        v = ampfactor * amp * np.cos(np.radians(ang))
        kwargs0['scale'] = 1. / np.abs(x[1] - x[0])
        for axnow, unow, vnow in zip(self.ax, u, v):
            axnow.quiver(x, y, unow, vnow, **dict(kwargs0, **kwargs))
        if show_beam and not self.pv:
            self.add_beam(bmaj, bmin, bpa, beamcolor)
    
    def set_axis(self, xticks: list = None, yticks: list = None,
                 xticksminor: list = None, yticksminor: list = None,
                 xticklabels: list = None, yticklabels: list= None,
                 xlabel: str = None, ylabel: str = None,
                 grid: dict = None, title: dict = None,
                 samexy: bool = True, loglog: int = None) -> None:
        """Use ax.set_* of matplotlib.

        Args:
            xticks (list, optional): Defaults to None.
            yticks (list, optional): Defaults to None.
            xticksminor (list or int, optional):
                If int, int times more than xticks. Defaults to None.
            yticksminor (list ot int, optional): Defaults to None.
                If int, int times more than xticks. Defaults to None.
            xticklabels (list, optional): Defaults to None.
            yticklabels (list, optional): Defaults to None.
            xlabel (str, optional): Defaults to None.
            ylabel (str, optional): Defaults to None.
            grid (dict, optional):
                True means merely grid(). Defaults to None.
            title (dict, optional):
                str means set_title(str) for 2D or fig.suptitle(str) for 3D.
                Defaults to None.
            samexy (bool, optional):
                True supports same ticks between x and y. Defaults to True.
            loglog (float, optional):
                If a float is given, plot on a log-log plane, and
                xlim=(xmax / loglog, xmax) and so does ylim. Defaults to None.
        """
        if self.pv:
            if xlabel is None:
                xlabel = 'Offset ' + ('(arcsec)' if self.dist == 1 else '(au)')
            if ylabel is None:
                ylabel = r'Velocity (km s$^{-1})$'
            samexy = False
        else:
            if xlabel is None:
                xlabel = 'R.A. ' + '(arcsec)' if self.dist == 1 else '(au)'
            if ylabel is None:
                ylabel = 'Dec. ' + '(arcsec)' if self.dist == 1 else '(au)'
        for ch, axnow in enumerate(self.ax):
            if samexy:
                axnow.set_xticks(axnow.get_yticks())
                axnow.set_yticks(axnow.get_xticks())
            if samexy or loglog is not None:
                axnow.set_aspect(1)
            if loglog is not None:
                axnow.set_xscale('log')
                axnow.set_yscale('log')
            if xticks is None: xticks = axnow.get_xticks()
            axnow.set_xticks(xticks)
            if yticks is None: yticks = axnow.get_yticks()
            axnow.set_yticks(yticks)
            if loglog is not None:
                xticklabels = [str(t if t < 0 else int(t)) for t in xticks]
                yticklabels = [str(t if t < 0 else int(t)) for t in yticks]
            if xticksminor is not None:
                if type(xticksminor) is int:
                    t = axnow.get_xticks()
                    dt = t[1] - t[0]
                    t = np.r_[t[0] - dt, t, t[-1] + dt]
                    xticksminor = np.linspace(t[0], t[-1],
                                              xticksminor*(len(t) - 1) + 1)
                axnow.set_xticks(xticksminor, minor=True)
            if yticksminor is not None:
                if type(yticksminor) is int:
                    t = axnow.get_yticks()
                    dt = t[1] - t[0]
                    t = np.r_[t[0] - dt, t, t[-1] + dt]
                    yticksminor = np.linspace(t[0], t[-1],
                                              yticksminor*(len(t) - 1) + 1)
                axnow.set_yticks(yticksminor, minor=True)
            if xticklabels is not None:
                axnow.set_xticklabels(xticklabels)
            if yticklabels is not None:
                axnow.set_yticklabels(yticklabels)
            if xlabel is not None:
                axnow.set_xlabel(xlabel)
            if ylabel is not None:
                axnow.set_ylabel(ylabel)
            if not (ch in self.bottomleft):
                plt.setp(axnow.get_xticklabels(), visible=False)
                plt.setp(axnow.get_yticklabels(), visible=False)
                axnow.set_xlabel('')
                axnow.set_ylabel('')
            axnow.set_xlim(*self.xlim)
            axnow.set_ylim(*self.ylim)
            if loglog is not None:
                axnow.set_xlim(self.xlim[1] / loglog, self.xlim[1])
                axnow.set_ylim(self.ylim[1] / loglog, self.ylim[1])
            if grid is not None:
                axnow.grid(**({} if grid == True else grid))
            if len(self.ax) == 1:
                if self.fig is None:
                    plt.figure(0).tight_layout()
        if title is not None:
            if len(self.ax) > 1:
                if type(title) is str: title = {'t':title}
                title = dict({'y':0.9}, **title)
                for i in range(self.npages):
                    fig = plt.figure(i)
                    fig.suptitle(**title)
            else:
                if type(title) is str: title = {'label':title}
                axnow.set_title(**title)
            
    def set_axis_radec(self, xlabel: str = 'R.A. (ICRS)',
                       ylabel: str = 'Dec. (ICRS)',
                       nticksminor: int = 2,
                       grid: dict = None, title: dict = None) -> None:
        """Use ax.set_* of matplotlib.

        Args:
            xlabel (str, optional): Defaults to 'R.A. (ICRS)'.
            ylabel (str, optional): Defaults to 'Dec. (ICRS)'.
            nticksminor (int, optional):
                Interval ratio of major and minor ticks. Defaults to 2.
            grid (dict, optional):
                True means merely grid(). Defaults to None.
            title (dict, optional):
                str means set_title(str) for 2D or fig.suptitle(str) for 3D.
                Defaults to None.
        """
        if self.rmax > 50.:
            print('WARNING: set_axis_radec() is not supported '
                  + 'with rmax>50 yet.')
        ra_h, ra_m, ra_s, _, dec_d, dec_m, dec_s, _ \
            = re.split('[hdms ]', self.center)
        ra_hm = ra_h + r'$^{\rm h}$' + ra_m + r'$^{\rm m}$'
        dec_sign = np.sign((dec_d := int(dec_d)))
        dec_d = str(dec_d // dec_sign)
        dec_sign = r'$-$' if dec_sign < 0 else r'$+$'
        dec_dm = dec_sign + dec_d + r'$^{\circ}$' + dec_m + r'$^{\prime}$'
        log2r = np.log10(2. * self.rmax)
        n = np.array([-3, -2, -1, 0, 1, 2, 3])
        def makegrid(second, mode):
            second = float(second)
            if mode == 'ra':
                scale, factor, sec = 1.5, 15, r'$^{\rm s}$'
            else:
                scale, factor, sec = 0.5, 1, r'$^{\rm \prime\prime}$'
                if dec_sign == r'$-$': factor *= -1
            sec = r'.$\hspace{-0.4}$' + sec
            dorder = log2r - scale - (order := np.floor(log2r - scale))
            if 0.00 < dorder <= 0.33:
                g = 1
            elif 0.33 < dorder <= 0.68:
                g = 2
            elif 0.68 < dorder <= 1.00:
                g = 5
            g *= 10**order
            decimals = max(-int(order), -1)
            rounded = round(second, decimals)
            lastdigit = round(rounded // 10**(-decimals-1) % 100 / 10) % 10
            rounded -= lastdigit * 10**(-decimals) % g
            ticks = (n*g - second + rounded) * factor
            ticksminor = np.linspace(ticks[0], ticks[-1], 6*nticksminor + 1)
            ticklabelvalues = np.divmod(np.round((rounded + n*g) % 60, 6), 1)
            decimals = max(decimals, 0)
            ticklabels = [f'{int(i):02d}{sec}' + f'{j:.{decimals:d}f}'[2:]
                          for i, j in zip(*ticklabelvalues)]
            return ticks, ticksminor, ticklabels
        xticks, xticksminor, xticklabels = makegrid(ra_s, 'ra')
        xticklabels[3] = ra_hm + xticklabels[3]
        yticks, yticksminor, yticklabels = makegrid(dec_s, 'dec')
        yticklabels[3] = dec_dm + '\n' + yticklabels[3]
        for ch, axnow in enumerate(self.ax):
            axnow.set_aspect(1)
            axnow.set_xticks(xticks)
            axnow.set_yticks(yticks)
            axnow.set_xticks(xticksminor, minor=True)
            axnow.set_yticks(yticksminor, minor=True)
            axnow.set_xticklabels(xticklabels)
            axnow.set_yticklabels(yticklabels)
            axnow.set_xlabel(xlabel)
            axnow.set_ylabel(ylabel)
            if not (ch in self.bottomleft):
                plt.setp(axnow.get_xticklabels(), visible=False)
                plt.setp(axnow.get_yticklabels(), visible=False)
                axnow.set_xlabel('')
                axnow.set_ylabel('')
            axnow.set_xlim(*self.xlim)
            axnow.set_ylim(*self.ylim)
            if grid is not None:
                axnow.grid(**({} if grid == True else grid))
            if len(self.ax) == 1:
                if self.fig is None:
                    plt.figure(0).tight_layout()
        if title is not None:
            if len(self.ax) > 1:
                if type(title) is str: title = {'t':title}
                title = dict({'y':0.9}, **title)
                for i in range(self.npages):
                    fig = plt.figure(i)
                    fig.suptitle(**title)
            else:
                if type(title) is str: title = {'label':title}
                axnow.set_title(**title)
            
    def savefig(self, filename: str = 'plotastrodata.png',
                show: bool = False, **kwargs) -> None:
        """Use savefig of matplotlib.

        Args:
            filename (str, optional):
                Output image file name. Defaults to 'plotastrodata.png'.
            show (bool, optional):
                True means doing plt.show(). Defaults to False.
        """
        kwargs0 = {'transparent': True, 'bbox_inches': 'tight'}
        for axnow in self.ax:
            axnow.set_xlim(*self.xlim)
            axnow.set_ylim(*self.ylim)
        ext = filename.split('.')[-1]
        for i in range(self.npages):
            ver = '' if self.npages == 1 else f'_{i:d}'
            fig = plt.figure(i)
            fig.patch.set_alpha(0)
            fig.savefig(filename.replace('.' + ext, ver + '.' + ext),
                        **dict(kwargs0, **kwargs))
        if show:
            plt.show()
        plt.close()

    def get_figax(self) -> tuple:
        """Output the external fig and ax after plotting.

        Returns:
            tuple: (fig, ax)
        """
        if len(self.ax) > 1:
            print('get_figax is not supported with channel maps')
            return -1
        return self.fig, self.ax[0]


def profile(fitsimage: str = '', Tb: bool = False,
            flux: bool = False, dist: float = 1.,
            restfrq: float = None, vsys: float = 0.,
            coords: list = [], radius: float = 0,
            xmin: float = -1e10, xmax: float = 1e10,
            ymin: float = None, ymax: float = None, yfactor: float = 1.,
            title: list = None, xticks: list = None, yticks: list = None,
            xticklabels: list = None, yticklabels: list = None,
            xticksminor: list = None, yticksminor: list = None,
            xlabel: str = r'Velocity (km s$^{-1}$)', ylabel: list = None,
            text: list = None,  savefig: dict = None, show: bool = True,
            gaussfit: bool = False, width: int = 1,
            getfigax: bool = False, gauss_kwargs: dict = {},
            **kwargs) -> list:
    """Use plot of matplotlib.

    Args:
        fitsimage (str, optional): Input fits name. Defaults to ''.
        Tb (bool, optional):
            True means line profiles of brightness T. Defaults to False.
        flux (bool, optional):
            True means line profiles in the unit of Jy. Defaults to False.
        dist (float, optional):
            Change x and y in arcsec to au. Defaults to 1..
        restfrq (float, optional):
            Used for velocity and brightness T. Defaults to None.
        vsys (float, optional): x-axis is v - vsys. Defaults to 0..
        coords (list, optional):
            Text coordinates of centers to make line profiles. Defaults to [].
        radius (float, optional): 0 means nearest pixel. Defaults to 0.
        xmin (float, optional): Minimum velocity. Defaults to -1e10.
        xmax (float, optional): Maximum velocity. Defaults to 1e10.
        ymin (float, optional): Mminimum intensity etc. Defaults to None.
        ymax (float, optional): Maximum intensity etc. Defaults to None.
        yfactor (float, optional):
            Y-axis is yfactor times intensity etc. Defaults to 1..
        title (list, optional):
            List of input dictionary for ax.set_title().
            List of title strings is also acceptable. Defaults to None.
        xticks (list, optional): Defaults to None.
        yticks (list, optional): Defaults to None.
        xticklabels (list, optional): Defaults to None.
        yticklabels (list, optional): Defaults to None.
        xticksminor (list, optional): Defaults to None.
        yticksminor (list, optional): Defaults to None.
        xlabel (str, optional): Defaults to r'Velocity (km s$^{-1}$)'.
        ylabel (list, optional): Defaults to None.
        text (list, optional):
            List of input dictionary for ax.text(). Defaults to None.
        savefig (dict, optional):
            List of input dictionary for fig.savefig().
            A file name string is also acceptable. Defaults to None.
        show (bool, optional): True means plt.show(). Defaults to True.
        gaussfit (bool, optional):
            True means doing Gaussian fitting. Defaults to False.
        width (int, optional): To rebin with the width. Defaults to 1.
        getfigax (bool, optional):
            True means return (fig, ax), where ax is a list.
            Defaults to False.
        gauss_kwargs (dict, optional):
            Input dictionary for ax.plot() to show the best Gaussian.
            Defaults to {}.
            
    Returns:
        tuple: (fig, ax), where ax is a list, if getfigax=True.
               Otherwise, no return.
    """
    kwargs0 = {'drawstyle':'steps-mid', 'color':'k'}
    gauss_kwargs0 = {'drawstyle':'default', 'color':'g'}
    savefig0 = {'bbox_inches':'tight', 'transparent':True}
    if type(coords) is str: coords = [coords]
    data, (x, y, v), (bmaj, bmin, _), bunit, _ \
        = fits2data(fitsimage, Tb, False, dist, None, restfrq,
                    vsys=vsys, vmin=xmin, vmax=xmax,
                    center=coords[0])
    xlist, ylist = coord2xy(coords) * 3600.
    xlist, ylist = xlist - xlist[0], ylist - ylist[0]
    x, y = np.meshgrid(x, y)
    prof = np.empty(((nprof := len(coords)), len(v)))
    for i, (xc, yc) in enumerate(zip(xlist, ylist)):
        r = np.hypot(x - xc, y - yc)
        if radius == 0:
            idx = np.unravel_index(np.argmin(r), np.shape(r))
            prof[i] = [d[idx] for d in data]
        elif flux:
            prof[i] = [np.sum(d[r < radius]) for d in data]
        else:
            prof[i] = [np.mean(d[r < radius]) for d in data]
    newlen = len(v) // (width := int(width))
    w, q = np.zeros(newlen), np.zeros((nprof, newlen))
    for i in range(width):
        w += v[i:i + newlen*width:width]
        q += prof[:, i:i + newlen*width:width]
    v, prof = w / width, q / width
    if Tb and flux:
        flux = False
        print('WARNING: ignore flux=True because Tb=True.')
    if flux:
        Omega = np.pi * bmaj * bmin / 4. / np.log(2.)
        dxdy = np.abs((y[1, 0]-y[0, 0]) * (x[0, 1]-x[0, 0]))
        prof *= dxdy / Omega
    prof *= yfactor
    xmin, xmax = np.min(v), np.max(v)
    if ymin is None: ymin = np.nanmin(prof)
    if ymax is None: ymax = np.nanmax(prof)
    if ylabel is None:
        if Tb:
            ylabel = r'$T_b$ (K)'
        elif flux:
            ylabel = 'Flux (Jy)'
        else:
            ylabel = bunit
    if type(ylabel) is str: ylabel = [ylabel] * nprof
    if gaussfit:
        bounds = [[ymin, xmin, v[1] - v[0]], [ymax, xmax, xmax - xmin]]
    def gauss(x, p, c, w):
        return p * np.exp(-4. * np.log(2.) * ((x - c) / w)**2)
    set_rcparams(20, 'w')
    fig = plt.figure(figsize=(6, 3 * nprof))
    ax = np.empty(nprof, dtype='object')
    for i in range(nprof):
        sharex = ax[i - 1] if i > 0 else None
        ax[i] = fig.add_subplot(nprof, 1, i + 1, sharex=sharex)
        if gaussfit:
            popt, pcov = curve_fit(gauss, v, prof[i], bounds=bounds)
            print('Gauss (peak, center, FWHM):', popt)
            print('Gauss uncertainties:', np.sqrt(np.diag(pcov)))
            ax[i].plot(v, gauss(v, *popt),
                       **dict(gauss_kwargs0, **gauss_kwargs))
        ax[i].plot(v, prof[i], **dict(kwargs0, **kwargs))
        if i == nprof - 1: ax[i].set_xlabel(xlabel)
        ax[i].set_ylabel(ylabel[i])
        ax[i].set_xlim(xmin, xmax)
        ax[i].set_ylim(ymin, ymax)
        if text is not None: ax[i].text(**text[i])
        if xticks is not None: ax[i].set_xticks(xticks)
        if yticks is not None: ax[i].set_yticks(yticks)
        if xticklabels is not None: ax[i].set_xticklabels(xticklabels)
        if yticklabels is not None: ax[i].set_yticklabels(yticklabels)
        if xticksminor is not None: ax[i].set_xticks(xticksminor, minor=True)
        if yticksminor is not None: ax[i].set_yticks(yticksminor, minor=True)
        if title is not None:
            if type(title[i]) is str: title[i] = {'label':title[i]}
            ax[i].set_title(**title[i])
        ax[i].hlines([0], xmin, xmax, linestyle='dashed', color='k')
    if getfigax:
        return fig, ax
    fig.tight_layout()
    if savefig is not None:
        if type(savefig) is str: savefig = {'fname':savefig} 
        fig.savefig(**dict(savefig0, **savefig))
    if show: plt.show()
    plt.close()    
