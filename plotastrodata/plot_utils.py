import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from plotastrodata.other_utils import coord2xy, rel2abs, estimate_rms, trim, listing
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
    plt.rcParams['xtick.major.size'] = 8
    plt.rcParams['ytick.major.size'] = 8
    plt.rcParams['xtick.minor.size'] = 4
    plt.rcParams['ytick.minor.size'] = 4
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['xtick.minor.width'] = 1.5
    plt.rcParams['ytick.minor.width'] = 1.5


class PlotAstroData():
    """Make a figure from 2D/3D FITS files or 2D/3D arrays.
    
    Basic rules --- For 3D data, a 1D velocity array or a FITS file
    with a velocity axis must be given to set up channels in each page.
    len(v)=1 (default) means to make a 2D figure.
    Spatial lengths are in the unit of arcsec, or au if dist (!= 1) is given.
    Angles are in the unit of degree.
    For ellipse, line, arrow, label, and marker,
    a single input can be treated without a list, e.g., anglelist=60,
    as well as anglelist=[60].
    Each element of poslist supposes a text coordinate
    like '01h23m45.6s 01d23m45.6s' or a list of relative x and y
    like [0.2, 0.3] (0 is left or bottom, 1 is right or top).
    Parameters for original methods in matplotlib.axes.Axes can be
    used as kwargs; see the default kwargs0 for reference.
    Position-velocity diagrams (pv=True) does not yet suppot ellipse, line,
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
                 pv : bool = False) -> None:
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
        """
        if fitsimage is not None:
            fd = FitsData(fitsimage)
            _, _, v = fd.get_grid(restfrq=restfrq, vsys=vsys,
                                  vmin=vmin, vmax=vmax)
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
            return [n, i, j]
        set_rcparams(fontsize=18 if nv == 1 else 12)
        ax = np.empty(nchan, dtype='object')
        for ch in range(nchan):
            n, i, j = ch2nij(ch)
            figsize = (7, 5) if nchan == 1 else (ncols*2, max(nrows, 1.5)*2)
            fig = plt.figure(n, figsize=figsize)
            sharex = ax[nij2ch(n, i - 1, j)] if i > 0 else None
            sharey = ax[nij2ch(n, i, j - 1)] if j > 0 else None
            ax[ch] = fig.add_subplot(nrows, ncols, i*ncols + j + 1,
                                     sharex=sharex, sharey=sharey)
            if nchan > 1:
                fig.subplots_adjust(hspace=0, wspace=0, right=0.87, top=0.87)
                ax[ch].text(0.9 * rmax, 0.7 * rmax,
                            rf'${v[ch]:.{veldigit:d}f}$', color='black',
                            backgroundcolor='white', zorder=20)
        self.ax = ax
        self.xdir = xdir = -1 if xflip else 1
        self.ydir = ydir = -1 if yflip else 1
        self.xlim = xlim = [xoff - xdir*rmax, xoff + xdir*rmax]
        self.ylim = ylim = [yoff - ydir*rmax, yoff + ydir*rmax]
        vlim = [vmin, vmax]
        if pv: self.ylim = ylim = vlim
        self.rmax = rmax
        self.center = center
        self.dist = dist
        self.rowcol = nrows * ncols
        self.npages = npages
        self.allchan = np.arange(nchan)
        self.bottomleft = nij2ch(np.arange(npages), nrows - 1, 0)
        self.pv = pv

        def pos2xy(poslist: list = []) -> list:
            """Text or relative to absolute coordinates.

            Args:
                poslist (list, optional):
                    Text coordinates or relative coordinates. Defaults to [].

            Returns:
                list: absolute coordinates.
            """
            if np.shape(poslist) in [(), (2,)]:
                poslist = [poslist]
            x, y = [None] * len(poslist), [None] * len(poslist)
            for i, p in enumerate(poslist):
                if type(p) == str:
                    x[i], y[i] = (coord2xy(p)-coord2xy(center)) * 3600.
                else:
                    x[i], y[i] = rel2abs(*p, xlim, ylim)
            return [x, y]
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
                     center: str = None, restfrq: float = None) -> list:
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

        
    def add_ellipse(self, poslist: list = [],
                    majlist: list = [], minlist: list = [], palist: list = [],
                    include_chan: list = None, **kwargs) -> None:
        """Use add_patch() and Ellipse of matplotlib.

        Args:
            poslist (list, optional): text or relative. Defaults to [].
            majlist (list, optional): Ellipse major axis. Defaults to [].
            minlist (list, optional): Ellipse minor axis. Defaults to [].
            palist (list, optional):
                Ellipse position angle (north to east). Defaults to [].
            include_chan (list, optional): None means all. Defaults to None.
        """
        kwargs0 = {'facecolor':'none', 'edgecolor':'gray',
                   'linewidth':1.5, 'zorder':10}
        if include_chan is None: include_chan = self.allchan
        for x, y, width, height, angle\
            in zip(*self.pos2xy(poslist), *listing(minlist, majlist, palist)):
            for ch, axnow in enumerate(self.ax):
                if not (ch in include_chan):
                    continue
                plt.figure(ch // self.rowcol)
                e = Ellipse((x, y), width=width, height=height,
                            angle=angle * self.xdir,
                            **dict(kwargs0, **kwargs))
                axnow.add_patch(e)
                
    def add_beam(self, bmaj: float = 0, bmin: float = 0,
                 bpa: float = 0, beamcolor: str = 'gray',
                 poslist: list = None) -> None:
        """Use add_ellipse().

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
        self.add_ellipse(include_chan=self.bottomleft, poslist=poslist,
                         majlist=[bmaj], minlist=[bmin], palist=[bpa],
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
        kwargs0 = {'color':'gray', 'linewidth':1.5, 'zorder':10}
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
            x, y = self.pos2xy([barpos[0], barpos[1] * 0.9])
            axnow.text(x[0], y[0], label, color=color, size=fontsize,
                       ha='center', va='top', zorder=10)
            x, y = self.pos2xy([barpos[0], barpos[1] * 1.1])
            axnow.plot([x[0] - length/2., x[0] + length/2.], [y[0], y[0]],
                       '-', linewidth=linewidth, color=color)
    
    def add_color(self, fitsimage: str = None,
                  x: list = None, y: list = None, skip: int = 1,
                  v: list = None, c: list = None,
                  center: str = 'common', restfrq: float = None,
                  Tb: bool = False, log: bool = False,
                  cfactor: float = 1,
                  sigma: float or str = 'out', show_cbar: bool = True,
                  cblabel: str = None, cbformat: float = '%.1e',
                  cbticks: list = None, cbticklabels: list = None,
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
        c = c * cfactor
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
        for ch, (axnow, cnow) in enumerate(zip(self.ax, c)):
            p = axnow.pcolormesh(x, y, cnow, shading='nearest',
                                 **dict(kwargs0, **kwargs))
        for ch in self.bottomleft:
            if not show_cbar:
                break
            cblabel = bunit if cblabel is None else cblabel
            plt.figure(ch // self.rowcol)
            if len(self.ax) == 1:
                ax = self.ax[0]
                cb = plt.colorbar(p, ax=ax, label=cblabel, format=cbformat)
            else:
                cax = plt.axes([0.88, 0.105, 0.015, 0.77])
                cb = plt.colorbar(p, cax=cax, label=cblabel, format=cbformat)
            cb.ax.tick_params(labelsize=14)
            font = mpl.font_manager.FontProperties(size=16)
            cb.ax.yaxis.label.set_font_properties(font)
            if cbticks is not None:
                cb.set_ticks(np.log10(cbticks) if log else cbticks)
            if cbticklabels is not None:
                cb.set_ticklabels(cbticklabels)
            elif log:
                cb.set_ticks(t := cb.get_ticks())
                cb.set_ticklabels([f'{d:.1e}' for d in 10**t])
        if show_beam:
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
        x, y = x[::skip], y[::skip]
        c = self.skipfill(c, skip)
        for axnow, cnow in zip(self.ax, c):
            axnow.contour(x, y, cnow, np.array(levels) * rms,
                          **dict(kwargs0, **kwargs))
        if show_beam:
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
            ang = np.degrees(np.arctan(stU / stQ) / 2.)
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
        if show_beam:
            self.add_beam(bmaj, bmin, bpa, beamcolor)
    
    def set_axis(self, xticks: list = None, yticks: list = None,
                 xticksminor: list = None, yticksminor: list = None,
                 xticklabels: list = None, yticklabels: list= None,
                 xlabel: str = None, ylabel: str = None,
                 grid: dict = None, title: dict = None,
                 samexy: bool = True) -> None:
        """Use ax.set_* of matplotlib.

        Args:
            xticks (list, optional): Defaults to None.
            yticks (list, optional): Defaults to None.
            xticksminor (list, optional): Defaults to None.
            yticksminor (list, optional): Defaults to None.
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
        """
        if self.pv:
            if xlabel is None:
                xlabel = 'Offset ' + '(arcsec)' if self.dist == 1 else '(au)'
            if ylabel is None: ylabel = r'Velocity (km s$^{-1})$'
            samexy = False
        else:
            if xlabel is None:
                xlabel = 'R.A. ' + '(arcsec)' if self.dist == 1 else '(au)'
            if ylabel is None:
                ylabel = 'Dec. ' + '(arcsec)' if self.dist == 1 else '(au)'
        if xticklabels is None and xticks is not None:
                xticklabels = [str(t) for t in xticks]
        if yticklabels is None and xticks is not None:
                yticklabels = [str(t) for t in yticks]
        for ch, axnow in enumerate(self.ax):
            if samexy:
                axnow.set_xticks(axnow.get_yticks())
                axnow.set_yticks(axnow.get_xticks())
                axnow.set_aspect(1)
            if xticks is not None: axnow.set_xticks(xticks)
            if yticks is not None: axnow.set_yticks(yticks)
            if xticksminor is not None:
                axnow.set_xticks(xticksminor, minor=True)
            if yticksminor is not None:
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
            if grid is not None:
                axnow.grid(**({} if grid == True else grid))
            if len(self.ax) == 1: plt.figure(0).tight_layout()
        if title is not None:
            if len(self.ax) > 1:
                if type(title) == str: title = {'t':title}
                title = dict({'y':0.9}, **title)
                for i in range(self.npages):
                    fig = plt.figure(i)
                    fig.suptitle(**title)
            else:
                if type(title) == str: title = {'label':title}
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
        ra0, ra1, ra2, _, dec0, dec1, dec2, _ \
            = re.split('[hdms ]', self.center)
        ra01 = ra0 + r'$^{\rm h}$' + ra1 + r'$^{\rm m}$'
        dec01 = dec0 + r'$^{\circ}$' + dec1 + r'$^{\prime}$'
        ra2, dec2 = float(ra2), float(dec2)
        log2r = np.log10(2. * self.rmax)
        n = np.array([-3, -2, -1, 0, 1, 2, 3])
        def makegrid(second, mode):
            if mode == 'ra':
                scale, factor, sec = 1.5, 15, r'$^{\rm s}$'
            else:
                scale, factor, sec = 0.5, 1, r'$^{\rm \prime\prime}$'
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
            lastdigit = round(rounded // 10**(-decimals-1) % 100 / 10)
            rounded -= lastdigit * 10**(-decimals) % g
            ticks = n*g*factor + second - rounded
            ticksminor = np.linspace(ticks[0], ticks[-1], 6*nticksminor + 1)
            tlint, tldec = np.divmod((rounded + n*g) % 60., 1)
            decimals = max(decimals, 0)
            ticklabels = [f'{i:.0f}.' + r'$\hspace{-0.4}$' + sec
                          + f'{j:.{decimals:d}f}'[2:]
                          for i, j in zip(tlint, tldec)]
            return [ticks, ticksminor, ticklabels]
        xticks, xticksminor, xticklabels = makegrid(ra2, 'ra')
        xticklabels[3] = ra01 + xticklabels[3]
        yticks, yticksminor, yticklabels = makegrid(dec2, 'dec')
        yticklabels[3] = dec01 + '\n' + yticklabels[3]
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
            if len(self.ax) == 1: plt.figure(0).tight_layout()
        if title is not None:
            if len(self.ax) > 1:
                if type(title) == str: title = {'t':title}
                title = dict({'y':0.9}, **title)
                for i in range(self.npages):
                    fig = plt.figure(i)
                    fig.suptitle(**title)
            else:
                if type(title) == str: title = {'label':title}
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
            ver = '' if len(self.ax) == 1 else f'_{i:d}'
            fig = plt.figure(i)
            fig.patch.set_alpha(0)
            fig.savefig(filename.replace('.' + ext, ver + '.' + ext),
                        **dict(kwargs0, **kwargs))
        if show:
            for axnow in self.ax:
                axnow.set_xlim(*self.xlim)
                axnow.set_ylim(*self.ylim)
            plt.show()
        plt.close()
