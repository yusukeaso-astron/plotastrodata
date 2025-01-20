import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from dataclasses import dataclass

from plotastrodata.other_utils import coord2xy, xy2coord, listing, estimate_rms
from plotastrodata.analysis_utils import AstroData, AstroFrame


plt.ioff()  # force to turn off interactive mode


def set_rcparams(fontsize: int = 18, nancolor: str = 'w',
                 dpi: int = 256) -> None:
    """Nice rcParams for figures.

    Args:
        fontsize (int, optional): plt.rcParams['font.size']. Defaults to 18.
        nancolor (str, optional): plt.rcParams['axes.facecolor']. Defaults to 'w'.
        dpi (int, optional): plt.rcParams['savefig.dpi']. Defaults to 256.
    """
    # plt.rcParams['font.family'] = 'arial'
    plt.rcParams['axes.facecolor'] = nancolor
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['savefig.dpi'] = dpi
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


def logticks(ticks: list[float], lim: list[float, float]
             ) -> tuple[list[float], list[str]]:
    """Make nice ticks for a log axis.

    Args:
        ticks (list): List of ticks.
        lim (list): [min, max].

    Returns:
        tuple: (new ticks, new labels).
    """
    order = int(np.floor((np.log10(lim[0]))))
    a = (lim[0] // 10**order + 1) * 10**order
    a = np.round(a, max(-order, 0))
    order = int(np.floor((np.log10(lim[1]))))
    b = (lim[1] // 10**order) * 10**order
    b = np.round(b, max(-order, 0))
    newticks = np.sort(np.r_[a, ticks, b])
    newlabels = [str(t if t < 1 else int(t)) for t in newticks]
    return newticks, newlabels


@dataclass
class PlotAxes2D():
    """Use Axes.set_* to adjust x and y axes.

    Args:
        samexy (bool, optional): True supports same ticks between x and y. Defaults to True.
        loglog (float, optional): If a float is given, plot on a log-log plane, and xim=(xmax / loglog, xmax) and so does ylim. Defaults to None.
        xscale (str, optional): Defaults to None.
        yscale (str, optional): Defaults to None.
        xlim (list, optional): Defaults to None.
        ylim (list, optional): Defaults to None.
        xlabel (str, optional): Defaults to None.
        ylabel (str, optional): Defaults to None.
        xticks (list, optional): Defaults to None.
        yticks (list, optional): Defaults to None.
        xticklabels (list, optional): Defaults to None.
        yticklabels (list, optional): Defaults to None.
        xticksminor (list or int, optional): If int, int times more than xticks. Defaults to None.
        yticksminor (list ot int, optional): Defaults to None. If int, int times more than xticks. Defaults to None.
        grid (dict, optional): True means merely grid(). Defaults to None.
        aspect (float, optional): Defaults to None.
    """
    samexy: bool = True
    loglog: bool | None = None
    xscale: str = 'linear'
    yscale: str = 'linear'
    xlim: list | None = None
    ylim: list | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    xticks: list | None = None
    yticks: list | None = None
    xticklabels: list | None = None
    yticklabels: list | None = None
    xticksminor: list | int = None
    yticksminor: list | int = None
    grid: dict | None = None
    aspect: float | None = None

    def set_xyaxes(self, ax):
        if self.loglog is not None:
            self.xscale = 'log'
            self.yscale = 'log'
            self.samexy = True
            if self.xlim is not None:
                self.xlim[0] = self.xlim[1] / self.loglog
            if self.ylim is not None:
                self.ylim[0] = self.ylim[1] / self.loglog
        ax.set_xscale(self.xscale)
        ax.set_yscale(self.yscale)
        if self.samexy:
            ax.set_xticks(ax.get_yticks())
            ax.set_yticks(ax.get_xticks())
            ax.set_aspect(1)
        if self.xticks is None:
            self.xticks = ax.get_xticks()
        if self.yticks is None:
            self.yticks = ax.get_yticks()
        if self.xscale == 'log':
            self.xticks, self.xticklabels = logticks(self.xticks, self.xlim)
        if self.yscale == 'log':
            self.yticks, self.yticklabels = logticks(self.yticks, self.ylim)
        ax.set_xticks(self.xticks)
        ax.set_yticks(self.yticks)
        if self.xticksminor is not None:
            if type(self.xticksminor) is int:
                t = ax.get_xticks()
                dt = t[1] - t[0]
                t = np.r_[t[0] - dt, t, t[-1] + dt]
                num = self.xticksminor * (len(t) - 1) + 1
                self.xticksminor = np.linspace(t[0], t[-1], num)
            ax.set_xticks(self.xticksminor, minor=True)
        if self.yticksminor is not None:
            if type(self.yticksminor) is int:
                t = ax.get_yticks()
                dt = t[1] - t[0]
                t = np.r_[t[0] - dt, t, t[-1] + dt]
                num = self.yticksminor * (len(t) - 1) + 1
                self.yticksminor = np.linspace(t[0], t[-1], num)
            ax.set_yticks(self.yticksminor, minor=True)
        if self.xticklabels is not None:
            ax.set_xticklabels(self.xticklabels)
        if self.yticklabels is not None:
            ax.set_yticklabels(self.yticklabels)
        if self.xlabel is not None:
            ax.set_xlabel(self.xlabel)
        if self.ylabel is not None:
            ax.set_ylabel(self.ylabel)
        if self.xlim is not None:
            ax.set_xlim(*self.xlim)
        if self.ylim is not None:
            ax.set_ylim(*self.ylim)
        if self.grid is not None:
            ax.grid(**({} if self.grid is True else self.grid))
        if self.aspect is not None:
            ax.set_aspect(self.aspect)


def set_minmax(data: np.ndarray, stretch: str, stretchscale: float,
               stretchpower: float,
               rms: float, kw: dict) -> np.ndarray:
    """Set vmin and vmax for color pcolormesh and RGB maps.

    Args:
        data (np.ndarray): Plotted data.
        stretch (str): 'log', 'asinh', 'power'. Any other means linear.
        stretchscale (float): For the arcsinh strech.
        stretchpower (float): For the power strech.
        rms (float): RMS noise level.
        kw (dict): Probably like {'vmin':0, 'vmax':1}.

    Returns:
        np.ndarray: Data clipped with the vmin and vmax.
    """
    if type(stretch) is str:
        data = [data]
        rms = [rms]
        stretch = [stretch]
        stretchscale = [stretchscale]
        if 'vmin' in kw:
            kw['vmin'] = [kw['vmin']]
        if 'vmax' in kw:
            kw['vmax'] = [kw['vmax']]
    z = (data, stretch, stretchscale, rms)
    for i, (c, st, stsc, r) in enumerate(zip(*z)):
        if stsc is None:
            stsc = r
        if st == 'log':
            if np.any(c > 0):
                c = np.log10(c.clip(np.nanmin(c[c > 0]), None))
        elif st == 'asinh':
            c = np.arcsinh(c / stsc)
        elif st == 'power':
            cmin = kw['min'][i] if 'vmin' in kw else r
            c = c.clip(cmin, None)
            c = ((c / cmin)**(1 - stretchpower) - 1) \
                / (1 - stretchpower) / np.log(10)
        data[i] = c
    n = len(data)
    for m in ['vmin', 'vmax']:
        if m in kw:
            for i, (c, st, stsc, _) in enumerate(zip(*z)):
                if st == 'log':
                    kw[m][i] = np.log10(kw[m][i])
                elif st == 'asinh':
                    kw[m][i] = np.arcsinh(kw[m][i] / stsc)
                elif st == 'power':
                    kw[m][i] = ((kw[m][i]/c.min())**(1 - stretchpower) - 1) \
                               / (1 - stretchpower) / np.log(10)
        else:
            kw[m] = [None] * n
            for i, (c, st, _, r) in enumerate(zip(*z)):
                if m == 'vmin':

                    kw[m][i] = np.log10(r) if st == 'log' else np.nanmin(c)
                else:
                    kw[m][i] = np.nanmax(c)
    data = [c.clip(a, b) for c, a, b in zip(data, kw['vmin'], kw['vmax'])]
    if n == 1:
        data = data[0]
        kw['vmin'] = kw['vmin'][0]
        kw['vmax'] = kw['vmax'][0]
    return data


def kwargs2AstroData(kw: dict) -> AstroData:
    """Get AstroData and remove its arguments from kwargs.

    Args:
        kw (dict): Parameters to make AstroData.

    Returns:
        AstroData: AstroData made from the parameters in kwargs.
    """
    tmp = {}
    d = AstroData(data=np.zeros((2, 2)))
    for k in vars(d):
        if k in kw:
            tmp[k] = kw[k]
            del kw[k]
    if tmp == {}:
        print('No argument given.')
        return None
    else:
        d = AstroData(**tmp)
        return d


def kwargs2AstroFrame(kw: dict) -> AstroFrame:
    """Get AstroFrame from kwargs.

    Args:
        kw (dict): Parameters to make AstroFrame.

    Returns:
        AstroFrame: AstroFrame made from the parameters in kwargs.
    """
    tmp = {}
    f = AstroFrame()
    for k in vars(f):
        if k in kw:
            tmp[k] = kw[k]
    f = AstroFrame(**tmp)
    return f


def kwargs2PlotAxes2D(kw: dict) -> PlotAxes2D:
    """Get PlotAxes2D and remove its arguments from kwargs.

    Args:
        kw (dict): Parameters to make PlotAxes2D.

    Returns:
        PlotAxes2D: PlotAxes2D made from the parameters in kwargs.
    """
    tmp = {}
    d = PlotAxes2D()
    for k in vars(d):
        if k in kw:
            tmp[k] = kw[k]
            del kw[k]
    d = PlotAxes2D(**tmp)
    return d


class PlotAstroData(AstroFrame):
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
    used as kwargs; see the default _kw for reference.
    Position-velocity diagrams (pv=True) does not yet suppot region, line,
    arrow, and segment because the units of abscissa and ordinate
    are different.

    kwargs is the arguments of AstroFrame to define plotting ranges.

    Args:
        v (np.ndarray, optional): Used to set up channels if fitsimage not given. Defaults to [0].
        vskip (int, optional): How many channels are skipped. Defaults to 1.
        veldigit (int, optional): How many digits after the decimal point. Defaults to 2.
        restfreq (float, optional): Used for velocity and brightness T. Defaults to None.
        channelnumber (int, optional): Specify a channel number to make 2D maps. Defaults to None.
        nrows (int, optional): Used for channel maps. Defaults to 4.
        ncols (int, optional): Used for channel maps. Defaults to 6.
        fontsize (int, optional): rc_Params['font.size']. None means 18 (2D) or 12 (3D). Defaults to None.
        nancolor (str, optional): Color for masked regions. Defaults to white.
        dpi (int, optional): Dot per inch for plotting an image. Defaults to 256.
        figsize (tuple, optional): Defaults to None.
        fig (optional): External plt.figure(). Defaults to None.
        ax (optional): External fig.add_subplot(). Defaults to None.
    """
    def __init__(self, v: np.ndarray = np.array([0]), vskip: int = 1,
                 veldigit: int = 2, restfreq: float | None = None,
                 channelnumber: int | None = None, nrows: int = 4, ncols: int = 6,
                 fontsize: int | None = None, nancolor: str = 'w', dpi: int = 256,
                 figsize: tuple[float, float] | None = None,
                 fig: object | None = None, ax: object | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        internalfig = fig is None
        internalax = ax is None
        if type(channelnumber) is int:
            nrows = ncols = 1
        if self.fitsimage is not None:
            self.read(d := AstroData(fitsimage=self.fitsimage,
                                     restfreq=restfreq, sigma=None))
            v = d.v
        if len(v) > 1:
            dv = v[1] - v[0]
            k0 = int(round((self.vmin - v[0]) / dv))
            if k0 < 0:
                vpre = v[0] - (1 + np.arange(-k0)[::-1]) * dv
                v = np.append(vpre, v)
            else:
                v = v[k0:]
            k1 = len(v) + int(round((self.vmax - v[-1]) / dv))
            if k1 > len(v):
                vpost = v[-1] + (1 + np.arange(k1 - len(v))) * dv
                v = np.append(v, vpost)
            else:
                v = v[:k1]
        if self.pv or v is None or len(v) == 1:
            nv = nrows = ncols = npages = nchan = 1
        else:
            nv = len(v := v[::vskip])
            npages = int(np.ceil(nv / nrows / ncols))
            nchan = npages * nrows * ncols
            v = np.r_[v, v[-1] + (np.arange(nchan-nv)+1) * (v[1] - v[0])]
            if type(channelnumber) is int:
                nchan = npages = 1

        def nij2ch(n: int, i: int, j: int):
            return n*nrows*ncols + i*ncols + j

        def ch2nij(ch: int) -> tuple:
            n = ch // (nrows*ncols)
            i = (ch - n*nrows*ncols) // ncols
            j = ch % ncols
            return n, i, j

        if fontsize is None:
            fontsize = 18 if nchan == 1 else 12
        set_rcparams(fontsize=fontsize, nancolor=nancolor, dpi=dpi)
        ax = np.empty(nchan, dtype='object') if internalax else [ax]
        for ch in range(nchan):
            n, i, j = ch2nij(ch)
            if figsize is None:
                sqrt_a = (self.ymax - self.ymin) / (self.xmax - self.xmin)
                sqrt_a = np.sqrt(np.abs(sqrt_a))
                if nchan == 1:
                    figsize = (7 / sqrt_a, 5 * sqrt_a)
                else:
                    figsize = (ncols * 2 / sqrt_a, max(nrows*2, 3) * sqrt_a)
            if internalfig:
                fig = plt.figure(n, figsize=figsize)
            sharex = ax[nij2ch(n, i - 1, j)] if i > 0 else None
            sharey = ax[nij2ch(n, i, j - 1)] if j > 0 else None
            if internalax:
                ax[ch] = fig.add_subplot(nrows, ncols, i*ncols + j + 1,
                                         sharex=sharex, sharey=sharey)
            if nchan > 1 or type(channelnumber) is int:
                fig.subplots_adjust(hspace=0, wspace=0, right=0.87, top=0.87)
                vellabel = v[ch] if channelnumber is None else v[channelnumber]
                ax[ch].text(0.9 * self.rmax, 0.7 * self.rmax,
                            rf'${vellabel:.{veldigit:d}f}$', color='black',
                            backgroundcolor='white', zorder=20)
        self.fig = None if internalfig else fig
        self.ax = ax
        self.rowcol = nrows * ncols
        self.npages = npages
        self.allchan = np.arange(nchan if channelnumber is None else nv)
        self.bottomleft = nij2ch(np.arange(npages), nrows - 1, 0)
        self.channelnumber = channelnumber

        def vskipfill(c: np.ndarray, v_in: np.ndarray = None) -> np.ndarray:
            """Skip and fill channels with nan.

            Args:
                c (np.ndarray): 2D or 3D arrays.
                v_in (np.ndarray): 1D array.

            Returns:
                np.ndarray: 3D arrays skipped and filled with nan.
            """
            if np.ndim(c) == 3:
                if v_in is not None:
                    if (k0 := np.argmin(np.abs(v - v_in[0]))) > 0:
                        prenan = np.full((k0, *np.shape(c)[1:]), np.nan)
                        d = np.append(prenan, c, axis=0)
                    else:
                        d = c
                d = d[::vskip]
            else:
                d = np.full((nv, *np.shape(c)), c)
            n = nchan if channelnumber is None else nv
            shape = (n - len(d), len(d[0]), len(d[0, 0]))
            dnan = np.full(shape, d[0] * np.nan)
            return np.concatenate((d, dnan), axis=0)
        self.vskipfill = vskipfill

    def add_region(self, patch: str = 'ellipse',
                   poslist: list[str | list[float, float]] = [],
                   majlist: list[float] = [], minlist: list[float] = [],
                   palist: list[float] = [],
                   include_chan: list[int] | None = None,
                   **kwargs) -> None:
        """Use add_patch() and Rectangle or Ellipse of matplotlib.

        Args:
            patch (str, optional): 'ellipse' or 'rectangle'. Defaults to 'ellipse'.
            poslist (list, optional): Text or relative center. Defaults to [].
            majlist (list, optional): Ellipse major axis. Defaults to [].
            minlist (list, optional): Ellipse minor axis. Defaults to [].
            palist (list, optional): Position angle (north to east). Defaults to [].
            include_chan (list, optional): None means all. Defaults to None.
        """
        _kw = {'facecolor': 'none', 'edgecolor': 'gray',
               'linewidth': 1.5, 'zorder': 10}
        _kw.update(kwargs)
        if include_chan is None:
            include_chan = self.allchan
        if not (patch in ['rectangle', 'ellipse']):
            print('Only patch=\'rectangle\' or \'ellipse\' supported. ')
            return -1
        for x, y, width, height, angle in zip(*self.pos2xy(poslist),
                                              *listing(minlist, majlist, palist)):
            for ch, axnow in enumerate(self.ax):
                if type(self.channelnumber) is int:
                    ch = self.channelnumber
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
                      angle=angle * self.xdir, **_kw)
                axnow.add_patch(p)

    def add_beam(self,
                 beam: list[float | None, float | None, float | None] = [None, None, None],
                 beamcolor: str = 'gray',
                 poslist: list[str | list[float, float]] | None = None) -> None:
        """Use add_region().

        Args:
            beam (list, optional): [bmaj, bmin, bpa]. Defaults to [None, None, None].
            beamcolor (str, optional): matplotlib color. Defaults to 'gray'.
            poslist (list, optional): text or relative. Defaults to None.
        """
        if None in beam:
            print('No beam to plot.')
            return False

        if poslist is None:
            poslist = [max(0.35 * beam[0] / self.rmax, 0.1)] * 2
        include_chan = self.bottomleft if self.channelnumber is None else self.allchan
        self.add_region('ellipse', poslist, *beam,
                        include_chan=include_chan,
                        facecolor=beamcolor, edgecolor=None)

    def add_marker(self, poslist: list[str | list[float, float]] = [],
                   include_chan: list[int] | None = None, **kwargs) -> None:
        """Use Axes.plot of matplotlib.

        Args:
            poslist (list, optional): Text or relative. Defaults to [].
            include_chan (list, optional): None means all. Defaults to None.
        """
        _kw = {'marker': '+', 'ms': 10, 'mfc': 'gray',
               'mec': 'gray', 'mew': 2, 'alpha': 1, 'zorder': 10}
        _kw.update(kwargs)
        if include_chan is None:
            include_chan = self.allchan
        for ch, axnow in enumerate(self.ax):
            if type(self.channelnumber) is int:
                ch = self.channelnumber
            if not (ch in include_chan):
                continue
            for x, y in zip(*self.pos2xy(poslist)):
                axnow.plot(x, y, **_kw)

    def add_text(self, poslist: list[str | list[float, float]] = [],
                 slist: list[str] = [],
                 include_chan: list[int] | None = None, **kwargs) -> None:
        """Use Axes.text of matplotlib.

        Args:
            poslist (list, optional): Text or relative. Defaults to [].
            slist (list, optional): List of text. Defaults to [].
            include_chan (list, optional): None means all. Defaults to None.
        """
        _kw = {'color': 'gray', 'fontsize': 15, 'ha': 'center',
               'va': 'center', 'zorder': 10}
        _kw.update(kwargs)
        if include_chan is None:
            include_chan = self.allchan
        for ch, axnow in enumerate(self.ax):
            if type(self.channelnumber) is int:
                ch = self.channelnumber
            if not (ch in include_chan):
                continue
            for x, y, s in zip(*self.pos2xy(poslist), listing(slist)):
                axnow.text(x=x, y=y, s=s, **_kw)

    def add_line(self, poslist: list[str | list[float, float]] = [],
                 anglelist: list[float] = [],
                 rlist: list[float] = [], include_chan: list[int] | None = None,
                 **kwargs) -> None:
        """Use Axes.plot of matplotlib.

        Args:
            poslist (list, optional): Text or relative. Defaults to [].
            anglelist (list, optional): North to east. Defaults to [].
            rlist (list, optional): List of radius. Defaults to [].
            include_chan (list, optional): None means all. Defaults to None.
        """
        _kw = {'color': 'gray', 'linewidth': 1.5,
               'linestyle': '-', 'zorder': 10}
        _kw.update(kwargs)
        if include_chan is None:
            include_chan = self.allchan
        for ch, axnow in enumerate(self.ax):
            if type(self.channelnumber) is int:
                ch = self.channelnumber
            if not (ch in include_chan):
                continue
            alist = np.radians(anglelist)
            for x, y, a, r in zip(*self.pos2xy(poslist), *listing(alist, rlist)):
                axnow.plot([x, x + r * np.sin(a)],
                           [y, y + r * np.cos(a)], **_kw)

    def add_arrow(self, poslist: list[str | list[float, float]] = [],
                  anglelist: list[float] = [],
                  rlist: list[float] = [], include_chan: list[int] | None = None,
                  **kwargs) -> None:
        """Use Axes.quiver of matplotlib.

        Args:
            poslist (list, optional): Text or relative. Defaults to [].
            anglelist (list, optional): North to east. Defaults to [].
            rlist (list, optional): List of radius. Defaults to [].
            include_chan (list, optional): None means all. Defaults to None.
        """
        _kw = {'color': 'gray', 'width': 0.012,
               'headwidth': 5, 'headlength': 5, 'zorder': 10}
        _kw.update(kwargs)
        if include_chan is None:
            include_chan = self.allchan
        for ch, axnow in enumerate(self.ax):
            if type(self.channelnumber) is int:
                ch = self.channelnumber
            if not (ch in include_chan):
                continue
            alist = np.radians(anglelist)
            for x, y, a, r in zip(*self.pos2xy(poslist), *listing(alist, rlist)):
                axnow.quiver(x, y, r * np.sin(a), r * np.cos(a),
                             angles='xy', scale_units='xy', scale=1,
                             **_kw)

    def add_scalebar(self, length: float = 0, label: str = '',
                     color: str = 'gray', barpos: tuple[float, float] = (0.8, 0.12),
                     fontsize: float = None, linewidth: float = 3,
                     bbox: dict = {'alpha': 0}) -> None:
        """Use Axes.text and Axes.plot of matplotlib.

        Args:
            length (float, optional): In the unit of arcsec. Defaults to 0.
            label (str, optional): Text like '100 au'. Defaults to ''.
            color (str, optional): Same for bar and label. Defaults to 'gray'.
            barpos (tuple, optional): Relative position. Defaults to (0.8, 0.12).
            fontsize (float, optional): None means 15 if one channel else 20. Defaults to None.
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
            x, y = self.pos2xy([barpos[0], barpos[1] - 0.012])
            axnow.text(x[0], y[0], label, color=color, size=fontsize,
                       ha='center', va='top', bbox=bbox, zorder=10)
            x, y = self.pos2xy([barpos[0], barpos[1] + 0.012])
            axnow.plot([x[0] - length/2., x[0] + length/2.], [y[0], y[0]],
                       '-', linewidth=linewidth, color=color)

    def add_color(self, xskip: int = 1, yskip: int = 1,
                  stretch: str = 'linear',
                  stretchscale: float | None = None,
                  stretchpower: float = 0,
                  show_cbar: bool = True, cblabel: str | None = None,
                  cbformat: float = '%.1e', cbticks: list[float] | None = None,
                  cbticklabels: list[str] | None = None, cblocation: str = 'right',
                  show_beam: bool = True, beamcolor: str = 'gray',
                  **kwargs) -> None:
        """Use Axes.pcolormesh of matplotlib. kwargs must include the arguments of AstroData to specify the data to be plotted.

        Args:
            xskip, yskip (int, optional): Spatial pixel skip. Defaults to 1.
            stretch (str, optional): 'log' means the mapped data are logarithmic. 'asinh' means the mapped data are arc sin hyperbolic. 'power' means the mapped data are power-law (see also stretchpower). Defaults to 'linear'.
            stretchscale (float, optional): color scale is asinh(data / stretchscale). Defaults to None.
            stretchpower (float, optional): color scale is ((data / vmin)**(1 - stretchpower) - 1) / (1 - stretchpower) / ln(10). 0 means the linear scale. 1 means the logarithmic scale. Defaults to 0.
            show_cbar (bool, optional): Show color bar. Defaults to True.
            cblabel (str, optional): Colorbar label. Defaults to None.
            cbformat (float, optional): Format for ticklabels of colorbar. Defaults to '%.1e'.
            cbticks (list, optional): Ticks of colorbar. Defaults to None.
            cbticklabels (list, optional): Ticklabels of colorbar. Defaults to None.
            cblocation (str, optional): 'left', 'top', 'left', 'right'. Only for 2D images. Defaults to 'right'.
            show_beam (bool, optional): Defaults to True.
            beamcolor (str, optional): Matplotlib color. Defaults to 'gray'.
        """
        _kw = {'cmap': 'cubehelix', 'alpha': 1, 'edgecolors': 'none', 'zorder': 1}
        _kw.update(kwargs)
        d = kwargs2AstroData(_kw)
        self.read(d, xskip, yskip)
        c, x, y, v, beam, sigma = d.data, d.x, d.y, d.v, d.beam, d.sigma
        bunit = d.bunit
        self.beam = beam
        self.sigma = sigma
        if stretchscale is None:
            stretchscale = sigma
        cmin_org = _kw['vmin'] if 'vmin' in _kw else sigma
        c = set_minmax(c, stretch, stretchscale, stretchpower, sigma, _kw)
        c = self.vskipfill(c, v)
        if type(self.channelnumber) is int:
            c = [c[self.channelnumber]]
        for axnow, cnow in zip(self.ax, c):
            p = axnow.pcolormesh(x, y, cnow, **_kw)
        for ch in self.bottomleft:
            if not show_cbar:
                break
            if cblabel is None:
                cblabel = bunit
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
                if stretch == 'log':
                    cbticks = np.log10(cbticks)
                elif stretch == 'asinh':
                    cbticks = np.arcsinh(np.array(cbticks) / stretchscale)
                elif stretch == 'power':
                    cbticks = (np.array(cbticks) / cmin_org)**(1 - stretchpower)
                    cbticks = (cbticks - 1) / (1 - stretchpower) / np.log(10)
                cb.set_ticks(cbticks)
            if cbticklabels is not None:
                cb.set_ticklabels(cbticklabels)
            elif stretch in ['log', 'asinh', 'power']:
                t = cb.get_ticks()
                t = t[(_kw['vmin'] < t) * (t < _kw['vmax'])]
                cb.set_ticks(t)
                if stretch == 'log':
                    ticklin = 10**t
                elif stretch == 'asinh':
                    ticklin = np.sinh(t) * stretchscale
                elif stretch == 'power':
                    ticklin = 1 + (1 - stretchpower) * np.log(10) * t
                    ticklin = cmin_org * ticklin**(1 / (1 - stretchpower))
                cb.set_ticklabels([f'{d:{cbformat[1:]}}' for d in ticklin])
        if show_beam and not self.pv:
            self.add_beam(beam, beamcolor)

    def add_contour(self, xskip: int = 1, yskip: int = 1,
                    levels: list[float] = [-12, -6, -3, 3, 6, 12, 24, 48, 96, 192, 384],
                    show_beam: bool = True, beamcolor: str = 'gray',
                    **kwargs) -> None:
        """Use Axes.contour of matplotlib. kwargs must include the arguments of AstroData to specify the data to be plotted.

        Args:
            xskip, yskip (int, optional): Spatial pixel skip. Defaults to 1.
            levels (list, optional): Contour levels in the unit of sigma. Defaults to [-12,-6,-3,3,6,12,24,48,96,192,384].
            show_beam (bool, optional): Defaults to True.
            beamcolor (str, optional): Matplotlib color. Defaults to 'gray'.
        """
        _kw = {'colors': 'gray', 'linewidths': 1.0, 'zorder': 2}
        _kw.update(kwargs)
        d = kwargs2AstroData(_kw)
        self.read(d, xskip, yskip)
        c, x, y, v, beam, sigma = d.data, d.x, d.y, d.v, d.beam, d.sigma
        self.beam = beam
        self.sigma = sigma
        c = self.vskipfill(c, v)
        if type(self.channelnumber) is int:
            c = [c[self.channelnumber]]
        for axnow, cnow in zip(self.ax, c):
            axnow.contour(x, y, cnow, np.sort(levels) * sigma, **_kw)
        if show_beam and not self.pv:
            self.add_beam(beam, beamcolor)

    def add_segment(self, ampfits: str = None, angfits: str = None,
                    Ufits: str = None, Qfits: str = None,
                    xskip: int = 1, yskip: int = 1,
                    amp: list[np.ndarray] | None = None,
                    ang: list[np.ndarray] | None = None,
                    stU: list[np.ndarray] | None = None,
                    stQ: list[np.ndarray] | None = None,
                    ampfactor: float = 1., angonly: bool = False,
                    rotation: float = 0.,
                    cutoff: float = 3.,
                    show_beam: bool = True, beamcolor: str = 'gray',
                    **kwargs) -> None:
        """Use Axes.quiver of matplotlib. kwargs must include the arguments of AstroData to specify the data to be plotted. fitsimage = [ampfits, angfits, Ufits, Qfits]. data = [amp, ang, stU, stQ].

        Args:
            ampfits (str, optional): In put fits name. Length of segment. Defaults to None.
            angfits (str, optional): In put fits name. North to east. Defaults to None.
            Ufits (str, optional): In put fits name. Stokes U. Defaults to None.
            Qfits (str, optional): In put fits name. Stokes Q. Defaults to None.
            xskip, yskip (int, optional): Spatial pixel skip. Defaults to 1.
            amp (list, optional): Length of segment. Defaults to None.
            ang (list, optional): North to east. Defaults to None.
            stU (list, optional): Stokes U. Defaults to None.
            stQ (list, optional): Stokes Q. Defaults to None.
            ampfactor (float, optional): Length of segment is amp times ampfactor. Defaults to 1..
            angonly (bool, optional): True means amp=1 for all. Defaults to False.
            rotation (float, optional): Segment angle is ang + rotation. Defaults to 0..
            cutoff (float, optional): Used when amp and ang are calculated from Stokes U and Q. In the unit of sigma. Defaults to 3..
            show_beam (bool, optional): Defaults to True.
            beamcolor (str, optional): Matplotlib color. Defaults to 'gray'.
        """
        _kw = {'angles': 'xy', 'scale_units': 'xy', 'color': 'gray',
               'pivot': 'mid', 'headwidth': 0, 'headlength': 0,
               'headaxislength': 0, 'width': 0.007, 'zorder': 3}
        _kw.update(kwargs)
        _kw['data'] = [amp, ang, stU, stQ]
        _kw['fitsimage'] = [ampfits, angfits, Ufits, Qfits]
        d = kwargs2AstroData(_kw)
        self.read(d, xskip, yskip)
        c, x, y, v, beam, sigma = d.data, d.x, d.y, d.v, d.beam, d.sigma
        amp, ang, stU, stQ = c
        sigmaU, sigmaQ = sigma[2:]
        self.beam = beam
        beam = [beam[i] for i in range(4) if beam[i][0] is not None][0]
        if stU is not None and stQ is not None:
            self.sigma = sigma = (sigmaU + sigmaQ) / 2.
            ang = np.degrees(np.arctan2(stU, stQ) / 2.)
            amp = np.hypot(stU, stQ)
            amp[amp < cutoff * sigma] = np.nan
        if amp is None:
            amp = np.ones_like(ang)
        if angonly:
            amp = np.sign(amp)**2
        amp = amp / np.nanmax(amp)
        U = ampfactor * amp * np.sin(np.radians(ang + rotation))
        V = ampfactor * amp * np.cos(np.radians(ang + rotation))
        U = self.vskipfill(U, v)
        V = self.vskipfill(V, v)
        if type(self.channelnumber) is int:
            U = [U[self.channelnumber]]
            V = [V[self.channelnumber]]
        _kw['scale'] = 1 if len(x) == 1 else 1. / np.abs(x[1] - x[0])
        for axnow, unow, vnow in zip(self.ax, U, V):
            axnow.quiver(x, y, unow, vnow, **_kw)
        if show_beam and not self.pv:
            self.add_beam(beam, beamcolor)

    def add_rgb(self, xskip: int = 1, yskip: int = 1,
                stretch: list[str, str, str] = ['linear'] * 3,
                stretchscale: list[float | None, float | None, float | None] = [None] * 3,
                stretchpower: float = 0,
                show_beam: bool = True,
                beamcolor: list[str, str, str] = ['red', 'green', 'blue'],
                **kwargs) -> None:
        """Use PIL.Image and imshow of matplotlib. kwargs must include the arguments of AstroData to specify the data to be plotted. A three-element array ([red, green, blue]) is supposed for all arguments, except for xskip, yskip and show_beam, including vmax and vmin.

        Args:
            xskip, yskip (int, optional): Spatial pixel skip. Defaults to 1.
            stretch (str, optional): 'log' means the mapped data are logarithmic. 'asinh' means the mapped data are arc sin hyperbolic. 'power' means the mapped data are power-law (see also stretchpower). Defaults to 'linear'.
            stretchscale (float, optional): color scale is asinh(data / stretchscale). Defaults to None.
            stretchpower (float, optional): color scale is ((data / vmin)**(1 - stretchpower) - 1) / (1 - stretchpower) / ln(10). 0 means the linear scale. 1 means the logarithmic scale. Defaults to 0.
            show_beam (bool, optional): Defaults to True.
            beamcolor (str, optional): Matplotlib color. Defaults to 'gray'.
        """
        from PIL import Image

        _kw = {}
        _kw.update(kwargs)
        d = kwargs2AstroData(_kw)
        self.read(d, xskip, yskip)
        c, x, y, v, beam, sigma = d.data, d.x, d.y, d.v, d.beam, d.sigma
        self.beam = beam
        self.sigma = sigma
        for i in range(len(stretchscale)):
            if stretchscale[i] is None:
                stretchscale[i] = sigma[i]
        c = set_minmax(c, stretch, stretchscale, stretchpower, sigma, _kw)
        if not (np.shape(c[0]) == np.shape(c[1]) == np.shape(c[2])):
            print('RGB shapes mismatch. Skip add_rgb.')
            return -1

        for i in range(3):
            c[i] = (c[i] - _kw['vmin'][i]) \
                   / (_kw['vmax'][i] - _kw['vmin'][i]) * 255
            c[i] = self.vskipfill(c[i], v)
        size = np.shape(c[0][0])
        for axnow, red, green, blue in zip(self.ax, *c):
            im = Image.new('RGB', size[::-1], (128, 128, 128))
            rgb = [red[::-1, :], green[::-1, :], blue[::-1, :]]
            for j in range(size[0]):
                for i in range(size[1]):
                    value = tuple(int(a[j, i]) for a in rgb)
                    im.putpixel((i, j), value)
            axnow.imshow(im, extent=[x[0], x[-1], y[0], y[-1]])
            axnow.set_aspect(np.abs((x[-1]-x[0]) / (y[-1]-y[0])))
        if show_beam and not self.pv:
            for i in range(3):
                self.add_beam(beam[i], beamcolor[i])

    def set_axis(self, title: dict | str | None = None, **kwargs) -> None:
        """Use Axes.set_* of matplotlib. kwargs can include the arguments of PlotAxes2D to adjust x and y axis.

        Args:
            title (dict, optional): str means set_title(str) for 2D or fig.suptitle(str) for 3D. Defaults to None.
        """
        _kw = {}
        _kw.update(kwargs)
        offunit = '(arcsec)' if self.dist == 1 else '(au)'
        if self.pv:
            offlabel = f'Offset {offunit}'
            vellabel = r'Velocity (km s$^{-1})$'
            if 'xlabel' not in _kw:
                _kw['xlabel'] = vellabel if self.swapxy else offlabel
            if 'ylabel' not in _kw:
                _kw['ylabel'] = offlabel if self.swapxy else vellabel
            _kw['samexy'] = False
        else:
            ralabel, declabel = f'R.A. {offunit}', f'Dec. {offunit}'
            if 'xlabel' not in _kw:
                _kw['xlabel'] = declabel if self.swapxy else ralabel
            if 'ylabel' not in _kw:
                _kw['ylabel'] = ralabel if self.swapxy else declabel
        if 'xlim' not in _kw:
            _kw['xlim'] = self.Xlim
        if 'ylim' not in _kw:
            _kw['ylim'] = self.Ylim
        pa2 = kwargs2PlotAxes2D(_kw)
        for ch, axnow in enumerate(self.ax):
            pa2.set_xyaxes(axnow)
            if not (ch in self.bottomleft):
                plt.setp(axnow.get_xticklabels(), visible=False)
                plt.setp(axnow.get_yticklabels(), visible=False)
                axnow.set_xlabel('')
                axnow.set_ylabel('')
            if len(self.ax) == 1:
                if self.fig is None:
                    plt.figure(0).tight_layout()
        if title is not None:
            if len(self.ax) > 1:
                t = {'y': 0.9}
                t_in = {'t': title} if type(title) is str else title
                t.update(t_in)
                for i in range(self.npages):
                    fig = plt.figure(i)
                    fig.suptitle(**t)
            else:
                t = {'label': title} if type(title) is str else title
                axnow.set_title(**t)

    def set_axis_radec(self, center: str | None = None,
                       xlabel: str = 'R.A. (ICRS)',
                       ylabel: str = 'Dec. (ICRS)',
                       nticksminor: int = 2,
                       grid: dict | None = None, title: dict | None = None) -> None:
        """Use ax.set_* of matplotlib.

        Args:
            center (str, optional): Defaults to None, initial one.
            xlabel (str, optional): Defaults to 'R.A. (ICRS)'.
            ylabel (str, optional): Defaults to 'Dec. (ICRS)'.
            nticksminor (int, optional): Interval ratio of major and minor ticks. Defaults to 2.
            grid (dict, optional): True means merely grid(). Defaults to None.
            title (dict, optional): str means set_title(str) for 2D or fig.suptitle(str) for 3D. Defaults to None.
        """
        if self.rmax > 50.:
            print('WARNING: set_axis_radec() is not supported '
                  + 'with rmax>50 yet.')
        if center is None:
            center = self.center
        if center is None:
            center = '00h00m00s 00d00m00s'
        if len(csplit := center.split()) == 3:
            center = f'{csplit[1]} {csplit[2]}'
        dec = np.radians(coord2xy(center)[1])

        def get_sec(x, i):
            return x.split(' ')[i].split('m')[1].strip('s')

        def get_hmdm(x, i):
            return x.split(' ')[i].split('m')[0]

        ra_s = get_sec(center, 0)
        dec_s = get_sec(center, 1)
        log2r = np.log10(2. * self.rmax)
        n = np.array([-3, -2, -1, 0, 1, 2, 3])

        def makegrid(second, mode):
            second = float(second)
            if mode == 'ra':
                scale, factor, sec = 1.5, 15 * np.cos(dec), r'$^{\rm s}$'
            else:
                scale, factor, sec = 0.5, 1, r'$^{\rm \prime\prime}$'
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
            decimals = max(decimals, 0)
            if mode == 'ra':
                xy, i = [ticks / 3600., ticks * 0], 0
            else:
                xy, i = [ticks * 0, ticks / 3600.], 1
            tickvalues = xy2coord(xy, center)
            tickvalues = [float(get_sec(t, i)) for t in tickvalues]
            tickvalues = np.divmod(tickvalues, 1)
            ticklabels = [f'{int(i):02d}{sec}' + f'{j:.{decimals:d}f}'[2:]
                          for i, j in zip(*tickvalues)]
            return ticks, ticksminor, ticklabels

        xticks, xticksminor, xticklabels = makegrid(ra_s, 'ra')
        yticks, yticksminor, yticklabels = makegrid(dec_s, 'dec')
        ra_hm = get_hmdm(xy2coord([xticks[3] / 3600., 0], center), 0)
        dec_dm = get_hmdm(xy2coord([0, yticks[3] / 3600.], center), 1)
        ra_hm = ra_hm.replace('h', r'$^{\rm h}$') + r'$^{\rm m}$'
        dec_dm = dec_dm.replace('d', r'$^{\circ}$') + r'$^{\prime}$'
        xticklabels[3] = ra_hm + xticklabels[3]
        yticklabels[3] = dec_dm + '\n' + yticklabels[3]
        pa2 = PlotAxes2D(True, None, 'linear', 'linear', self.Xlim, self.Ylim,
                         xlabel, ylabel, xticks, yticks, xticklabels,
                         yticklabels, xticksminor, yticksminor, grid)
        for ch, axnow in enumerate(self.ax):
            pa2.set_xyaxes(axnow)
            if not (ch in self.bottomleft):
                plt.setp(axnow.get_xticklabels(), visible=False)
                plt.setp(axnow.get_yticklabels(), visible=False)
                axnow.set_xlabel('')
                axnow.set_ylabel('')
            if len(self.ax) == 1:
                if self.fig is None:
                    plt.figure(0).tight_layout()
        if title is not None:
            if len(self.ax) > 1:
                t = {'y': 0.9}
                t_in = {'t': title} if type(title) is str else title
                t.update(t_in)
                for i in range(self.npages):
                    fig = plt.figure(i)
                    fig.suptitle(**t)
            else:
                t = {'label': title} if type(title) is str else title
                axnow.set_title(**t)

    def savefig(self, filename: str | None = None,
                show: bool = False, **kwargs) -> None:
        """Use savefig of matplotlib.

        Args:
            filename (str, optional): Output image file name. Defaults to None.
            show (bool, optional): True means doing plt.show(). Defaults to False.
        """
        _kw = {'transparent': True, 'bbox_inches': 'tight'}
        _kw.update(kwargs)
        for axnow in self.ax:
            axnow.set_xlim(*self.Xlim)
            axnow.set_ylim(*self.Ylim)
        if type(filename) is str:
            ext = filename.split('.')[-1]
            for i in range(self.npages):
                ver = '' if self.npages == 1 else f'_{i:d}'
                fig = plt.figure(i)
                fig.patch.set_alpha(0)
                fname = filename.replace(f'.{ext}', f'{ver}.{ext}')
                fig.savefig(fname, **_kw)
        if show:
            plt.show()
        plt.close()

    def get_figax(self) -> tuple[object, object]:
        """Output the external fig and ax after plotting.

        Returns:
            tuple: (fig, ax)
        """
        if len(self.ax) > 1:
            print('get_figax is not supported with channel maps')
            return -1
        return self.fig, self.ax[0]


def plotprofile(coords: list[str] | str = [],
                xlist: list[float] = [], ylist: list[float] = [],
                ellipse: list[float, float, float] | None = None,
                ninterp: int = 1,
                flux: bool = False, width: int = 1,
                gaussfit: bool = False, gauss_kwargs: dict = {},
                title: list[str] | None = None, text: list[str] | None = None,
                dist: float = 1., vsys: float = 0.,
                nrows: int = 0, ncols: int = 1, fig=None, ax=None,
                getfigax: bool = False,
                savefig: dict = None, show: bool = True,
                **kwargs) -> tuple[object, object]:
    """Use Axes.plot of matplotlib to plot line profiles at given coordinates. kwargs must include the arguments of AstroData to specify the data to be plotted. kwargs can include the arguments of PlotAxes2D to adjust x and y axes.

    Args:
        coords (list, optional): Coordinates. Defaults to [].
        xlist (list, optional): Offset from the center. Defaults to [].
        ylist (list, optional): Offset from the center. Defaults to [].
        ellipse (list, optional): [major, minor, pa], For average. Defaults to None.
        ninterp (int, optional): Number of points for interpolation. Defaults to 1.
        flux (bool, optional): y axis is flux density. Defaults to False.
        width (int, optional): Rebinning step along v. Defaults to 1.
        gaussfit (bool, optional): Fit the profiles. Defaults to False.
        gauss_kwargs (dict, optional): Kwargs for Axes.plot. Defaults to {}.
        title (list, optional): For each plot. Defaults to None.
        text (list, optional): For each plot. Defaults to None.

    Returns:
        tuple: (fig, ax), where ax is a list, if getfigax=True. Otherwise, no return.
    """
    _kw = {'drawstyle': 'steps-mid', 'color': 'k'}
    _kw.update(kwargs)
    _kwgauss = {'drawstyle': 'default', 'color': 'g'}
    _kwgauss.update(gauss_kwargs)
    savefig0 = {'bbox_inches': 'tight', 'transparent': True}
    if type(coords) is str:
        coords = [coords]
    vmin, vmax = _kw['xlim'] if 'xlim' in _kw else [-1e10, 1e10]
    f = AstroFrame(dist=dist, vsys=vsys, vmin=vmin, vmax=vmax)
    d = kwargs2AstroData(_kw)
    Tb = d.Tb
    f.read(d)
    d.binning([width, 1, 1])
    v, prof, gfitres = d.profile(coords=coords, xlist=xlist, ylist=ylist,
                                 ellipse=ellipse, ninterp=ninterp,
                                 flux=flux, gaussfit=gaussfit)
    nprof = len(prof)
    if 'ylabel' in _kw:
        ylabel = _kw['ylabel']
    elif Tb:
        ylabel = r'$T_b$ (K)'
    elif flux:
        ylabel = 'Flux (Jy)'
    else:
        ylabel = d.bunit
    if type(ylabel) is str:
        ylabel = [ylabel] * nprof

    def gauss(x, p, c, w):
        return p * np.exp(-4. * np.log(2.) * ((x - c) / w)**2)

    set_rcparams(20, 'w')
    if ncols == 1:
        nrows = nprof
    if fig is None:
        fig = plt.figure(figsize=(6 * ncols, 3 * nrows))
    if nprof > 1 and ax is not None:
        print('External ax is supported only when len(coords)=1.')
        ax = None
    ax = np.empty(nprof, dtype='object') if ax is None else [ax]
    if 'xlabel' not in _kw:
        _kw['xlabel'] = 'Velocity (km s$^{-1}$)'
    if 'xlim' not in _kw:
        _kw['xlim'] = [v.min(), v.max()]
    _kw['samexy'] = False
    pa2d = kwargs2PlotAxes2D(_kw)
    for i in range(nprof):
        sharex = None if i < nrows - 1 else ax[i - 1]
        ax[i] = fig.add_subplot(nrows, ncols, i + 1, sharex=sharex)
        if gaussfit:
            ax[i].plot(v, gauss(v, *gfitres['best'][i]), **_kwgauss)
        ax[i].plot(v, prof[i], **_kw)
        ax[i].hlines([0], v.min(), v.max(), linestyle='dashed', color='k')
        ax[i].set_ylabel(ylabel[i])
        pa2d.set_xyaxes(ax[i])
        if text is not None:
            ax[i].text(**text[i])
        if title is not None:
            if type(title[i]) is str:
                title[i] = {'label': title[i]}
            ax[i].set_title(**title[i])
        if i <= nprof - ncols - 1:
            plt.setp(ax[i].get_xticklabels(), visible=False)
    if getfigax:
        return fig, ax
    fig.tight_layout()
    if savefig is not None:
        s = {'fname': savefig} if type(savefig) is str else savefig
        savefig0.update(s)
        fig.savefig(**savefig0)
    if show:
        plt.show()
    plt.close()


def plotslice(length: float, dx: float | None = None, pa: float = 0,
              dist: float = 1, xoff: float = 0, yoff: float = 0,
              xflip: bool = True, yflip: bool = False,
              txtfile: str | None = None,
              fig: object | None = None, ax: object | None = None,
              getfigax: bool = False,
              savefig: str | dict | None = None, show: bool = False,
              **kwargs) -> None:
    """Use Axes.plot of matplotlib to plot a 1D spatial slice in a 2D map. kwargs must include the arguments of AstroData to specify the data to be plotted. kwargs can include the arguments of PlotAxes2D to adjust x and y axes.

    Args:
        length (float): Slice length.
        dx (float, optional): Grid increment. Defaults to None.
        pa (float, optional): Degree. Position angle. Defaults to 0.
        fitsimage to show: same as in PlotAstroData.
    """
    _kw = {'linestyle': '-', 'marker': 'o'}
    _kw.update(kwargs)
    savefig0 = {'bbox_inches': 'tight', 'transparent': True}
    center = _kw['center'] if 'center' in _kw else None
    f = AstroFrame(rmax=length / 2, dist=dist, xoff=xoff, yoff=yoff,
                   xflip=xflip, yflip=yflip, center=center)
    d = kwargs2AstroData(_kw)
    Tb = d.Tb
    f.read(d)
    if np.ndim(d.data) > 2:
        print('Only 2D map is supported.')
        return -1

    r, z = d.slice(length=length, pa=pa, dx=dx)
    xunit = 'arcsec' if dist == 1 else 'au'
    yunit = 'K' if Tb else d.bunit
    yquantity = 'Tb' if Tb else 'intensity'

    if txtfile is not None:
        np.savetxt(txtfile, np.c_[r, z],
                   header=f'x ({xunit}), {yquantity} ({yunit}); '
                   + f'positive x is pa={pa:.2f} deg.')
    set_rcparams()
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(1, 1, 1)
    if 'xlabel' not in _kw:
        _kw['xlabel'] = f'Offset ({xunit})'
    if 'ylabel' not in _kw:
        _kw['ylabel'] = f'Intensity ({yunit})'
    if 'xlim' not in _kw:
        _kw['xlim'] = [r.min(), r.max()]
    _kw['samexy'] = False
    pa2d = kwargs2PlotAxes2D(_kw)
    ax.plot(r, z, **_kw)
    if d.sigma is not None:
        ax.plot(r, r * 0 + 3 * d.sigma, 'k--')
    pa2d.set_xyaxes(ax)
    if getfigax:
        return fig, ax
    fig.tight_layout()
    if savefig is not None:
        s = {'fname': savefig} if type(savefig) is str else savefig
        savefig0.update(s)
        fig.savefig(**savefig0)
    if show:
        plt.show()
    plt.close()


def plot3d(levels: list[float] = [3, 6, 12],
           cmap: str = 'Jet', alpha: float = 0.08,
           xlabel: str = 'R.A. (arcsec)',
           ylabel: str = 'Dec. (arcsec)',
           vlabel: str = 'Velocity (km/s)',
           xskip: int = 1, yskip: int = 1,
           eye_p: float = 0, eye_i: float = 180,
           xplus: dict = {}, xminus: dict = {},
           yplus: dict = {}, yminus: dict = {},
           vplus: dict = {}, vminus: dict = {},
           outname: str = 'plot3d', show: bool = False,
           return_data_layout: bool = False,
           **kwargs) -> None | dict:
    """Use Plotly. kwargs must include the arguments of AstroData to specify the data to be plotted. kwargs must include the arguments of AstroFrame to specify the ranges and so on for plotting.

    Args:
        levels (list, optional): Contour levels. Defaults to [3,6,12].
        cmap (str, optional): Color map name. Defaults to 'Jet'.
        alpha (float, optional): opacity in plotly. Defaults to 0.08.
        xlabel (str, optional): Defaults to 'R.A. (arcsec)'.
        ylabel (str, optional): Defaults to 'Dec. (arcsec)'.
        vlabel (str, optional): Defaults to 'Velocity (km/s)'.
        xskip (int, optional): Number of pixel to skip. Defaults to 1.
        yskip (int, optional): Number of pixel to skip. Defaults to 1.
        eye_p (float, optional): Azimuthal angle of camera. Defaults to 0.
        eye_i (float, optional): Inclination angle of camera. Defaults to 180.
        xplus (dict, optional): 2D data to be plotted on the y-v plane at the positive edge of x. This dictionary must have a key of data and can have keys of levels, sigma, cmap, and alpha. Defaults to {}.
        xminus (dict, optional): See xplus. Defaults to {}.
        yplus (dict, optional): See xplus. Defaults to {}.
        yminus (dict, optional): See xplus. Defaults to {}.
        vplus (dict, optional): See xplus. Defaults to {}.
        vminus (dict, optional): See xplus. Defaults to {}.
        outname (str, optional): Output file name. Defaults to 'plot3d'.
        show (bool, optional): auto_play in plotly. Defaults to False.
        return_data_layout (bool, optional): Whether to return data and layout for plotly.graph_objs.Figure. Defaults to False.

    Returns:
        dict: {'data': data, 'layout': layout}, if return_data_layout=True. Otherwise, no return.
    """
    import plotly.graph_objs as go
    from skimage import measure

    f = kwargs2AstroFrame(kwargs)
    d = kwargs2AstroData(kwargs)
    f.read(d, xskip, yskip)
    volume, x, y, v, sigma = d.data, d.x, d.y, d.v, d.sigma
    dx, dy, dv = x[1] - x[0], y[1] - y[0], v[1] - v[0]
    volume[np.isnan(volume)] = 0
    if dx < 0:
        x, dx, volume = x[::-1], -dx, volume[:, :, ::-1]
    if dy < 0:
        y, dy, volume = y[::-1], -dy, volume[:, ::-1, :]
    if dv < 0:
        v, dv, volume = v[::-1], -dv, volume[::-1, :, :]
    s, ds = [x, y, v], [dx, dy, dv]
    deg = np.radians(1)
    xeye = -np.sin(eye_i * deg) * np.sin(eye_p * deg)
    yeye = -np.sin(eye_i * deg) * np.cos(eye_p * deg)
    zeye = np.cos(eye_i * deg)
    margin = dict(l=0, r=0, b=0, t=0)
    camera = dict(eye=dict(x=xeye, y=yeye, z=zeye), up=dict(x=0, y=1, z=0))
    xaxis = dict(range=[x[0], x[-1]], title=xlabel)
    yaxis = dict(range=[y[0], y[-1]], title=ylabel)
    zaxis = dict(range=[v[0], v[-1]], title=vlabel)
    scene = dict(aspectmode='cube', camera=camera,
                 xaxis=xaxis, yaxis=yaxis, zaxis=zaxis)
    layout = go.Layout(margin=margin, scene=scene, showlegend=False)

    data = []
    for lev in levels:
        if lev * sigma > np.max(volume):
            continue
        vertices, simplices, _, _ = measure.marching_cubes(volume, lev * sigma)
        Xg, Yg, Zg = [t[0] + i * dt for t, i, dt
                      in zip(s, vertices.T[::-1], ds)]
        i, j, k = simplices.T
        mesh = dict(type='mesh3d', x=Xg, y=Yg, z=Zg, i=i, j=j, k=k,
                    intensity=Zg * 0 + lev,
                    colorscale=cmap, reversescale=False,
                    cmin=np.min(levels), cmax=np.max(levels),
                    opacity=alpha, name='', showscale=False)
        data.append(mesh)
        Xe, Ye, Ze = [], [], []
        for t in vertices[simplices]:
            Xe += [x[0] + dx * t[k % 3][2] for k in range(4)] + [None]
            Ye += [y[0] + dy * t[k % 3][1] for k in range(4)] + [None]
            Ze += [v[0] + dv * t[k % 3][0] for k in range(4)] + [None]
        lines = dict(type='scatter3d', x=Xe, y=Ye, z=Ze,
                     mode='lines', opacity=0.04, visible=True,
                     name='', line=dict(color='rgb(0,0,0)', width=1))
        data.append(lines)

    def plot_on_wall(sign: int, axis: int, **kwargs):
        if kwargs == {}:
            return

        match axis:
            case 2:
                shape = np.shape(d.data[:, :, 0])
            case 1:
                shape = np.shape(d.data[:, 0, :])
            case 0:
                shape = np.shape(d.data[0, :, :])
        if np.shape(kwargs['data']) != shape:
            print('The shape of the 2D data is inconsistent with the shape of the 3D data.')
            return

        _kw = {'levels': [3, 6, 12, 24, 48, 96, 192, 384],
               'sigma': 'hist', 'cmap': 'Jet', 'alpha': 0.3}
        _kw.update(kwargs)
        volume = _kw['data']
        levels = _kw['levels']
        cmap = _kw['cmap']
        alpha = _kw['alpha']
        sigma = estimate_rms(data=volume, sigma=_kw['sigma'])
        volume[np.isnan(volume)] = 0
        a = int(sign == -1)
        b = int(sign == 1)
        volume = np.moveaxis([volume * a, volume * b], 0, axis)
        if d.x[1] - d.x[0] < 0:
            volume = volume[:, :, ::-1]
        if d.y[1] - d.y[0] < 0:
            volume = volume[:, ::-1, :]
        if d.v[1] - d.v[0] < 0:
            volume = volume[::-1, :, :]
        for lev in levels:
            if lev * sigma > np.max(volume):
                continue
            vertices, simplices, _, _ = measure.marching_cubes(volume, lev * sigma)
            Xg, Yg, Zg = [t[0] + i * dt for t, i, dt
                          in zip(s, vertices.T[::-1], ds)]
            match axis:
                case 2:
                    Xg = Xg * 0 + (x[-1] if sign == 1 else x[0])
                case 1:
                    Yg = Yg * 0 + (y[-1] if sign == 1 else y[0])
                case 0:
                    Zg = Zg * 0 + (v[-1] if sign == 1 else v[0])
            i, j, k = simplices.T
            mesh2d = dict(type='mesh3d', x=Xg, y=Yg, z=Zg,
                          i=i, j=j, k=k,
                          intensity=Zg * 0 + lev,
                          colorscale=cmap, reversescale=False,
                          cmin=np.min(levels), cmax=np.max(levels),
                          opacity=alpha, name='', showscale=False)
            data.append(mesh2d)

    klist = [xplus, xminus, yplus, yminus, vplus, vminus]
    slist = [1, -1, 1, -1, 1, -1]
    alist = [2, 2, 1, 1, 0, 0]
    for kw, sign, axis in zip(klist, slist, alist):
        plot_on_wall(sign=sign, axis=axis, **kw)

    if return_data_layout:
        return {'data': data, 'layout': layout}
    else:
        fig = go.Figure(data=data, layout=layout)
        fig.write_html(file=outname + '.html', auto_play=show)
