import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from fits_utils import FitsData
from other_utils import coord2xy, rel2abs

def fits2data(fitsimage: str, Tb: bool = False,
              center: str = '', vsys: float = 0) -> list:
    fd = FitsData(fitsimage)
    fd.gen_header()
    bunit = fd.header['BUNIT'] if 'BUNIT' in fd.header else ''
    fd.gen_data(Tb=Tb, drop=True)
    fd.gen_grid(ang='arcsec', vel='km/s', center=center, vsys=vsys)
    grid = [fd.x, fd.y, fd.v]
    fd.gen_beam(ang='arcsec')
    beam = [fd.bmaj, fd.bmin, fd.bpa]
    return [fd.data, grid, beam, bunit]

def find_rms(data, sigma):
    nums = [float, int, np.float64, np.int64, np.float32, np.int32]
    if type(sigma) in nums: noise = sigma
    elif sigma == 'edge': noise = np.nanstd(data[::len(data) - 1])
    elif sigma == 'neg': noise = np.sqrt(np.nanmean(data[data < 0]**2))
    elif sigma == 'med': noise = np.sqrt(np.nanmedian(data**2) / 0.454936)
    elif sigma == 'iter':
        n = data.copy()
        for i in range(20):
            ave, sig = np.nanmean(n), np.nanstd(n)
            n = n - np.nanmean(n)
            n = n[np.abs(n) < 3.5 * sig]
        noise = np.nanstd(n)
    elif sigma == 'out':
        n, n0, n1 = data.copy(), len(data), len(data[0])
        n[n0//5 : n0*4//5, n1//5 : n1*4//5] = np.nan
        noise = np.nanstd(n)
    print(f'sigma = {noise:.2e}')
    return noise


def set_rcparams(fontsize=18, nancolor='w', direction='inout'):
    #plt.rcParams['font.family'] = 'arial'
    plt.rcParams['axes.facecolor'] = nancolor
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['legend.fontsize'] = 15
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.direction'] = direction
    plt.rcParams['ytick.direction'] = direction
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


class plotastro2D():
    """Make a figure from 2D FITS files or 2D arrays.
    
    Basic rules --- Lengths are in the unit of arcsec.
    Angles are in the unit of degree.
    For ellipse, line, arrow, label, and marker,
    a single input must be listed like poslist=[[0.2, 0.3]],
    and each element of poslist supposes a text coordinate
    like '01h23m45.6s 01d23m45.6s' or a list of relative x and y
    like [0.2, 0.3] (0 is left or bottom, 1 is right or top).
    Parameters for original methods in matplotlib.axes.Axes can be
    used as kwargs; see kwargs0 for reference.
    """
    def __init__(self, fig=None, ax=None,
                 center: str = '', rmax: float = 100,
                 xoff: float = 0, yoff: float = 0,
                 xflip: bool = True, yflip: bool = False) -> None:
        set_rcparams()
        self.fig = plt.figure(figsize=(7, 5)) if fig is None else fig
        self.ax = self.fig.add_subplot(1, 1, 1) if ax is None else ax
        self.center = center
        self.rmax = rmax
        self.xoff, self.yoff = xoff, yoff
        self.xmin, self.ymin = xmin, ymin = xoff - rmax, yoff - rmax
        self.xmax, self.ymax = xmax, ymax = xoff + rmax, yoff + rmax
        self.xedge = [xmax, xmin] if xflip else [xmin, xmax]
        self.yedge = [ymax, ymin] if yflip else [ymin, ymax]
        self.xflip, self.yflip = xflip, yflip

    def __pos2xy(self, pos: list = []) -> list:
        x, y = [None] * len(pos), [None] * len(pos)
        for i, p in enumerate(pos):
            if type(p) == str:
                x[i], y[i] = coord2xy(p) - coord2xy(self.center)
            else:
                x[i], y[i] = rel2abs(*p, self.xedge, self.yedge)
        return [x, y]
        
    
    def add_ellipse(self, poslist: list = [],
                    majlist: list = [], minlist: list = [],
                    palist: list = [], **kwargs) -> None:
        kwargs0 = {'facecolor':'none', 'edgecolor':'gray', 'linewidth':1.5}                
        for x, y, width, height, angle\
            in zip(*self.__pos2xy(poslist), minlist, majlist, palist):
            e = Ellipse((x, y), width=width, height=height,
                        angle=angle * (-1 if self.xflip else 1),
                        **dict(kwargs0, **kwargs), zorder=10)
            self.ax.add_patch(e)

    def add_beam(self, bmaj, bmin, bpa, beamcolor) -> None:                
        bpos = max(0.7 * bmaj / self.rmax, 0.075)
        self.add_ellipse(poslist=[[bpos, bpos]],
                         majlist=[bmaj], minlist=[bmin], palist=[bpa],
                         facecolor=beamcolor, edgecolor=None)

    def add_color(self, fitsimage: str = None,
                   x: list = None, y: list = None, skip: int = 1,
                   c: list = None, cmap: str = 'cubehelix',
                   cmin: float = None, cmax: float = None,
                   Tb: bool = False, logc: bool = False,
                   alpha: float = 1, show_cbar: bool = True,
                   clabel: str = None, cformat: float = '%.1e',
                   show_beam: bool = True, beamcolor: str = 'gray',
                   bmaj: float = 0., bmin: float = 0.,
                   bpa: float = 0., **kwargs) -> None:
        kwargs0 = {'cmap':'cubehelix', 'alpha':1}
        if not fitsimage is None:
            c, (x, y, _), (bmaj, bmin, bpa), bunit \
                = fits2data(fitsimage, Tb, self.center)
        x, y, c = x[::skip], y[::skip], c[::skip, ::skip]
        c = np.array(c)
        if logc:
            cmin = c[c > 0].min() if cmin is None else cmin
            c = np.log10(c.clip(cmin, cmax))
        else:
            if not (cmin is None and cmax is None):
                c = c.clip(cmin, cmax)
        
        p = self.ax.pcolormesh(x, y, c, shading='nearest',
                               **dict(kwargs0, **kwargs), zorder=1)
        if show_cbar:
            clabel = bunit if clabel is None else clabel
            plt.colorbar(p, ax=self.ax, label=clabel, format=cformat)
        if show_beam:
            self.add_beam(bmaj, bmin, bpa, beamcolor)
            
    def add_contour(self, fitsimage: str = None,
                     x: list = None, y: list = None, skip: int = 1,
                     c: list = None, sigma: str or float = 'neg',
                     levels: list = [-12,-6,-3,3,6,12,24,48,96,192,384],
                     Tb: bool = False,
                     show_beam: bool = True, beamcolor: str = 'gray',
                     bmaj: float = 0., bmin: float = 0., bpa: float = 0.,
                     **kwargs) -> None:
        kwargs0 = {'colors':'gray', 'linewidths':1.0}
        if not fitsimage is None:
            c, (x, y, _), (bmaj, bmin, bpa), _ \
                = fits2data(fitsimage, Tb, self.center)
        x, y, c = x[::skip], y[::skip], c[::skip, ::skip]
        self.ax.contour(x, y, c, np.array(levels) * find_rms(c, sigma),
                        **dict(kwargs0, **kwargs), zorder=2)
        if show_beam:
            self.add_beam(bmaj, bmin, bpa, beamcolor)
    
    def add_vector(self, ampfits: str = None, angfits: str = None,
                     x: list = None, y: list = None, skip: int = 1,
                     amp: list = None, ang: list = None,
                     ampfactor: float = 1.,
                     show_beam: bool = True, beamcolor: str = 'gray',
                     bmaj: float = 0., bmin: float = 0., bpa: float = 0.,
                     **kwargs) -> None:
        kwargs0 = {'angles':'xy', 'scale_units':'xy', 'color':'gray',
                   'pivot':'mid', 'headwidth':0, 'headlength':0,
                   'headaxislength':0, 'width':0.007}
        if not ampfits is None:
            amp, (x, y, _), (bmaj, bmin, bpa), _ \
                = fits2data(ampfits, self.center)
        if not angfits is None:
            ang, (x, y, _), (bmaj, bmin, bpa), _ \
                = fits2data(angfits, self.center)
        if amp is None and not ang is None:
            amp = np.ones_like(ang)
        x, y = x[::skip], y[::skip]
        amp, ang = amp[::skip, ::skip], ang[::skip, ::skip]
        u = ampfactor * amp * np.sin(np.radians(ang))
        v = ampfactor * amp * np.cos(np.radians(ang))
        kwargs0['scale'] = 1. / np.abs(x[1] - x[0])
        self.ax.quiver(x, y, u, v, **dict(kwargs0, **kwargs), zorder=3)
        if show_beam:
            self.add_beam(bmaj, bmin, bpa, beamcolor)
            
    def add_scalebar(self, length: float = 0, label: str = '',
                     color: str = 'gray', barpos: tuple = (0.83, 0.17),
                     fontsize: float = 24, linewidth: float = 3):
        x = [self.xmax, self.xmin] if self.xflip else [self.xmin, self.xmax]
        y = [self.ymax, self.ymin] if self.yflip else [self.ymin, self.ymax]
        if length > 0 and label != '':
            a, b = barpos
            x0, y0 = rel2abs(a, b * 0.9, self.xedge, self.yedge)
            self.ax.text(x0, y0, label, color=color, size=fontsize,
                         ha='center', va='top', zorder=10)
            x0, y0 = rel2abs(a, b, self.xedge, self.yedge)
            self.ax.plot([x0 - length/2., x0 + length/2.], [y0, y0],
                         '-', linewidth=linewidth, color=color)
            
    def add_marker(self, poslist: list = [], **kwargs):
        kwsmark0 = {'marker':'+', 'ms':30, 'mfc':'gray',
                    'mec':'gray', 'mew':2, 'alpha':1}
        for x, y in zip(*self.__pos2xy(poslist)):
            self.ax.plot(x, y, **dict(kwsmark0, **kwargs))
            
    def add_label(self, poslist: list = [],
                  slist: list = [], **kwargs) -> None:
        kwargs0 = {'color':'gray'}
        for x, y, s in zip(*self.__pos2xy(poslist), slist):
            self.ax.text(x=x, y=y, s=s,
                         **dict(kwargs0, **kwargs), zorder=20)

    def add_line(self, poslist: list = [], anglelist: list = [],
                 rlist: list = [], **kwargs):
        kwargs0 = {'color':'gray', 'linewidth':1.5}
        for x, y, a, r \
            in zip(*self.__pos2xy(poslist), np.radians(anglelist), rlist):
            self.ax.plot([x, x + r * np.sin(a)],
                         [y, y + r * np.cos(a)],
                         **dict(kwargs0, **kwargs))

    def add_arrow(self, poslist: list = [], anglelist: list = [],
                  rlist: list = [], **kwargs):
        kwargs0 = {'color':'gray', 'width':0.012,
                   'headwidth':5, 'headlength':5}
        for x, y, a, r \
            in zip(*self.__pos2xy(poslist), np.radians(anglelist), rlist):
            self.ax.quiver(x, y, r * np.sin(a), r * np.cos(a),
                           angles='xy', scale_units='xy', scale=1,
                           **dict(kwargs0, **kwargs))

        
    def set_axis(self, xticks: list = None, yticks: list = None,
                 xticksminor: list = None, yticksminor: list = None,
                 xticklabels: list = None, yticklabels: list= None,
                 xlabel: str = 'R.A. (arcsec)',
                 ylabel: str = 'Dec. (arcsec)',
                 samexy: bool = True) -> None:
        if samexy:
            self.ax.set_xticks(self.ax.get_yticks())
            self.ax.set_yticks(self.ax.get_xticks())
            self.ax.set_aspect(1)
        if not xticks is None: self.ax.set_xticks(xticks)
        if not yticks is None: self.ax.set_yticks(yticks)
        if not xticksminor is None:
            self.ax.set_xticks(xticksminor, minor=True)
        if not yticksminor is None:
            self.ax.set_yticks(yticksminor, minor=True)
        if not xticklabels is None: self.ax.set_xticklabels(xticklabels)
        if not yticklabels is None: self.ax.set_yticklabels(yticklabels)
        if self.xflip:
            self.ax.set_xlim(self.xmax, self.xmin)
        else:
            self.ax.set_xlim(self.xmin, self.xmax)
        if self.yflip:
            self.ax.set_ylim(self.ymax, self.ymin)
        else:
            self.ax.set_ylim(self.ymin, self.ymax)
        if not xlabel is None: self.ax.set_xlabel(xlabel)
        if not ylabel is None: self.ax.set_ylabel(ylabel)
        self.fig.tight_layout()
        
    def savefig(self, filename: str, transparent: bool =True) -> None:
        self.fig.patch.set_alpha(0)
        self.fig.savefig(filename, bbox_inches='tight',
                         transparent=transparent)
        
        
class plotastro3D():
    """Make a figure from 3D FITS files or 3D arrays.
    
    Basic rules --- Lengths are in the unit of arcsec.
    Angles are in the unit of degree.
    For ellipse, line, arrow, label, and marker,
    a single input must be listed like poslist=[[0.2, 0.3]],
    and each element of poslist supposes a text coordinate
    like '01h23m45.6s 01d23m45.6s' or a list of relative x and y
    like [0.2, 0.3] (0 is left or bottom, 1 is right or top).
    Parameters for original methods in matplotlib.axes.Axes can be
    used as kwargs; see kwargs0 for reference.
    """
    def __init__(self, fitsimage: str = None,
                 npages: int = 1, nrows: int = 4, ncols: int = 6,
                 vsys: float = 0, vmin: float = -100., vmax: float = 100.,
                 vskip: int = 1,
                 center: str = '', rmax: float = 100,
                 xoff: float = 0, yoff: float = 0,
                 xflip: bool = True, yflip: bool = False) -> None:
        set_rcparams()
        if not fitsimage is None:
            fd = FitsData(fitsimage)
            fd.gen_grid(ang='arcsec', vel='km/s', vsys=vsys)
            k0 = np.argmin(np.abs(fd.v - vmin))
            k1 = np.argmin(np.abs(fd.v - vmax))
            k0, k1 = sorted([k0, k1])    
            nch = len(fd.v[k0:k1+1:vskip])
            npages = int(np.ceil(nch / nrows / ncols))
        
        figlist = [None] * npages
        axlist = [None] * npages
        for i in range(npages):
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                                   sharex=True, sharey=True, squeeze=False,
                                   figsize=(ncols*2, max(nrows, 1.5)*2))
            fig.subplots_adjust(hspace=0, wspace=0, right=0.87, top=0.87)
            figlist[i] = fig
            axlist[i] = ax
        self.npages = npages
        self.nrows = nrows
        self.ncols = ncols
        self.figlist = figlist
        self.axlist = axlist
        self.vsys = vsys
        self.vmin, self.vmax = vmin, vmax
        self.vskip = vskip
        self.center = center
        self.rmax = rmax
        self.xoff, self.yoff = xoff, yoff
        self.xmin, self.ymin = xmin, ymin = xoff - rmax, yoff - rmax
        self.xmax, self.ymax = xmax, ymax = xoff + rmax, yoff + rmax
        self.xedge = [xmax, xmin] if xflip else [xmin, xmax]
        self.yedge = [ymax, ymin] if yflip else [ymin, ymax]
        self.xflip, self.yflip = xflip, yflip
        allchan = [None] * (npages * nrows * ncols)
        for n in range(npages):
            for i in range(nrows):
                for j in range(ncols):
                    allchan[n*nrows*ncols + i*ncols + j] = [n, i, j]
        self.allchan = allchan
        
    def __pos2xy(self, pos: list = []) -> list:
        x, y = [None] * len(pos), [None] * len(pos)
        for i, p in enumerate(pos):
            if type(p) == str:
                x[i], y[i] = coord2xy(p) - coord2xy(self.center)
            else:
                x[i], y[i] = rel2abs(*p, self.xedge, self.yedge)
        return [x, y]

    def add_ellipse(self, pagerowcol: list = None, poslist: list = [],
                    majlist: list = [], minlist: list = [],
                    palist: list = [], **kwargs) -> None:
        kwargs0 = {'facecolor':'none', 'edgecolor':'gray', 'linewidth':1.5}
        if pagerowcol is None: pagerowcol = self.allchan
        for x, y, width, height, angle\
            in zip(*self.__pos2xy(poslist), minlist, majlist, palist):
            e = Ellipse((x, y), width=width, height=height,
                        angle=angle * (-1 if self.xflip else 1),
                        **dict(kwargs0, **kwargs), zorder=10)
            for n in range(self.npages):
                for i in range(self.nrows):
                    for j in range(self.ncols):
                        if [n, i, j] in pagerowcol:
                            self.axlist[n][i, j].add_patch(e)

    def add_beam(self, bmaj, bmin, bpa, beamcolor) -> None:                
        bpos = max(0.7 * bmaj / self.rmax, 0.075)
        pagerowcol = [[n, self.nrows - 1, 0] for n in range(self.npages)]
        self.add_ellipse(pagerowcol=pagerowcol,
                         poslist=[[bpos, bpos]],
                         majlist=[bmaj], minlist=[bmin], palist=[bpa],
                         facecolor=beamcolor, edgecolor=None)

    def add_contour(self, fitsimage: str = None,
                    x: list = None, y: list = None, skip: int = 1,
                    c: list = None, sigma: str or float = 'edge',
                    levels: list = [-12,-6,-3,3,6,12,24,48,96,192,384],
                    Tb: bool = False,
                    show_beam: bool = True, beamcolor: str = 'gray',
                    bmaj: float = 0., bmin: float = 0., bpa: float = 0.,
                    **kwargs) -> None:
        kwargs0 = {'colors':'gray', 'linewidths':1.0}
        if not fitsimage is None:
            c, (x, y, v), (bmaj, bmin, bpa), _ \
                = fits2data(fitsimage, Tb, self.center)
        rms = find_rms(c, sigma)
        k0 = np.argmin(np.abs(v - self.vmin))
        k1 = np.argmin(np.abs(v - self.vmax))
        k0, k1 = sorted([k0, k1])    
        v = v[k0:k1+1:self.vskip]
        x, y = x[::skip], y[::skip]
        c = c[k0:k1+1:self.vskip, ::skip, ::skip]
        for n in range(self.npages):
            for i in range(self.nrows):
                for j in range(self.ncols):
                    ax = self.axlist[n][i, j]
                    ch = n*self.nrows*self.ncols + i*self.ncols + j
                    if ch < len(v):
                        ax.contour(x, y, c[ch],
                                   np.array(levels) * rms,
                                   **dict(kwargs0, **kwargs), zorder=2)
        if show_beam:
            self.add_beam(bmaj, bmin, bpa, beamcolor)
