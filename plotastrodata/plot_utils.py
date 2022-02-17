import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from other_utils import coord2xy, rel2abs, estimate_rms, trim
from fits_utils import FitsData, fits2data



def pos2xy(pad, pos: list = []) -> list:
    x, y = [None] * len(pos), [None] * len(pos)
    for i, p in enumerate(pos):
        if type(p) == str:
            center = pad.gridpar['center']
            x[i], y[i] = (coord2xy(p)-coord2xy(center)) * 3600.
        else:
            x[i], y[i] = rel2abs(*p, pad.xlim, pad.ylim)
    return [x, y]


def readfits(plotastro, fitsimage: str, Tb: bool = False,
             sigma: str = None, restfrq: float = None) -> list:
    data, grid, beam, bunit, rms \
        = fits2data(fitsimage=fitsimage, Tb=Tb, log=False,
                    sigma=sigma, restfrq=restfrq, **plotastro.gridpar)
    return [data, grid[:2], beam, bunit, rms]


def readdata(x: list = None, y: list = None, v: list = None,
             lims: list = None, data: list = None):
    dataout, grid = trim(data=data, x=x, y=y, v=v,
                         xlim=lims[0], ylim=lims[1], vlim=lims[2])
    return [dataout, grid[:2]]
    

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


class plotastrodata():
    """Make a figure from 2D/3D FITS files or 2D/3D arrays.
    
    Basic rules --- For 3D data, a 1D velocity array or a FITS file
    with a velocity axis must be given to set up channels in each page.
    len(v)=1 (default) means to make a 2D figure.
    Lengths are in the unit of arcsec.
    Angles are in the unit of degree.
    For ellipse, line, arrow, label, and marker,
    a single input must be listed like poslist=[[0.2, 0.3]],
    and each element of poslist supposes a text coordinate
    like '01h23m45.6s 01d23m45.6s' or a list of relative x and y
    like [0.2, 0.3] (0 is left or bottom, 1 is right or top).
    Parameters for original methods in matplotlib.axes.Axes can be
    used as kwargs; see the default kwargs0 for reference.
    """
    def __init__(self, fitsimage: str = None,
                 v: list = [0], vskip: int = 1,
                 vmin: float = -1e10, vmax: float = 1e10,
                 vsys: float = 0., veldigit: int = 2,
                 nrows: int = 4, ncols: int = 6,
                 center: str = None, rmax: float = 1e10, dist: float = 1.,
                 xoff: float = 0, yoff: float = 0,
                 xflip: bool = True, yflip: bool = False):
        if not fitsimage is None:
            fd = FitsData(fitsimage)
            _, _, v = fd.get_grid(vsys=vsys, vmin=vmin, vmax=vmax)
        nv = len(v := v[::vskip])
        if nv == 1:
            nrows = ncols = npages = nchan = 1
        else:
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
        self.xlim = [xoff - xdir*rmax, xoff + xdir*rmax]
        self.ylim = [yoff - ydir*rmax, yoff + ydir*rmax]
        self.vlim = [vmin, vmax]
        self.lims = [self.xlim, self.ylim, self.vlim]
        self.gridpar = {'center':center, 'rmax':rmax,
                        'dist':dist, 'xoff':xoff, 'yoff':yoff,
                        'vsys':vsys, 'vmin':vmin, 'vmax':vmax}
        self.rowcol = nrows * ncols
        self.npages = npages
        self.allchan = np.arange(nchan)
        self.bottomleft = nij2ch(np.arange(npages), nrows - 1, 0)

        def skipfill(c: list, skip: int = 1) -> list:
            if np.ndim(c) == 3:
                d = c[::vskip, ::skip, ::skip]
            else:
                d = c[::skip, ::skip]
                d = np.full((nv, *np.shape(d)), d)
            shape = (nchan - len(d), len(d[0]), len(d[0, 0]))
            dnan = np.full(shape, d[0] * np.nan)
            return np.concatenate((d, dnan), axis=0)
        self.skipfill = skipfill
        
    def add_ellipse(self, poslist: list = [],
                    majlist: list = [], minlist: list = [], palist: list = [],
                    include_chan: list = None, **kwargs) -> None:
        kwargs0 = {'facecolor':'none', 'edgecolor':'gray',
                   'linewidth':1.5, 'zorder':10}
        if include_chan is None: include_chan = self.allchan
        for x, y, width, height, angle\
            in zip(*pos2xy(self, poslist), minlist, majlist, palist):
            for ch, axnow in enumerate(self.ax):
                if not (ch in include_chan):
                    continue
                plt.figure(ch // self.rowcol)
                e = Ellipse((x, y), width=width, height=height,
                            angle=angle * self.xdir,
                            **dict(kwargs0, **kwargs))
                axnow.add_patch(e)
                
    def add_beam(self, bmaj, bmin, bpa, beamcolor) -> None:
        bpos = max(0.7 * bmaj / self.gridpar['rmax'], 0.075)
        self.add_ellipse(include_chan=self.bottomleft,
                         poslist=[[bpos, bpos]],
                         majlist=[bmaj], minlist=[bmin], palist=[bpa],
                         facecolor=beamcolor, edgecolor=None)
    
    def add_marker(self, poslist: list = [],
                   include_chan: list = None, **kwargs):
        kwsmark0 = {'marker':'+', 'ms':10, 'mfc':'gray',
                    'mec':'gray', 'mew':2, 'alpha':1}
        if include_chan is None: include_chan = self.allchan
        for ch, axnow in enumerate(self.ax):
            if not (ch in include_chan):
                continue
            for x, y in zip(*pos2xy(self, poslist)):
                axnow.plot(x, y, **dict(kwsmark0, **kwargs), zorder=10)
            
    def add_label(self, poslist: list = [], slist: list = [],
                  include_chan: list = None, **kwargs) -> None:
        kwargs0 = {'color':'gray', 'fontsize':15, 'zorder':10}
        if include_chan is None: include_chan = self.allchan
        for ch, axnow in enumerate(self.ax):
            if not (ch in include_chan):
                continue
            for x, y, s in zip(*pos2xy(self, poslist), slist):
                axnow.text(x=x, y=y, s=s, **dict(kwargs0, **kwargs))

    def add_line(self, poslist: list = [], anglelist: list = [],
                 rlist: list = [], include_chan: list = None, **kwargs):
        kwargs0 = {'color':'gray', 'linewidth':1.5, 'zorder':10}
        if include_chan is None: include_chan = self.allchan
        for ch, axnow in enumerate(self.ax):
            if not (ch in include_chan):
                continue
            for x, y, a, r \
                in zip(*pos2xy(self, poslist), np.radians(anglelist), rlist):
                axnow.plot([x, x + r * np.sin(a)],
                           [y, y + r * np.cos(a)],
                           **dict(kwargs0, **kwargs))

    def add_arrow(self, poslist: list = [], anglelist: list = [],
                  rlist: list = [], include_chan: list = None, **kwargs):
        kwargs0 = {'color':'gray', 'width':0.012,
                   'headwidth':5, 'headlength':5, 'zorder':10}
        if include_chan is None: include_chan = self.allchan
        for ch, axnow in enumerate(self.ax):
            if not (ch in include_chan):
                continue
            for x, y, a, r \
                in zip(*pos2xy(self, poslist), np.radians(anglelist), rlist):
                axnow.quiver(x, y, r * np.sin(a), r * np.cos(a),
                             angles='xy', scale_units='xy', scale=1,
                             **dict(kwargs0, **kwargs))
                
    def add_scalebar(self, length: float = 0, label: str = '',
                     color: str = 'gray', barpos: tuple = (0.75, 0.17),
                     fontsize: float = None, linewidth: float = 3):
        if length == 0 or label == '':
            print('Please set length and label.')
            return -1
        if fontsize is None:
            fontsize = 20 if len(self.ax) == 1 else 15 
        for ch, axnow in enumerate(self.ax):
            if not (ch in self.bottomleft):
                continue
            xrel, yrel = barpos
            x, y = rel2abs(xrel, yrel * 0.9, self.xlim, self.ylim)
            axnow.text(x, y, label, color=color, size=fontsize,
                       ha='center', va='top', zorder=10)
            x, y = rel2abs(xrel, yrel, self.xlim, self.ylim)
            axnow.plot([x - length/2., x + length/2.], [y, y],
                       '-', linewidth=linewidth, color=color)
    
    def add_color(self, fitsimage: str = None,
                  x: list = None, y: list = None, skip: int = 1,
                  v: list = None, c: list = None,
                  restfrq: float = None,
                  Tb: bool = False, log: bool = False,
                  show_cbar: bool = True,
                  cblabel: str = None, cbformat: float = '%.1e',
                  cbticks: list = None, cbticklabels: list = None,
                  show_beam: bool = True, beamcolor: str = 'gray',
                  bmaj: float = 0., bmin: float = 0.,
                  bpa: float = 0., **kwargs) -> None:
        kwargs0 = {'cmap':'cubehelix', 'alpha':1, 'zorder':1}
        if fitsimage is None:
            bunit, rms = '', estimate_rms(c, 'out')
            c, (x, y) = readdata(x, y, v, self.lims, c)    
        else:
            c, (x, y), (bmaj, bmin, bpa), bunit, rms \
                = readfits(self, fitsimage, Tb, 'out', restfrq)
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
            if not (cbticks is None):
                cb.set_ticks(np.log10(cbticks) if log else cbticks)
            if not (cbticklabels is None):
                cb.set_ticklabels(cbticklabels)
            elif log:
                cb.set_ticks(t := cb.get_ticks())
                cb.set_ticklabels([f'{d:.1e}' for d in 10**t])
        if show_beam:
            self.add_beam(bmaj, bmin, bpa, beamcolor)

    def add_contour(self, fitsimage: str = None,
                    x: list = None, y: list = None, skip: int = 1,
                    v: list = None, c: list = None,
                    restfrq: float = None,
                    sigma: str or float = 'edge',
                    levels: list = [-12,-6,-3,3,6,12,24,48,96,192,384],
                    Tb: bool = False,
                    show_beam: bool = True, beamcolor: str = 'gray',
                    bmaj: float = 0., bmin: float = 0., bpa: float = 0.,
                    **kwargs) -> None:
        kwargs0 = {'colors':'gray', 'linewidths':1.0, 'zorder':2}
        if fitsimage is None:
            if np.ndim(c) == 2 and sigma == 'edge': sigma = 'out'
            rms = estimate_rms(c, sigma)
            c, (x, y) = readdata(x, y, v, self.lims, c)
        else:
            c, (x, y), (bmaj, bmin, bpa), _, rms \
                = readfits(self, fitsimage, Tb, sigma, restfrq)
        x, y = x[::skip], y[::skip]
        c = self.skipfill(c, skip)
        for axnow, cnow in zip(self.ax, c):
            axnow.contour(x, y, cnow, np.array(levels) * rms,
                          **dict(kwargs0, **kwargs))
        if show_beam:
            self.add_beam(bmaj, bmin, bpa, beamcolor)
            
    def add_segment(self, ampfits: str = None, angfits: str = None,
                    x: list = None, y: list = None, skip: int = 1,
                    v: list = None, amp: list = None, ang: list = None,
                    ampfactor: float = 1.,
                    show_beam: bool = True, beamcolor: str = 'gray',
                    bmaj: float = 0., bmin: float = 0., bpa: float = 0.,
                    **kwargs) -> None:
        kwargs0 = {'angles':'xy', 'scale_units':'xy', 'color':'gray',
                   'pivot':'mid', 'headwidth':0, 'headlength':0,
                   'headaxislength':0, 'width':0.007, 'zorder':3}
        if ampfits is None:
            amp, (x, y) = readdata(x, y, v, self.lims, amp)
        else:
            amp, (x, y), (bmaj, bmin, bpa), _, _ = readfits(self, ampfits)
        if angfits is None:
            ang, (x, y) = readdata(x, y, v, self.lims, ang)
        else:
            ang, (x, y), (bmaj, bmin, bpa), _, _ = readfits(self, angfits)
        if amp is None: amp = np.ones_like(ang)
        x, y = x[::skip], y[::skip]
        amp = self.skipfill(amp, skip)
        ang = self.skipfill(ang, skip)
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
                 xlabel: str = 'R.A. (arcsec)',
                 ylabel: str = 'Dec. (arcsec)',
                 grid: dict = None, samexy: bool = True) -> None:
        
        if xticklabels is None and not (xticks is None):
                xticklabels = [str(t) for t in xticks]
        if yticklabels is None and not (xticks is None):
                yticklabels = [str(t) for t in yticks]
        for ch, axnow in enumerate(self.ax):
            if samexy:
                axnow.set_xticks(axnow.get_yticks())
                axnow.set_yticks(axnow.get_xticks())
                axnow.set_aspect(1)
            if not xticks is None: axnow.set_xticks(xticks)
            if not yticks is None: axnow.set_yticks(yticks)
            if not xticksminor is None:
                axnow.set_xticks(xticksminor, minor=True)
            if not yticksminor is None:
                axnow.set_yticks(yticksminor, minor=True)
            if not (xticklabels is None):
                axnow.set_xticklabels(xticklabels)
            if not (yticklabels is None):
                axnow.set_yticklabels(yticklabels)
            if not xlabel is None:
                axnow.set_xlabel(xlabel)
            if not ylabel is None:
                axnow.set_ylabel(ylabel)
            if not (ch in self.bottomleft):
                plt.setp(axnow.get_xticklabels(), visible=False)
                plt.setp(axnow.get_yticklabels(), visible=False)
                axnow.set_xlabel('')
                axnow.set_ylabel('')
            axnow.set_xlim(*self.xlim)
            axnow.set_ylim(*self.ylim)
            if not grid is None:
                axnow.grid(**({} if grid == True else grid))
            if len(self.ax) == 1: plt.figure(0).tight_layout()
            
    def savefig(self, filename: str = 'plotastro3D.png',
                transparent: bool =True) -> None:
        for axnow in self.ax:
            axnow.set_xlim(*self.xlim)
            axnow.set_ylim(*self.ylim)
        ext = filename.split('.')[-1]
        for i in range(self.npages):
            ver = '' if len(self.ax) == 1 else f'_{i:d}'
            fig = plt.figure(i)
            fig.patch.set_alpha(0)
            fig.savefig(filename.replace('.' + ext, ver + '.' + ext),
                        bbox_inches='tight', transparent=transparent)
            
    def show(self):
        for axnow in self.ax:
            axnow.set_xlim(*self.xlim)
            axnow.set_ylim(*self.ylim)
        plt.show()
