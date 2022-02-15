from re import I
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits
from astropy import constants, units, wcs

from other_utils import coord2xy, rel2abs


def Jy2K(header = None, bmaj: float = None, bmin: float = None,
         freq: float = None) -> float:
    """Calculate a conversion factor in the unit of K/Jy.

    Args:
        header (optional): astropy.io.fits.open('a.fits')[0].header
                           Defaults to None.
        bmaj (float, optional): beam major axis in degree. Defaults to None.
        bmin (float, optional): beam minor axis in degree. Defaults to None.
        freq (float, optional): rest frequency in Hz. Defaults to None.

    Returns:
        float: the conversion factor in the unit of K/Jy.
    """
    if not header is None:
        bmaj, bmin = header['BMAJ'], header['BMIN']
        if 'RESTFREQ' in header.keys(): freq = header['RESTFREQ']
        if 'RESTFRQ' in header.keys(): freq = header['RESTFRQ']
    omega = bmaj * bmin * np.radians(1)**2 * np.pi / 4. * np.log(2.)
    lam = constants.c.to('m/s').value / freq
    a = units.Jy.to('J*s**(-1)*m**(-2)*Hz**(-1)') \
        * lam**2 / 2. / constants.k_B.to('J/K').value / omega
    return a


class FitsData:
    """For practical treatment of data in a FITS file."""
    def __init__(self, fitsimage: str):
        self.fitsimage = fitsimage

    def gen_hdu(self):
        self.hdu = fits.open(self.fitsimage)[0]
        
    def gen_header(self) -> None:
        if not hasattr(self, 'hdu'):
            self.gen_hdu()
        self.header = self.hdu.header

    def get_header(self, key: str = None) -> int or float:
        if not hasattr(self, 'header'):
            self.gen_header()
        if key is None:
            return self.header
        if key in self.header:
            return self.header[key]
        print(f'{key} is not in the header.')
        return None

    def gen_beam(self, dist: float = 1.) -> None:
        bmaj = self.get_header('BMAJ')
        bmin = self.get_header('BMIN')
        bpa = self.get_header('BPA')
        bmaj = 0 if bmaj is None else bmaj * 3600.
        bmin = 0 if bmin is None else bmin * 3600.
        bpa = 0 if bpa is None else bpa
        self.bmaj, self.bmin, self.bpa = bmaj * dist, bmin * dist, bpa

    def get_beam(self, dist: float = 1.) -> list:
        if not hasattr(self, 'bmaj'):
            self.gen_beam(dist)
        return [self.bmaj, self.bmin, self.bpa]

    def gen_data(self, Tb: bool = False, log: bool = False,
                 drop: bool = True) -> None:
        self.data = None
        if not hasattr(self, 'hdu'):
            self.gen_hdu()
        h, d = self.hdu.header, self.hdu.data
        if drop == True: d = np.squeeze(d)
        if Tb == True: d *= Jy2K(header=h)
        if log == True: d = np.log10(d.clip(np.min(d[d > 0]), None))
        self.data = d
        
    def get_data(self, Tb: bool = False, log: bool = False,
                 drop: bool = True) -> list:
        if not hasattr(self, 'data'):
            self.gen_data(Tb=Tb, log=log, drop=drop)
        return self.data

    def gen_grid(self, center: str = None, rmax: float = 1e10,
                 xoff: float = 0., yoff: float = 0., dist: float = 1.,
                 restfrq: float = None, vsys: float = 0.,
                 vmin: float = -1e10, vmax: float = 1e10) -> None:
        if not hasattr(self, 'header'):
            self.gen_header()
        h = self.header
        if not center is None:
            cx, cy = coord2xy(center)
        else:
            cx, cy = h['CRVAL1'], h['CRVAL2']
        self.x, self.y, self.v = None, None, None
        self.dx, self.dy, self.dv = None, None, None
        if h['NAXIS'] > 0:
            if h['NAXIS1'] > 1:
                s = np.arange(h['NAXIS1'])
                s = (s-h['CRPIX1']+1) * h['CDELT1'] + h['CRVAL1'] - cx
                s *= 3600. * dist
                i0 = np.argmin(np.abs(s - (xoff - rmax)))
                i1 = np.argmin(np.abs(s - (xoff + rmax)))
                i0, i1 = sorted([i0, i1])
                s = s[i0:i1 + 1]
                self.x, self.dx = s, s[1] - s[0]
                if hasattr(self, 'data'):
                    if np.ndim(self.data) == 2:
                        self.data = self.data[:, i0:i1 + 1]
                    else:
                        self.data = self.data[:, :, i0:i1 + 1]
        if h['NAXIS'] > 1:
            if h['NAXIS2'] > 1:
                s = np.arange(h['NAXIS2'])
                s = (s-h['CRPIX2']+1) * h['CDELT2'] + h['CRVAL2'] - cy
                s *= 3600. * dist
                i0 = np.argmin(np.abs(s - (yoff - rmax)))
                i1 = np.argmin(np.abs(s - (yoff + rmax)))
                i0, i1 = sorted([i0, i1])
                s = s[i0:i1 + 1]
                self.y, self.dy = s, s[1] - s[0]
                if hasattr(self, 'data'):
                    if np.ndim(self.data) == 2:
                        self.data = self.data[i0:i1 + 1, :]
                    else:
                        self.data = self.data[:, i0:i1 + 1, :]
        if h['NAXIS'] > 2:
            if h['NAXIS3'] > 1:
                s = np.arange(h['NAXIS3'])
                s = (s-h['CRPIX3']+1) * h['CDELT3'] + h['CRVAL3']
                freq = 0
                if 'RESTFREQ' in h.keys(): freq = h['RESTFREQ']
                if 'RESTFRQ' in h.keys(): freq = h['RESTFRQ']
                if not restfrq is None: freq = restfrq
                if freq > 0:
                    s = (freq-s) / freq
                    s = s * constants.c.to('km*s**(-1)').value - vsys
                    i0 = np.argmin(np.abs(s - vmin))
                    i1 = np.argmin(np.abs(s - vmax))
                    i0, i1 = sorted([i0, i1])
                    s = s[i0:i1 + 1]
                    self.v, self.dv = s, s[1] - s[0]
                if hasattr(self, 'data'):
                    self.data = self.data[i0:i1 + 1, :, :]
                    
    def get_grid(self, center: str = None, rmax: float = 1e10,
                 xoff: float = 0., yoff: float = 0., dist: float = 1.,
                 restfrq: float = None, vsys: float = 0.,
                 vmin: float = -1e10, vmax: float = 1e10) -> None:
        if not hasattr(self, 'x') or not hasattr(self, 'y'):
            self.gen_grid(center, rmax, xoff, yoff, dist,
                          restfrq, vsys, vmin, vmax)
        if hasattr(self, 'v'):
            return [self.x, self.y, self.v]
        else:
            return [self.x, self.y, None]


def find_rms(data, sigma):
    nums = [float, int, np.float64, np.int64, np.float32, np.int32]
    if type(sigma) in nums: noise = sigma
    elif sigma == 'edge': noise = np.nanstd(data[::len(data) - 1])
    elif sigma == 'neg': noise = np.sqrt(np.nanmean(data[data < 0]**2))
    elif sigma == 'med': noise = np.sqrt(np.nanmedian(data**2) / 0.454936)
    elif sigma == 'iter':
        n = data.copy()
        for _ in range(20):
            ave, sig = np.nanmean(n), np.nanstd(n)
            n = n - ave
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
    #plt.rcParams['figure.autolayout'] = True
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
                 center: str = None, rmax: float = 100, dist: float = 1.,
                 xoff: float = 0, yoff: float = 0,
                 xflip: bool = True, yflip: bool = False) -> None:
        set_rcparams()
        self.fig = plt.figure(figsize=(7, 5)) if fig is None else fig
        self.ax = self.fig.add_subplot(1, 1, 1) if ax is None else ax
        self.gridpar = {'center':center, 'rmax':rmax,
                        'dist':dist, 'xoff':xoff, 'yoff':yoff}
        self.xflip = -1 if xflip else 1
        self.yflip = -1 if yflip else 1
        self.xedge = [xoff - self.xflip*rmax, xoff + self.xflip*rmax]
        self.yedge = [yoff - self.yflip*rmax, yoff + self.yflip*rmax]

    def __pos2xy(self, pos: list = []) -> list:
        x, y = [None] * len(pos), [None] * len(pos)
        for i, p in enumerate(pos):
            if type(p) == str:
                x[i], y[i] = coord2xy(p) - coord2xy(self.gridpar['center'])
            else:
                x[i], y[i] = rel2abs(*p, self.xedge, self.yedge)
        return [x, y]
        
    
    def add_ellipse(self, poslist: list = [],
                    majlist: list = [], minlist: list = [],
                    palist: list = [], **kwargs) -> None:
        kwargs0 = {'facecolor':'none', 'edgecolor':'gray',
                   'linewidth':1.5, 'zorder':10}
        for x, y, width, height, angle \
            in zip(*self.__pos2xy(poslist), minlist, majlist, palist):
            e = Ellipse((x, y), width=width, height=height,
                        angle=angle * self.xflip,
                        **dict(kwargs0, **kwargs))
            self.ax.add_patch(e)

    def add_beam(self, bmaj, bmin, bpa, beamcolor) -> None:                
        bpos = max(0.7 * bmaj / self.gridpar['rmax'], 0.075)
        self.add_ellipse(poslist=[[bpos, bpos]],
                         majlist=[bmaj], minlist=[bmin], palist=[bpa],
                         facecolor=beamcolor, edgecolor=None)

    def add_color(self, fitsimage: str = None,
                   x: list = None, y: list = None, skip: int = 1,
                   c: list = None,
                   cmin: float = None, cmax: float = None,
                   Tb: bool = False, log: bool = False,
                   show_cbar: bool = True,
                   clabel: str = None, cformat: float = '%.1e',
                   show_beam: bool = True, beamcolor: str = 'gray',
                   bmaj: float = 0., bmin: float = 0.,
                   bpa: float = 0., **kwargs) -> None:
        kwargs0 = {'cmap':'cubehelix', 'alpha':1, 'zorder':1}
        if not fitsimage is None:
            fd = FitsData(fitsimage)
            fd.gen_data(Tb=Tb, log=log)
            x, y, _ = fd.get_grid(**self.gridpar)
            c = fd.data
            bmaj, bmin, bpa = fd.get_beam(dist=self.gridpar['dist'])
            bunit = fd.get_header('BUNIT')
        x, y, c = x[::skip], y[::skip], c[::skip, ::skip]
        c = np.array(c)
        if not (cmin is None):
            c = c.clip(np.log10(cmin), None) if log else c.clip(cmin, None)
        if not (cmax is None):
            c = c.clip(None, np.log10(cmax)) if log else c.clip(None, cmax)
        p = self.ax.pcolormesh(x, y, c, shading='nearest',
                               **dict(kwargs0, **kwargs))
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
        kwargs0 = {'colors':'gray', 'linewidths':1.0, 'zorder':2}
        if not fitsimage is None:
            fd = FitsData(fitsimage)
            fd.gen_data(Tb=Tb, log=False)
            x, y, _ = fd.get_grid(**self.gridpar)
            c = fd.data
            bmaj, bmin, bpa = fd.get_beam(dist=self.gridpar['dist'])
        rms = find_rms(c, sigma),
        x, y, c = x[::skip], y[::skip], c[::skip, ::skip]
        self.ax.contour(x, y, c, np.array(levels) * rms,
                        **dict(kwargs0, **kwargs))
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
                   'headaxislength':0, 'width':0.007, 'zorder':3}
        if not ampfits is None:
            fd = FitsData(ampfits)
            fd.gen_data(Tb=False, log=False)
            x, y, _ = fd.get_grid(**self.gridpar)
            amp = fd.data
            bmaj, bmin, bpa = fd.get_beam(dist=self.gridpar['dist'])
        if not angfits is None:
            fd = FitsData(angfits)
            fd.gen_data(Tb=False, log=False)
            x, y, _ = fd.get_grid(**self.gridpar)
            ang = fd.data
            bmaj, bmin, bpa = fd.get_beam(dist=self.gridpar['dist'])
        if amp is None and not ang is None:
            amp = np.ones_like(ang)
        x, y = x[::skip], y[::skip]
        amp, ang = amp[::skip, ::skip], ang[::skip, ::skip]
        u = ampfactor * amp * np.sin(np.radians(ang))
        v = ampfactor * amp * np.cos(np.radians(ang))
        kwargs0['scale'] = 1. / np.abs(x[1] - x[0])
        self.ax.quiver(x, y, u, v, **dict(kwargs0, **kwargs))
        if show_beam:
            self.add_beam(bmaj, bmin, bpa, beamcolor)
            
    def add_scalebar(self, length: float = 0, label: str = '',
                     color: str = 'gray', barpos: tuple = (0.83, 0.17),
                     fontsize: float = 24, linewidth: float = 3):
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
                    'mec':'gray', 'mew':2, 'alpha':1, 'zorder':10}
        for x, y in zip(*self.__pos2xy(poslist)):
            self.ax.plot(x, y, **dict(kwsmark0, **kwargs))
            
    def add_label(self, poslist: list = [],
                  slist: list = [], **kwargs) -> None:
        kwargs0 = {'color':'gray', 'fontsize':18, 'zorder':10}
        for x, y, s in zip(*self.__pos2xy(poslist), slist):
            self.ax.text(x=x, y=y, s=s, **dict(kwargs0, **kwargs))

    def add_line(self, poslist: list = [], anglelist: list = [],
                 rlist: list = [], **kwargs):
        kwargs0 = {'color':'gray', 'linewidth':1.5, 'zorder':10}
        for x, y, a, r \
            in zip(*self.__pos2xy(poslist), np.radians(anglelist), rlist):
            self.ax.plot([x, x + r * np.sin(a)],
                         [y, y + r * np.cos(a)],
                         **dict(kwargs0, **kwargs))

    def add_arrow(self, poslist: list = [], anglelist: list = [],
                  rlist: list = [], **kwargs):
        kwargs0 = {'color':'gray', 'width':0.012,
                   'headwidth':5, 'headlength':5, 'zorder':10}
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
        self.ax.set_xlim(*self.xedge)
        self.ax.set_ylim(*self.yedge)
        if not xlabel is None: self.ax.set_xlabel(xlabel)
        if not ylabel is None: self.ax.set_ylabel(ylabel)
        self.fig.tight_layout()
       
    def savefig(self, filename: str = 'plotastro2D.png',
                transparent: bool =True) -> None:
        self.fig.patch.set_alpha(0)
        self.fig.savefig(filename, bbox_inches='tight',
                         transparent=transparent)
        
    def show(self):
        plt.show()
            
        
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
    def __init__(self, fitsimage: str = None, v: list = None,
                 nrows: int = 4, ncols: int = 6,
                 vmin: float = -1e10, vmax: float = 1e10,
                 vsys: float = 0., vskip: int = 1,
                 veldigit: int = 2,
                 center: str = None, rmax: float = 1e10,
                 dist: float = 1., xoff: float = 0, yoff: float = 0,
                 xflip: bool = True, yflip: bool = False) -> None:
        set_rcparams(fontsize=12)
        if not fitsimage is None:
            fd = FitsData(fitsimage)
            _, _, v = fd.get_grid(vsys=vsys, vmin=vmin, vmax=vmax)
        self.nv = len(v := v[::vskip])
        npages = int(np.ceil(self.nv / nrows / ncols))
        self.nchan = npages * nrows * ncols
        lennan = self.nchan - self.nv
        v = np.r_[v, v[-1] + (np.arange(lennan)+1)*(v[1]-v[0])]
        self.shape = (npages, nrows, ncols)
        ax = np.empty(self.shape, dtype='object')
        for n, i, j in np.ndindex(self.shape):
            fig = plt.figure(n, figsize=(ncols*2, max(nrows, 1.5)*2))
            sharex = ax[n, i - 1, j] if i > 0 else None
            sharey = ax[n, i, j - 1] if j > 0 else None
            ax[n, i, j] = fig.add_subplot(nrows, ncols, i*ncols + j + 1,
                                          sharex=sharex, sharey=sharey)
            fig.subplots_adjust(hspace=0, wspace=0, right=0.87, top=0.87)
            ch = n*nrows*ncols + i*ncols + j
            ax[n, i, j].text(0.9 * rmax, 0.7 * rmax,
                             rf'${v[ch]:.{veldigit:d}f}$', color='black',
                             backgroundcolor='white', zorder=20)
          
        self.ax = ax
        self.npages, self.nrows, self.ncols = npages, nrows, ncols
        self.vskip = vskip
        self.gridpar = {'center':center, 'rmax':rmax, 'dist':dist,
                        'xoff':xoff, 'yoff':yoff,
                        'vsys':vsys, 'vmin':vmin, 'vmax':vmax}
        self.xflip = -1 if xflip else 1
        self.yflip = -1 if yflip else 1
        self.xedge = [xoff - self.xflip*rmax, xoff + self.xflip*rmax]
        self.yedge = [yoff - self.yflip*rmax, yoff + self.yflip*rmax]
        self.allchan = np.arange(self.nchan)
        self.bottomleft = self.__ich(np.arange(npages), nrows-1, 0)
        
    def __ich(self, n: int, i: int, j: int) -> int:
        return n*self.nrows*self.ncols + i*self.ncols + j
        
    def __pos2xy(self, pos: list = []) -> list:
        x, y = [None] * len(pos), [None] * len(pos)
        for i, p in enumerate(pos):
            if type(p) == str:
                x[i], y[i] = coord2xy(p) - coord2xy(self.gridpar['center'])
            else:
                x[i], y[i] = rel2abs(*p, self.xedge, self.yedge)
        return [x, y]

    def add_ellipse(self, include_chan: list = None, poslist: list = [],
                    majlist: list = [], minlist: list = [],
                    palist: list = [], **kwargs) -> None:
        kwargs0 = {'facecolor':'none', 'edgecolor':'gray',
                   'linewidth':1.5, 'zorder':10}
        if include_chan is None: include_chan = self.allchan
        for x, y, width, height, angle\
            in zip(*self.__pos2xy(poslist), minlist, majlist, palist):
            for i, axnow in enumerate(np.ravel(self.ax)):
                if not (i in include_chan):
                    continue
                plt.figure(i // (self.nrows * self.ncols))
                e = Ellipse((x, y), width=width, height=height,
                            angle=angle * self.xflip,
                            **dict(kwargs0, **kwargs))
                axnow.add_patch(e)

    def add_beam(self, bmaj, bmin, bpa, beamcolor) -> None:                
        bpos = max(0.7 * bmaj / self.gridpar['rmax'], 0.075)
        self.add_ellipse(include_chan=self.bottomleft,
                         poslist=[[bpos, bpos]],
                         majlist=[bmaj], minlist=[bmin], palist=[bpa],
                         facecolor=beamcolor, edgecolor=None)

    def add_color(self, fitsimage: str = None,
                   x: list = None, y: list = None, skip: int = 1,
                   c: list = None,
                   cmin: float = None, cmax: float = None,
                   Tb: bool = False, log: bool = False,
                   show_cbar: bool = True,
                   clabel: str = None, cformat: float = '%.1e',
                   show_beam: bool = True, beamcolor: str = 'gray',
                   bmaj: float = 0., bmin: float = 0.,
                   bpa: float = 0., **kwargs) -> None:
        kwargs0 = {'cmap':'cubehelix', 'alpha':1, 'zorder':1}
        if not fitsimage is None:
            fd = FitsData(fitsimage)
            fd.gen_data(Tb=Tb, log=log)
            x, y, _ = fd.get_grid(**self.gridpar)
            c = fd.data
            bmaj, bmin, bpa = fd.get_beam(dist=self.gridpar['dist'])
            bunit = fd.get_header('BUNIT')
        c = np.array(c)
        x, y = x[::skip], y[::skip]
        if not (cmin is None):
            c = c.clip(np.log10(cmin), None) if log else c.clip(cmin, None)
        else:
            cmin = np.nanmin(c)
        if not (cmax is None):
            c = c.clip(None, np.log10(cmax)) if log else c.clip(None, cmax)
        else:
            cmax = np.nanmax(c)
        if np.ndim(c) == 3:
            c = c[::self.vskip, ::skip, ::skip]
        else:
            c = c[::skip, ::skip]
            c = np.full((self.nv, *np.shape(c)), c)
        lennan = self.nchan - len(c)
        cnan = np.full((lennan, *np.shape(c[0])), c[0] * np.nan)
        c = np.concatenate((c, cnan), axis=0)
        for axnow, cnow in zip(np.ravel(self.ax), c):
            p = axnow.pcolormesh(x, y, cnow, shading='nearest',
                                 vmin=cmin, vmax=cmax,
                                 **dict(kwargs0, **kwargs))
        #if show_cbar:
        #    clabel = bunit if clabel is None else clabel
        #    plt.colorbar(p, ax=self.ax, label=clabel, format=cformat)
        if show_beam:
            self.add_beam(bmaj, bmin, bpa, beamcolor)

    def add_contour(self, fitsimage: str = None,
                    x: list = None, y: list = None, skip: int = 1,
                    v: list = None, c: list = None,
                    sigma: str or float = 'edge',
                    levels: list = [-12,-6,-3,3,6,12,24,48,96,192,384],
                    Tb: bool = False,
                    show_beam: bool = True, beamcolor: str = 'gray',
                    bmaj: float = 0., bmin: float = 0., bpa: float = 0.,
                    **kwargs) -> None:
        kwargs0 = {'colors':'gray', 'linewidths':1.0, 'zorder':2}
        if not fitsimage is None:
            fd = FitsData(fitsimage)
            fd.gen_data(Tb=Tb, log=False)
            x, y, _ = fd.get_grid(**self.gridpar)
            c = fd.data
            bmaj, bmin, bpa = fd.get_beam(dist=self.gridpar['dist'])
        c = np.array(c)
        x, y = x[::skip], y[::skip]
        if np.ndim(c) == 3:
            rms = find_rms(c, sigma)
            c = c[::self.vskip, ::skip, ::skip]
        else:
            rms = find_rms(c, 'neg' if sigma == 'edge' else sigma)
            c = c[::skip, ::skip]
            c = np.full((self.nv, *np.shape(c)), c)
        lennan = self.nchan - len(c)
        cnan = np.full((lennan, *np.shape(c[0])), c[0] * np.nan)
        c = np.concatenate((c, cnan), axis=0)        
        for axnow, cnow in zip(np.ravel(self.ax), c):
            axnow.contour(x, y, cnow, np.array(levels) * rms,
                          **dict(kwargs0, **kwargs))
        if show_beam:
            self.add_beam(bmaj, bmin, bpa, beamcolor)
            
    def add_scalebar(self, length: float = 0, label: str = '',
                     color: str = 'gray', barpos: tuple = (0.75, 0.17),
                     fontsize: float = 15, linewidth: float = 3):
        if length == 0 or label == '':
            print('Please set length and label.')
            return -1
        for n in range(self.npages):
            axnow = self.ax[n, self.nrows - 1, 0]
            a, b = barpos
            x0, y0 = rel2abs(a, b * 0.9, self.xedge, self.yedge)
            axnow.text(x0, y0, label, color=color, size=fontsize,
                       ha='center', va='top', zorder=10)
            x0, y0 = rel2abs(a, b, self.xedge, self.yedge)
            axnow.plot([x0 - length/2., x0 + length/2.], [y0, y0],
                       '-', linewidth=linewidth, color=color)
            
    def add_marker(self, include_chan: list = None,
                   poslist: list = [], **kwargs):
        kwsmark0 = {'marker':'+', 'ms':10, 'mfc':'gray',
                    'mec':'gray', 'mew':2, 'alpha':1}
        if include_chan is None: include_chan = self.allchan
        xlist, ylist = self.__pos2xy(poslist)
        for i, axnow in enumerate(np.ravel(self.ax)):
            if not (i in include_chan):
                continue
            for x, y in zip(xlist, ylist):
                axnow.plot(x, y, **dict(kwsmark0, **kwargs), zorder=10)
            
    def add_label(self, include_chan: list = None,
                  poslist: list = [], slist: list = [], **kwargs) -> None:
        kwargs0 = {'color':'gray', 'fontsize':15, 'zorder':10}
        if include_chan is None: include_chan = self.allchan
        xlist, ylist = self.__pos2xy(poslist)
        for i, axnow in enumerate(np.ravel(self.ax)):
            if not (i in include_chan):
                continue
            for x, y, s in zip(xlist, ylist, slist):
                axnow.text(x=x, y=y, s=s, **dict(kwargs0, **kwargs))

    def add_line(self, include_chan: list = None,
                 poslist: list = [], anglelist: list = [],
                 rlist: list = [], **kwargs):
        kwargs0 = {'color':'gray', 'linewidth':1.5, 'zorder':10}
        if include_chan is None: include_chan = self.allchan
        xlist, ylist = self.__pos2xy(poslist)
        for i, axnow in enumerate(np.ravel(self.ax)):
            if not (i in include_chan):
                continue
            for x, y, a, r \
                in zip(xlist, ylist, np.radians(anglelist), rlist):
                axnow.plot([x, x + r * np.sin(a)],
                           [y, y + r * np.cos(a)],
                           **dict(kwargs0, **kwargs))

    def add_arrow(self, include_chan: list = None,
                  poslist: list = [], anglelist: list = [],
                  rlist: list = [], **kwargs):
        kwargs0 = {'color':'gray', 'width':0.012,
                   'headwidth':5, 'headlength':5, 'zorder':10}
        if include_chan is None: include_chan = self.allchan
        xlist, ylist = self.__pos2xy(poslist)
        for i, axnow in enumerate(np.ravel(self.ax)):
            if not (i in include_chan):
                continue
            for x, y, a, r \
                in zip(xlist, ylist, np.radians(anglelist), rlist):
                axnow.quiver(x, y, r * np.sin(a), r * np.cos(a),
                             angles='xy', scale_units='xy', scale=1,
                             **dict(kwargs0, **kwargs))
                
    def set_axis(self, xticks: list = None, yticks: list = None,
                 xticksminor: list = None, yticksminor: list = None,
                 xticklabels: list = None, yticklabels: list= None,
                 xlabel: str = 'R.A. (arcsec)',
                 ylabel: str = 'Dec. (arcsec)',
                 samexy: bool = True) -> None:
        for n, i, j in np.ndindex(self.shape):
            axnow = self.ax[n, i, j]
            ch = self.__ich(n, i, j)
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
            if not xticklabels is None: axnow.set_xticklabels(xticklabels)
            if not (ch in self.bottomleft):
                axnow.set_xticklabels([''] * len(axnow.get_xticks()))
            if not yticklabels is None: axnow.set_yticklabels(yticklabels)
            if not (ch in self.bottomleft):
                axnow.set_yticklabels([''] * len(axnow.get_yticks()))
            axnow.set_xlim(*self.xedge)
            axnow.set_ylim(*self.yedge)
            if not xlabel is None: axnow.set_xlabel(xlabel)
            if not (ch in self.bottomleft):
                axnow.set_xlabel('')
            if not ylabel is None: axnow.set_ylabel(ylabel)
            if not (ch in self.bottomleft):
                axnow.set_ylabel('')

    def savefig(self, filename: str = 'plotastro3D.png',
                transparent: bool =True) -> None:
        ext = filename.split('.')[-1]
        for i in range(self.npages):
            fig = plt.figure(i)
            fig.patch.set_alpha(0)
            fig.savefig(filename.replace('.' + ext, f'_{i:d}.' + ext),
                        bbox_inches='tight', transparent=transparent)
    def show(self):
        plt.show()
