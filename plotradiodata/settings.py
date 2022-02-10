
import matplotlib as mpl
from astropy.io import fits
from matplotlib.patches import Ellipse
from PIL import Image

from other_utils import listing, rel2abs, coord2xy



def vec_uv(vec0array, vec1array, vectype, vecrot):
    if type(vectype) is str: vectype = [vectype] * len(vec1array)
    if type(vecrot) in [float, int]: vecrot = [vecrot] * len(vec1array)
    d_u, d_v = [], []
    for d0, d1, t, r in zip(vec0array, vec1array, vectype, vecrot):
        d = d0 * np.exp(1j * d1 * deg) if t == 'ampang' else d1 + 1j * d0
        d = d * np.exp(1j * r * deg)
        d0, d1 = np.imag(d), np.real(d)
        d_u.append(d0)
        d_v.append(d1)
    return [np.array(d_u), np.array(d_v)]


class DataForPlot:
    def __init__(self, lenv=1):
        self.lenv = lenv

    def reshaping(self, dd, m):
        if dd is None:
            return [None, 0]
        if len(dd) == 0:
            return [[], 0]

        d = [np.squeeze(a) for a in dd]
        ndim = len(np.shape(d[0])) + 1
        if self.lenv < 2:
            if ndim == 3:
                return [np.array(d), len(d)]
            elif ndim == 2:
                return [np.array([d]), 1]
            else:
                print(f'### Error: array for {m} is {ndim:d}D'
                      + ' without a velocity asix.')
                return 0
        else:
            if ndim == 4:
                return [np.array(d), len(d)]
            elif ndim == 3:
                if len(d) == self.lenv:
                    return [np.array([d]), 1]
                else:
                    a = np.full((self.lenv, *np.shape(d)), d)
                    a = np.moveaxis(a, 0, 1)
                    return [a, len(a)]
            elif ndim == 2:
                return [np.full((self.lenv, *np.shape(d)), d), 1]
            else:
                print(f'### Error: array for {m} is {ndim:d}D.')
                return 0

    def gen_color(self, d):
        self.color, self.ncol = self.reshaping(d, 'color')

    def gen_contour(self, d):
        self.contour, self.ncon = self.reshaping(d, 'contour')

    def gen_vector(self, d0, d1, vectype='nochange', vecrot=0):
        self.vec0, self.nvec0 = self.reshaping(d0, 'vec0')
        self.vec1, self.nvec1 = self.reshaping(d1, 'vec1')
        if self.nvec0 == 0 and self.nvec1 > 0:
            self.vec0 = self.vec1 * 0 + 1
        self.vec0, self.vec1 = vec_uv(self.vec0, self.vec1, vectype, vecrot)
        self.nvec = self.nvec1

    def gen_grid(self, x=None, y=None, v=None):
        if not (x is None): self.x = x
        if not (y is None): self.y = y
        if not (v is None): self.v = v


def set_rcparams(fontsize=24, nancolor='w', direction='out'):
    #plt.rcParams['font.family'] = 'arial'
    plt.rcParams['axes.facecolor'] = nancolor
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['legend.fontsize'] = 15
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.direction'] = direction
    plt.rcParams['ytick.direction'] = direction
    plt.rcParams['xtick.major.size'] = 8
    plt.rcParams['ytick.major.size'] = 8
    plt.rcParams['xtick.minor.size'] = 4
    plt.rcParams['ytick.minor.size'] = 4
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['xtick.minor.width'] = 1.5
    plt.rcParams['ytick.minor.width'] = 1.5


def set_grid(col, con, vec, x, y, v=None, chan=False):
    if len(np.shape(col)) == 2 and (not chan): col = [col]
    if len(np.shape(con)) == 2 and (not chan): con = [con]
    if len(np.shape(vec)) == 2 and (not chan): vec = [vec]
    if len(col) > 0: vyx = np.shape(col[0] if chan else [col[0]])
    elif len(con) > 0: vyx = np.shape(con[0] if chan else [con[0]])
    elif len(vec) > 0: vyx = np.shape(vec[0] if chan else [vec[0]])
    lenv, leny, lenx = vyx
    if x is None: x = np.arange(lenx)
    if y is None: y = np.arange(leny)
    x = np.array(x)
    y = np.array(y)
    if chan:
        if v is None: v = np.arange(lenv)
        v = np.array(v)
        return [x, y, v]
    else:
        return [x, y]


def sub_grid(d, center, rmax, vmin, vmax, nskipx, nskipv, chan=False):
    x, y = d.x - center[0], d.y - center[1]
    if rmax == 0:
        i0, i1, j0, j1 = 0, len(x), 0, len(y)
    else:
        x_bl, y_bl, x_tr, y_tr = rmax, -rmax, -rmax, rmax
        i0, i1 = np.argmin(np.abs(x - x_bl)), np.argmin(np.abs(x - x_tr)) + 1
        j0, j1 = np.argmin(np.abs(y - y_bl)), np.argmin(np.abs(y - y_tr)) + 1
        i0, j0 = i0 + nskipx // 2, j0 + nskipx // 2
    i2, j2 = nskipx, nskipx 
    d.x, d.y = x[i0:i1:i2], y[j0:j1:j2]
    iskip = [i0, i1, i2, j0, j1, j2, None, None, None]
    if chan:
        v = d.v - center[2]
        k0, k1 = np.argmin(np.abs(v - vmin)), np.argmin(np.abs(v - vmax)) + 1
        k2 = (1 if k0 < k1 else -1) * nskipv
        d.v = v[k0:k1:k2]
        iskip[6:] = [k0, k1, k2]
    return [d, iskip]


def skipping(d, iskip, vecskip):
    i0, i1, i2, j0, j1, j2, k0, k1, k2 = iskip
    v0, v2 = vecskip // 2, vecskip
    xvec, yvec = d.x[v0::v2], d.y[v0::v2]
    if k0 is None:
        if d.ncol > 0:
            d.color = d.color[:, j0:j1:j2, i0:i1:i2]
        if d.ncon > 0:
            d.contour = d.contour[:, j0:j1:j2, i0:i1:i2]
        if d.nvec > 0:
            d.vec0 = d.vec0[:, j0:j1:j2, i0:i1:i2]
            d.vec1 = d.vec1[:, j0:j1:j2, i0:i1:i2]
            d.vec0 = d.vec0[:, v0::v2, v0::v2]
            d.vec1 = d.vec1[:, v0::v2, v0::v2]
    else:
        if d.ncol > 0:
            d.color = d.color[:, k0:k1:k2, j0:j1:j2, i0:i1:i2]
        if d.ncon > 0:
            d.contour = d.contour[:, k0:k1:k2, j0:j1:j2, i0:i1:i2]
        if d.nvec > 0:
            d.vec0 = d.vec0[:, k0:k1:k2, j0:j1:j2, i0:i1:i2]
            d.vec1 = d.vec1[:, k0:k1:k2, j0:j1:j2, i0:i1:i2]
            d.vec0 = d.vec0[:, :, v0::v2, v0::v2]
            d.vec1 = d.vec1[:, :, v0::v2, v0::v2]
    return d, xvec, yvec


def find_rms(d_contour, rms):
    noise = []
    nums = [float, int, np.float64, np.int64, np.float32, np.int32]
    for d, r in zip(d_contour, rms):
        if type(r) in nums: n = r
        elif r == 'edge': n = np.nanstd(d[::len(d) - 1])
        elif r == 'neg': n = np.sqrt(np.nanmean(d[d < 0]**2))
        elif r == 'med': n = np.sqrt(np.nanmedian(d**2) / 0.454936)
        elif r == 'iter':
            n = d
            for i in range(20):
                ave, sig = np.nanmean(n), np.nanstd(n)
                n = n - np.nanmean(n)
                n = n[np.abs(n) < 3.5 * sig]
            n = np.nanstd(n)
        elif r == 'out':
            n, n0, n1 = d.copy(), len(d), len(d[0])
            n[n0 // 5:n0 * 4 // 5, n1 // 5:n1 * 4 // 5] = np.nan
            n = np.nanstd(n)
        noise.append(n)
        print(f'rms = {n:.4e}')
    return noise


def set_color(d, cmin, cmax, logcolor, bunit):
    sifn = [int, float, np.float64, np.int64, str, type(None)]
    if type(cmin)  in sifn: cmin  = [cmin]  * d.ncol
    if type(cmax)  in sifn: cmax  = [cmax]  * d.ncol
    if type(bunit) in sifn: bunit = [bunit] * d.ncol
    for i in range(d.ncol):
        if cmin[i] is None:
            a = d.color[i]
            if logcolor: cmin[i] = np.sqrt(np.mean(a[a < 0]**2))
            else: cmin[i] = np.nanmin(a)
        if cmax[i] is None:
            cmax[i] = np.nanmax(d.color[i])
        if logcolor:
            d.color[i] = np.log10(d.color[i].clip(cmin[i], None))
            cmin[i], cmax[i] = np.log10(cmin[i]), np.log10(cmax[i])
    return [d, cmin, cmax, bunit]


def set_rgb(rarray, garray, barray, logcolor):
    rarray = np.array(rarray)
    garray = np.array(garray)
    barray = np.array(barray)
    if len(rarray) > 0: size = np.shape(rarray[0])
    elif len(garray) > 0: size = np.shape(garray[0])
    elif len(barray) > 0: size = np.shape(barray[0])
    else: return

    if logcolor:
        if len(rarray) == 0:
            rarray, rrms = np.ones(size), 1
        else:
            rarray = rarray[0]
            rrms = np.nanstd(rarray[rarray < 0]) * 3
        if len(garray) == 0:
            garray, grms = np.ones(size), 1
        else:
            garray = garray[0]
            grms = np.nanstd(garray[garray < 0]) * 3
        if len(barray) == 0:
            barray, brms = np.ones(size), 1
        else:
            barray = barray[0]
            brms = np.nanstd(barray[barray < 0]) * 3
        for a, b in zip([rarray, garray, barray], [rrms, grms, brms]):
            a[:, :] = np.log(a.clip(b, None) / b)
    else:
        if len(rarray) == 0: rarray = np.zeros(size)
        else: rarray = rarray[0]
        if len(garray) == 0: garray = np.zeros(size)
        else: garray = garray[0]
        if len(barray) == 0: barray = np.zeros(size)
        else: barray = barray[0]
        for a in [rarray, garray, barray]:
            a[:, :] = a.clip(0, None)
    for a in [rarray, garray, barray]:
        m = 1 if np.nanmax(a) == 0 else np.nanmax(a)
        a[:, :] = a / m * 255
    return [rarray, garray, barray]


def set_contour(d, colors, linewidths, rms, levels):
    if d.ncon == 0:
        return [[], [], [], []]
    if colors == []: colors = 'gray'
    if type(colors) is str: colors = [colors] * d.ncon
    if type(linewidths) in [float, int, np.float64, np.int64]:
        linewidths = [linewidths] * d.ncon
    if type(rms) in [float, int, np.float64, np.int64, str]:
        rms = [rms] * d.ncon
    if len(levels) > 0:
        if type(levels[0]) is list or isinstance(levels[0], np.ndarray):
            levels = [np.array(l) for l in levels]
        else:
            levels = [np.array(levels)] * d.ncon
    rms = find_rms(d.contour, rms)
    return [colors, linewidths, rms, levels]


def set_vector(d, kwsvec, vecskip):
    kwsvec0 = {'angles':'xy', 'scale_units':'xy', 'color':'gray',
               'scale':1 / vecskip / np.abs(d.x[1] - d.x[0]), 'pivot':'mid',
               'headwidth':0, 'headlength':0, 'headaxislength':0, 'width':0.007}
    if type(kwsvec) is dict: kwsvec = [kwsvec] * d.nvec
    kwsvec = [dict(kwsvec0, **v) for v in kwsvec]
    return kwsvec


def set_axes(ax, d, xticks, yticks, xticksminor, yticksminor,
             xticklabels, yticklabels, xlabel, ylabel, sameaxes, rotation):
    ax.set_xlim(d.x[0], d.x[-1])
    ax.set_ylim(d.y[0], d.y[-1])
    if sameaxes:
        ax.set_yticks(ax.get_xticks())
        ax.set_xlim(d.x[0], d.x[-1])
        ax.set_ylim(d.y[0], d.y[-1])
    if len(xticks) > 0: ax.set_xticks(xticks)
    if len(yticks) > 0: ax.set_yticks(yticks)
    if len(xticksminor) > 0: ax.set_xticks(xticksminor, minor=True)
    if len(yticksminor) > 0: ax.set_yticks(yticksminor, minor=True)
    if len(xticklabels) > 0: ax.set_xticklabels(xticklabels, rotation=rotation)
    if len(yticklabels) > 0: ax.set_yticklabels(yticklabels)
    if xlabel != '': ax.set_xlabel(xlabel)
    if ylabel != '': ax.set_ylabel(ylabel)
    if sameaxes:
        ax.set_xticks(ax.get_yticks())
        ax.set_yticks(ax.get_xticks())
        ax.set_aspect(1, adjustable='box')


def set_colorbar(cb, cbticks, cbticklabels, logcolor, chan=False):
    for c in cb:
        if c is None: continue
        if chan:
            c.ax.tick_params(labelsize=12)
            font = mpl.font_manager.FontProperties(size=12)
        else:
            c.ax.tick_params(labelsize=32)
            font = mpl.font_manager.FontProperties(size=32)
        c.ax.yaxis.label.set_font_properties(font)
        if len(cbticks) > 0:
            c.set_ticks(np.log10(cbticks) if logcolor else cbticks)
        if len(cbticklabels) > 0:
            c.set_ticklabels(cbticklabels)
        elif logcolor:
            t = c.get_ticks()
            c.set_ticks(t)
            c.set_ticklabels([f'{d:.1e}' for d in 10**t])


def set_markpos(markpos, markrel, center, afits, bfits, cfits):
    if not markrel:
        if center == '':
            afits, bfits, cfits = listing(afits, bfits, cfits)
            if afits != '': fitsname = afits[0]
            elif bfits != '': fitsname = bfits[0]
            elif cfits != '': fitsname = cfits[0]
            h = fits.open(fitsname)[0].header
            c = np.array([h['CRVAL1'], h['CRVAL2']])
        else:
            c = coord2xy(center)
        mx, my = np.array([np.array(coord2xy(m)) - c for m in markpos]).T
        mx, my = mx * 3600 * np.cos(my * deg), my * 3600
        markpos = np.array([mx, my]).T
    return markpos


def show_color(ax, d, cmap, cmin, cmax, bunit, cbformat, alpha, ch):
    cmap, cmin, cmax = listing(cmap, cmin, cmax)
    bunit, cbformat = listing(bunit, cbformat)
    if len(d.color) == 0:
        return [[None], [None]]
    else:
        a = d.color if ch is None else [d.color[0][ch]]
        pcol, alp = [], alpha
        for i in range(len(a)):
            if len(a) > 1:
                a[i][a[i] < cmin[i]] = None
                alp = 1. / len(a) if alpha == 1 else alpha[i]
            p = ax.pcolormesh(d.x, d.y, a[i], shading='nearest',
                              cmap=cmap[i], vmin=cmin[i], vmax=cmax[i],
                              alpha=alp, zorder=1)
            pcol.append(p)
        cb = []
        if ch is None:
            for p, l, f in zip(pcol, bunit, cbformat):
                if l is None: c = None
                else: c = plt.colorbar(p, ax=ax, label=l, format=f)
                cb.append(c)
        return [cb, pcol]

def show_rgb(ax, d):
    size = np.shape(d.color[0])
    im = Image.new('RGB', size, (128, 128, 128))
    for a in d.color:
        if d.x[1] > d.x[0]: a[:, :] = a[:, ::-1]
        if d.y[1] > d.y[0]: a[:, :] = a[::-1, :]
    for j in range(size[0]):
        for i in range(size[1]):
            xy = (i, j)
            value = tuple(int(a[j, i]) for a in d.color)
            im.putpixel(xy, value)
    ax.imshow(im, extent=[d.x[0], d.x[-1], d.y[0], d.y[-1]])


def show_contour(ax, d, levels, rms, colors, linewidths, ch):
    for a, l, r, c, w in zip(d.contour, levels, rms, colors, linewidths):
        if not (ch is None): a = a[ch]
        ax.contour(d.x, d.y, a, l * r, colors=c, linewidths=w, zorder=2)


def show_vectors(ax, d, kwsvec, xvec, yvec, ch):
    for a0, a1, ve in zip(d.vec0, d.vec1, kwsvec):
        if not (ch is None): a0, a1 = a0[ch], a1[ch]
        ax.quiver(xvec, yvec, a0, a1, **ve, zorder=3)


def show_lines(ax, line_c, line_a, line_r, kwsline, rmax):
    if len(line_a) > 0 and len(line_r) == 0: line_r = [3. * rmax] * len(line_a)
    if len(line_a) > 0 and len(line_c) == 0: line_c = [''] * len(line_a)
    if type(kwsline) is dict: kwsline = [kwsline] * len(line_a)
    kwsline = [dict({'color':'gray'}, **k) for k in kwsline]
    for c, a, r, k in zip(line_c, np.array(line_a) * deg, line_r, kwsline):
        xc, yc = [0, 0] if c == '' else coord2xy(c)
        ax.plot([xc, xc + r * np.sin(a)], [yc, yc + r * np.cos(a)], **k)


def show_markers(ax, center, markpos, kwsmark):
    kwsmark0 = {'marker':'+', 'ms':50, 'mfc':'gray',
                'mec':'gray', 'mew':3, 'alpha':1}
    markpos = np.array(markpos) - np.array([center] * len(markpos))
    for m, k in zip(markpos, kwsmark):
        ax.plot(m[0], m[1], **dict(kwsmark0, **k))


def add_beam(ax, d, bmaj, bmin, bpa, beamcolor):
    bmaj, bmin, bpa, beamcolor = listing(bmaj, bmin, bpa, beamcolor)
    for w, h, a, c in zip(bmin, bmaj, bpa, beamcolor):
        if c is None: continue
        bpos = max(1.4 * h / 2. / np.abs(d.x[-1] - d.x[0]), 0.075)
        a *= np.sign(d.x[1] - d.x[0])
        e = Ellipse(rel2abs(bpos, bpos, d.x, d.y),
                    width=w, height=h, angle=a, facecolor=c, zorder=4)
        ax.add_patch(e)


def add_label(ax, d, labels, k=None):
    labels = listing(labels)
    if len(d.v) > 0:
        dv = np.abs(d.v[1] - d.v[0]) / 2.
        for l in labels:
            vshow = d.v[k] if l['v'] == 'all' else l['v']
            if l['s'] != '' and np.abs(d.v[k] - vshow) < dv:
                l['x'], l['y'] = rel2abs(l['x'], l['y'], d.x, d.y)
                ax.text(**l, ha='center', va='center', zorder=5)
    else:
        for l in labels:
            if l['s'] != '':
                l['x'], l['y'] = rel2abs(l['x'], l['y'], d.x, d.y)
                ax.text(**l, zorder=5)


def add_scalebar(ax, d, scalebar):
    a = {'length':0, 'label':'', 'color':'gray',
         'corner':'bottom right', 'fontsize':32, 'linewidth':3}
    if len(d.v) > 0: a['fontsize'], a['linewidth'] = 12, 2
    scalebar = dict(a, **scalebar)
    if scalebar['length'] > 0 and scalebar['label'] != '':
        barpos = {'bottom right':[0.83, 0.17], 'bottom left':[0.17, 0.12],
                  'top right':[0.83, 0.88], 'top left':[0.17, 0.88],
                  'top':[0.50, 0.88], 'left':[0.17, 0.50],
                  'right':[0.83, 0.50], 'bottom':[0.50, 0.12]}
        if len(d.v) > 0:
            barpos = {'bottom right':[0.75, 0.15], 'bottom left':[0.17, 0.12],
                      'top right':[0.83, 0.88], 'top left':[0.17, 0.88],
                      'top':[0.50, 0.88], 'left':[0.17, 0.50],
                      'right':[0.83, 0.50], 'bottom':[0.50, 0.12]}
        a, b = barpos[scalebar['corner']]
        x, y = rel2abs(a, b * 0.9, d.x, d.y)
        ax.text(x, y, scalebar['label'], color=scalebar['color'],
                size=scalebar['fontsize'], ha='center', va='top', zorder=5)
        x, y = rel2abs(a, b, d.x, d.y)
        ax.plot([x - scalebar['length'] / 2., x + scalebar['length'] / 2.],
                [y, y], '-', lw=scalebar['linewidth'], color=scalebar['color'])
