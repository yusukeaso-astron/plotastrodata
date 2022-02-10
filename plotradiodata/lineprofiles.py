from astropy.io import fits
from scipy.optimize import curve_fit

from other_utils import coord2xy



def lineprofile(fitsimage='', coords=[], radius='point', unit='K',
                vmin=None, vmax=None, fmin=None, fmax=None, title=[],
                xticks=[], yticks=[], xticklabels=[], yticklabels=[],
                xticksminor=[], yticksminor=[], color='k',
                xlabel=r'Velocity (km s$^{-1}$)', ylabel=[], text=[],
                cunit3='', savefig='', show=True, gfit=False, width=1):

    set_rcParams(20, 'w')
    
    if fits != '':
        f = fits.open(fitsimage)[0]
        d, h = f.data, f.header
        if h['NAXIS'] == 4: d = d[0]
    
    crpix = [h['CRPIX' + s] for s in ['1', '2', '3']]
    crval = [h['CRVAL' + s] for s in ['1', '2', '3']]
    cdelt = [h['CDELT' + s] for s in ['1', '2', '3']]
    naxis = [h['NAXIS' + s] for s in ['1', '2', '3']]
    
    x = crval[0] + (np.arange(naxis[0]) - crpix[0] + 1) * cdelt[0]
    y = crval[1] + (np.arange(naxis[1]) - crpix[1] + 1) * cdelt[1]
    v = crval[2] + (np.arange(naxis[2]) - crpix[2] + 1) * cdelt[2]
    x, y = np.meshgrid(x, y)
    if cunit3 != '': h['CUNIT3'] = cunit3
    if h['CUNIT3'] == 'Hz': v = (1 - v / h['RESTFRQ']) * cc / 1e3
    if h['CUNIT3'] == 'M/S' or h['CUNIT3'] == 'm/s': v = v / 1e3

    if radius == 'point': radius = (y[1, 0] - y[0, 0]) / arcsec
    if vmin is None: vmin = np.min(v)
    if vmax is None: vmax = np.max(v)    

    p = []
    for xc, yc in zip(coord2xy(coords)):
        r = np.hypot(x - xc, y - yc)
        cnd = (r < radius * arcsec)
        pp = np.array([np.mean(dd[cnd]) for dd in d])
        p.append(pp)
    p = np.array(p)
    if unit == 'K': p *= Jy2K(h)
    if unit == 'mK': p *= Jy2K(h) * 1000
    if unit == 'mJy': p *= 1000

    if width > 1:
        width = int(width)
        newlen = len(v) // width
        w, q = np.zeros(newlen), np.zeros((len(p), newlen))
        for i in range(width):
            w += v[i:i + newlen * width:width]
            q += p[:, i:i + newlen * width:width]
        v, p = w / width, q / width
    p = np.array([q[(vmin <= v) * (v <= vmax)] for q in p])
    v = v[(vmin <= v) * (v <= vmax)]

    def gauss(x, p, c, w):
        return p / np.exp(4 * np.log(2) * ((x - c) / w)**2)

    if fmin is None: fmin = np.nanmin(p)
    if fmax is None: fmax = np.nanmax(p)
    fig, ax = plt.subplots(nrows=len(p), ncols=1, sharex=True,
                           figsize=(6, 3 * len(p)))
    if len(p) == 1: ax = [ax]
    if gfit:
        vstart = vmin + (vmax - vmin) / 4.
        vend = vmax - (vmax - vmin) / 4.
        bounds = [[0, vstart, v[1] - v[0]], [fmax, vend, vmax - vmin]]

    if type(ylabel) == str: ylabel = [ylabel] * len(p)
    for i in range(len(p)):
        if gfit:
            popt, pcov = curve_fit(gauss, v, p[i], bounds=bounds)
            #ax[i].plot(v, gauss(v, *popt), drawstyle='steps-mid', color='r')
            ax[i].plot(v, gauss(v, *popt), drawstyle='default', color='g')
            print(popt)
        ax[i].plot(v, p[i], drawstyle='steps-mid', color=color)
        if i == len(p) - 1: ax[i].set_xlabel(xlabel)
        if len(ylabel) == 0:
            if unit == 'K': ax[i].set_ylabel(r'$T_b$ (K)')
            if unit == 'mK': ax[i].set_ylabel(r'$T_b$ (mK)')
            if unit == 'Jy': ax[i].set_ylabel('Flux (Jy)')
            if unit == 'Jy/beam': ax[i].set_ylabel(r'Mean (Jy beam$^{-1}$)')
            if not (unit in ['K', 'mK', 'Jy', 'Jy/beam']):
                ax[i].set_ylabel(unit)
        else:
            ax[i].set_ylabel(ylabel[i])
        ax[i].set_xlim(vmin, vmax)
        ax[i].set_ylim(fmin, fmax)
        if len(text) > 0:
            for t in text[i]: ax[i].text(**t)
        if len(xticks) > 0: ax[i].set_xticks(xticks)
        if len(yticks) > 0: ax[i].set_yticks(yticks)
        if len(xticklabels) > 0: ax[i].set_xticklabels(xticklabels)
        if len(yticklabels) > 0: ax[i].set_yticklabels(yticklabels)
        if len(xticksminor) > 0: ax[i].set_xticks(xticksminor, minor=True)
        if len(yticksminor) > 0: ax[i].set_yticks(yticksminor, minor=True)
        if len(title) > 0: ax[i].set_title(**title[i])
        ax[i].hlines([0], vmin, vmax, linestyle='dashed', color='k')

    plt.tight_layout()
    if savefig != '':
        plt.savefig(savefig, bbox_inches='tight', transparent=True)
    if show: plt.show()
    plt.close()    
