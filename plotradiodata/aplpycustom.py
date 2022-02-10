import matplotlib as mpl
from astropy.io import fits
import aplpy

from other_utils import listing, coord2xy
from fits_utils import fits2data, hdu4aplpy



def aplpycustom(colorfits='', Tbcolor=False, cfactor=1, bunit='',
             cbticks=[], kwscbar={}, nancolor='white', 
             kwscolor={'vmin':None, 'vmax':None,
                       'stretch':'linear', 'cmap':'jet'},
             contourfits=[],  rms='neg', colors='gray',
             linewidths=1.8, Tbcontour=False,
             levels=[-12,-6,-3,3,6,12,24,48,96,192,384,768,1536,3072],
             vecampfits=[], vecangfits=[],
             kwsvec={'step':1, 'scale':1, 'rotate':0, 'color':'gray'},
             center='', rmax=0, xoff=0, yoff=0,
             labels=[{'x':0.1, 'y':0.1, 'text':'', 'relative':True,
                      'color':'gray', 'ha':'center', 'va':'center'}],
             scalebar={'length':0, 'label':'', 'color':'gray',
                       'corner':'bottom right', 'fontsize':32, 'linewidth':3},
             title={}, beamcolor='gray', beampos='bottom left',
             markpos=[], markrel=True,
             kwsmark={'marker':'+', 'facecolor':'gray', 'edgecolor':None,
                      's':1000, 'linewidths':3},
             line_c=[], line_a=[], line_r=[], kwsline={'color':'gray'},
             circle_c=[], circle_r=[], kwscircle=[{'color':'gray'}],
             xlabel='', ylabel='', xspacing=None, yspacing=None,
             savefig='', show=True):

    a = {'vmin':None, 'vmax':None, 'stretch':'linear', 'cmap':'jet'}
    kwscolor = dict(a, **kwscolor)
    kwscolor['cmap'] = copy.copy(mpl.cm.get_cmap(kwscolor['cmap']))
    a = {'length':0, 'label':'', 'color':'gray',
         'corner':'bottom right', 'fontsize':32, 'linewidth':3}
    scalebar = dict(a, **scalebar)
    if type(title) is str: title = {'title':title}
    kwsmark0 = {'marker':'+', 'facecolor':'gray', 'edgecolor':None,
                's':1000, 'linewidths':3}
    if type(kwsline) is dict: kwsline = [kwsline] * len(line_a)
    kwsline = [dict({'color':'gray'}, **k) for k in kwsline]

    set_rcparams(24, nancolor, 'out')

    colorfits, contourfits, vecampfits, vecangfits = \
        listing(colorfits, contourfits, vecampfits, vecangfits)

    colorarray,   _, _ = fits2data(colorfits,   Tbcolor)
    contourarray, _, _ = fits2data(contourfits, Tbcontour)
    vec1array,    _, _ = fits2data(vecangfits,  False)
    vec0array,    _, _ = fits2data(vecampfits,  False)

    d = DataForPlot(1)
    d.gen_color(colorarray)
    d.gen_contour(contourarray)
    d.gen_vector(vec0array, vec1array)

    colors, linewidths, rms, levels = \
        set_contour(d, colors, linewidths, rms, levels)
    if type(kwsvec) is dict: kwsvec = [kwsvec] * d.nvec
    kwsvec = [dict({'color':'gray'}, **k) for k in kwsvec]

    c_color = hdu4aplpy(colorfits, d.color)
    c_contour = hdu4aplpy(contourfits, d.contour)
    if len(vecampfits) == 0 and len(vecangfits) > 0: vecampfits = vecangfits
    c_vecamp = hdu4aplpy(vecampfits, d.vec0)
    c_vecang = hdu4aplpy(vecangfits, d.vec1)

    if len(colorfits) == 0:
        fig = aplpy.FITSFigure(c_contour[0], figsize=(9, 9))
    else:
        fig = aplpy.FITSFigure(c_color[0], figsize=(12, 9))
        d = c_color[0].data
        h = c_color[0].header
        if Tbcolor:
            h['BUNIT'] = h['BUNIT'].replace('JY/BEAM', 'K')
            h['BUNIT'] = h['BUNIT'].replace('Jy/beam', 'K')
        if cfactor != 1:
            d[:, :] = d * cfactor
            h['BUNIT'] += ' / {:.2e}'.format(cfactor)
        if kwscolor['vmin'] is None:
            if kwscolor['stretch'] == 'linear':
                kwscolor['vmin'] = np.nanmin(d)
            elif kwscolor['stretch'] == 'log':
                kwscolor['vmin'] = np.sqrt(np.nanmean(d[d < 0]**2))
        if kwscolor['vmax'] is None: kwscolor['vmax'] = np.nanmax(d)
        d[:, :] = d.clip(kwscolor['vmin'], None)
        fig.show_colorscale(**kwscolor)
        if bunit == '': bunit = h['BUNIT']
        if not (kwscbar is None):
            fig.add_colorbar(**kwscbar)
            if bunit != '': fig.colorbar.set_axis_label_text(bunit)
            if len(cbticks) > 0: fig.colorbar.set_ticks(cbticks)
            fig.colorbar.set_pad(0.2)
            fig.colorbar.set_axis_label_font(size=32)
            fig.colorbar.set_font(size=32)
        fig.set_nan_color(nancolor)


    h = [c.header for c in c_color + c_contour + c_vecang]
    if not(xspacing is None): fig.ticks.set_xspacing(xspacing * arcsec * 15)
    if not(yspacing is None): fig.ticks.set_yspacing(yspacing * arcsec)
    if xlabel != '': aplpy.AxisLabels(fig).set_xtext(xlabel)
    if ylabel != '': aplpy.AxisLabels(fig).set_ytext(ylabel)
    if center == '':
        xc, yc = (h[0]['CRVAL1'], h[0]['CRVAL2']) 
    else:
        xc, yc = coord2xy(center)
    xc, yc = xc + xoff * arcsec, yc + yoff * arcsec
    if rmax <= 0: rmax = h[0]['NAXIS2'] * h[0]['CDELT2'] / 2 / arcsec
    fig.recenter(xc, yc, radius=rmax * arcsec)
    for c, l, r, o, w in zip(c_contour, levels, rms, colors, linewidths):
        fig.show_contour(data=c, levels=l * r, colors=o, linewidths=w)
    for pdata, adata, vk in zip(c_vecamp, c_vecang, kwsvec):
        fig.show_vectors(pdata=pdata, adata=adata, **vk)
    if not (beamcolor is None or beamcolor == []):
        beamcolor = listing(beamcolor)
        if len(beamcolor) < len(h):
            beamcolor += [None] * (len(h) - len(beamcolor))
        bsort = np.argsort([hh['BMAJ'] * hh['BMIN'] for hh in h])[::-1]
        hsort = [h[bsort[i]] for i in range(len(bsort))]
        beams = [[hh['BMAJ'], hh['BMIN'], hh['BPA']] for hh in hsort]
        bc = [beamcolor[bsort[i]] for i in range(len(bsort))]
        for b, c in zip(beams, bc):
            if not (c is None):
                fig.add_beam(major=b[0], minor=b[1], angle=b[2],
                             color=c, corner=beampos)
    if not (scalebar is None or scalebar == {}):
        scalebar['length'] *= arcsec
        if scalebar['length'] > 0: fig.add_scalebar(**scalebar)
    for l in labels:
        a = {'x':0.1, 'y':0.1, 'text':'', 'relative':True, 'color':'gray',
             'ha':'center', 'va':'center'}
        if l['text'] != '': fig.add_label(**dict(a, **l))
    xw, yw = coord2xy(circle_c)
    if len(xw) > 0:
        for s, t, r, k in zip(xw, yw, circle_r, kwscircle):
            fig.show_circles(s, t, r * arcsec, **dict({'color':'gray'}, **k))
    xw, yw = [], []
    if markrel:
        if len(markpos) > 0:
            xw = xc + np.array(markpos)[:, 0] * arcsec / np.cos(yc * deg)
            yw = yc + np.array(markpos)[:, 1] * arcsec
    else:
        xw, yw = coord2xy(markpos)
    kwsmark = listing(kwsmark)
    if len(kwsmark) == 1: kwsmark = kwsmark * len(xw)
    for s, t, u in zip(xw, yw, kwsmark):
        fig.show_markers(s, t, **dict(kwsmark0, **u))
    xw, yw = [], []
    if len(line_a) > 0:
        line_a = np.array(line_a) * deg
        if len(line_r) == 0: line_r = [3. * rmax] * len(line_a)
        if len(line_c) == 0: xw, yw = np.full((len(line_a), 2), [xc, yc]).T
        else: xw, yw = coord2xy(line_c)
        line_r = np.array(line_r) * arcsec
    for lx, ly, lp, lr, lk in zip(xw, yw, line_a, line_r, kwsline):
        line_o = np.array([lx, ly])
        line_e = np.array([np.sin(lp) / np.cos(ly * deg), np.cos(lp)])
        line_list = [np.c_[line_o + lr * line_e, line_o]]
        fig.show_lines(line_list=line_list, **lk)
    if not (title is None or title == {}): fig.set_title(**title)
    if savefig != '': fig.savefig(savefig, adjust_bbox=True, transparent=True)
    if show: plt.show()
    plt.close()    
