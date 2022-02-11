from scipy.interpolate import RectBivariateSpline as RBS

from fits_utils import fits2data, read_bunit
from plotradiodata.images import image_array



def pv_array(fig=None, ax=None, x=None, y=None, v=None,
             xlabel='offset', ylabel='velocity',
             length=1, width=0, pa=0, flipaxes=False,
             colorarray=None, cmap='cubehelix', logcolor=False,
             bunit='c', cbticks=[], cbticklabels=[], cbformat='%f',
             nancolor='white', cmin=None, cmax=None, alpha=1,
             contourarray=[], colors='gray', linewidths=1.0, rms='edge',
             levels=[-24, -12, -6, -3, 3, 6, 12, 24, 48, 96, 198, 384],
             vec0array=[], vec1array=[], vecskip=1, vecrot=0,
             vectype='nochange', kwsvec={'color':'gray', 'width':0.007},
             center=[0, 0, 0], nskipx=1, nskipv=1, vmin=-1e4, vmax=1e4,
             labels=[{'x':0.1, 'y':0.1, 's':'', 'color':'gray'}],
             savefig='', show=True, grid=None, 
             xticks=[], yticks=[], xticksminor=[], yticksminor=[],
             xticklabels=[], yticklabels=[]):

    length, width, pa = length / 2., width / 2., pa * np.radians(1.)
    rmax = np.hypot(length, width)

    set_rcparams(24, nancolor, 'inout')
    x, y, v = set_grid(colorarray, contourarray, vec1array, x, y, v, chan=True)
    d = DataForPlot(len(v))
    d.gen_color(colorarray)
    d.gen_contour(contourarray)
    d.gen_vector(vec0array, vec1array, vectype, vecrot)
    d.gen_grid(x, y, v)
    d, iskip = sub_grid(d, center, rmax, vmin, vmax, nskipx, nskipv, chan=True)
    d, cmin, cmax, bunit = set_color(d, cmin, cmax, logcolor, bunit)
    colors, linewidths, rms, levels = \
        set_contour(d, colors, linewidths, rms, levels)
    kwsvec = set_vector(d, kwsvec, vecskip)
    d, xvec, yvec = skipping(d, iskip, vecskip)

    dx, sx = np.abs(d.x[1] - d.x[0]), np.sign(d.x[1] - d.x[0])
    dy, sy = np.abs(d.y[1] - d.y[0]), np.sign(d.y[1] - d.y[0])
    x_ip = np.linspace(-rmax, rmax, int(2 * rmax / dx * 3))
    y_ip = np.linspace(-rmax, rmax, int(2 * rmax / dy * 3))
    ns = max(3, int(round(2 * length / dx)))
    nt = max(3, int(round(2 * width / dx)))
    r = np.linspace(-length, length, ns)
    t = np.linspace(-width, width, nt) if width > 0 else np.array([0])
    s, t = np.meshgrid(r, t)
    xpv = (s * np.sin(pa) + t * np.cos(pa)) * sx
    ypv = (s * np.cos(pa) - t * np.sin(pa)) * sy
    ipv = np.round((xpv - x_ip[0]) / (x_ip[1] - x_ip[0])).astype(np.int64)
    jpv = np.round((ypv - y_ip[0]) / (y_ip[1] - y_ip[0])).astype(np.int64)

    def cube2pv(cube):
        pv = []
        for ch in cube:
            ch[np.isnan(ch)] = 0
            f = RBS(d.y * sy, d.x * sx, ch)
            pv.append(np.mean(f(y_ip, x_ip)[jpv, ipv], axis=0))
        return np.array(pv)

    colorarray = [cube2pv(a) for a in d.color] if d.ncol > 0 else []
    contourarray = [cube2pv(a) for a in d.contour] if d.ncon > 0 else []
    if logcolor: cbticks = np.log10(cbticks)

    if flipaxes:
        x, y = d.v, r
        colorarray = [c.T for c in colorarray]
        contourarray = [c.T for c in contourarray]
    else:
        x, y = r, d.v

    image_array(fig=fig, ax=ax, sameaxes=False, x=x, y=y,
                xlabel=xlabel, ylabel=ylabel,
                bmaj=0, bmin=0, bpa=0,
                colorarray=colorarray, cmap=cmap, logcolor=False,
                bunit=bunit, cbticks=cbticks, cbticklabels=cbticklabels,
                cbformat=cbformat, nancolor=nancolor,
                cmin=cmin, cmax=cmax, alpha=alpha,
                contourarray=contourarray, colors=colors,
                linewidths=linewidths, rms=rms, levels=levels,
                vec0array=vec0array, vec1array=vec1array,
                vecskip=vecskip, vecrot=vecrot,
                vectype=vectype, kwsvec=kwsvec,
                center=[0, 0], rmax=0, nskipx=1, labels=labels,
                scalebar={'length':0, 'label':''},
                beamcolor=None, savefig=savefig, show=show, grid=grid,
                xticks=xticks, yticks=yticks,
                xticksminor=xticksminor, yticksminor=yticksminor,
                xticklabels=xticklabels, yticklabels=yticklabels,
                line_c=[], line_a=[], line_r=[], kwsline={'color':'gray'})


def pv_fits(fig=None, ax=None,
            xlabel='Offset (arcsec)', ylabel=r'Velocity (km s$^{-1}$)',
            length=1, width=0, pa=0, flipaxes=False,
            colorfits='', cmap='cubehelix', Tbcolor=False, logcolor=False,
            bunit='c', cbticks=[], cbticklabels=[], cbformat='%f',
            nancolor='white', cmin=None, cmax=None, alpha=1,
            contourfits=[], colors='gray', linewidths=1.0, rms='edge',
            levels=[-24, -12, -6, -3, 3, 6, 12, 24, 48, 96, 198, 384],
            Tbcontour=False,
            vec0fits=[], vec1fits=[], vecskip=1, vecrot=0,
            vectype='ampang', kwsvec={'color':'gray', 'width':0.007},
            center='', xoff=0, yoff=0,
            nskipx=1, nskipv=1, vmin=-1e4, vmax=1e4,
            labels=[{'x':0.1, 'y':0.1, 'v':'all', 's':'', 'color':'gray'}],
            savefig='', show=True, grid=None,
            xticks=[], yticks=[], xticksminor=[], yticksminor=[],
            xticklabels=[], yticklabels=[]):

    vec0array,    xvec, bvec = fits2data(vec0fits,    False,     center)
    vec1array,    xvec, bvec = fits2data(vec1fits,    False,     center)
    contourarray, xcon, bcon = fits2data(contourfits, Tbcontour, center)
    colorarray,   xcol, bcol = fits2data(colorfits,   Tbcolor,   center)
    x, y, v = [a for a in [xcol, xcon, xvec] if len(a) > 0][0][0]
    bmaj, bmin, bpa = np.array(bcol + bcon + bvec).T
    bunit = read_bunit(colorfits, bunit)

    pv_array(vectype=vectype, center=[xoff, yoff, 0], flipaxes=flipaxes,
             fig=fig, ax=ax, length=length, width=width, pa=pa,
             x=x, y=y, v=v, xlabel=xlabel, ylabel=ylabel,
             colorarray=colorarray, cmap=cmap, bunit=bunit, logcolor=logcolor,
             contourarray=contourarray, colors=colors,
             linewidths=linewidths, rms=rms, levels=levels,
             vec0array=vec0array, vec1array=vec1array,
             kwsvec=kwsvec, vecskip=vecskip, vecrot=vecrot,
             nskipx=nskipx, nskipv=nskipv, vmin=vmin, vmax=vmax,
             labels=labels, savefig=savefig, show=show, grid=grid,
             nancolor=nancolor, cbticks=cbticks,
             cbticklabels=cbticklabels, cbformat=cbformat,
             cmin=cmin, cmax=cmax, alpha=alpha,
             xticks=xticks, yticks=yticks,
             xticksminor=xticksminor, yticksminor=yticksminor,
             xticklabels=xticklabels, yticklabels=yticklabels)
