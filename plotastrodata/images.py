from fits_utils import fits2data, read_bunit
from plotastrodata.settings import *



def image_array(fig=None, ax=None, sameaxes=True,
                x=None, y=None, xlabel='x', ylabel='y',
                bmaj=0, bmin=0, bpa=0,
                colorarray=[], cmap='cubehelix', logcolor=False, cfactor=1,
                bunit='c', cbticks=[], cbticklabels=[], cbformat='%.2f',
                nancolor='white', cmin=None, cmax=None, alpha=1,
                contourarray=[], colors='gray', linewidths=1.0, rms='neg',
                levels=[-24, -12, -6, -3, 3, 6, 12, 24, 48, 96, 198, 384],
                vec0array=[], vec1array=[], vecskip=1, vecrot=0,
                vectype='nochange', kwsvec={'color':'gray', 'width':0.007},
                center=[0, 0], rmax=0, nskipx=1,
                labels=[{'x':0.1, 'y':0.1, 's':'', 'color':'gray'}],
                scalebar={'length':0, 'label':'', 'color':'gray',
                          'fontsize':32, 'linewidth':3},
                beamcolor='gray', savefig='', show=True, grid=None,
                xticks=[], yticks=[], xticksminor=[], yticksminor=[],
                xticklabels=[], yticklabels=[],
                line_c=[], line_a=[], line_r=[], kwsline={'color':'gray'},
                markpos=[],
                kwsmark=[{'marker':'+', 'ms':50, 'mfc':'gray',
                          'mec':'gray', 'mew':3, 'alpha':1}]):

    set_rcparams(24, nancolor, 'inout')
    x, y = set_grid(colorarray, contourarray, vec1array, x, y, None, chan=False)
    d = DataForPlot(1)
    d.gen_color(colorarray)
    d.gen_contour(contourarray)
    d.gen_vector(vec0array, vec1array, vectype, vecrot)
    d.gen_grid(x, y, [])
    if d.ncol > 0: d.color *= cfactor

    d, iskip = sub_grid(d, center, rmax, None, None, nskipx, None, chan=False)
    d, cmin, cmax, bunit = set_color(d, cmin, cmax, logcolor, bunit)
    colors, linewidths, rms, levels = \
        set_contour(d, colors, linewidths, rms, levels)
    kwsvec = set_vector(d, kwsvec, vecskip)
    d, xvec, yvec = skipping(d, iskip, vecskip)

    if fig is None:
        if d.ncol == 0 or np.all(bunit is None): figsize = (9, 9)
        else: figsize = (12, 9)
        fig = plt.figure(figsize=figsize)
        exfig = False
    else:
        exfig = True
    if ax is None: ax = fig.add_subplot(1, 1, 1)
    set_axes(ax, d, xticks, yticks, xticksminor, yticksminor,
             xticklabels, yticklabels, xlabel, ylabel, sameaxes, 0)
    cb, _ = show_color(ax, d, cmap, cmin, cmax, bunit, cbformat, alpha, None)
    set_colorbar(cb, cbticks, cbticklabels, logcolor)
    show_contour(ax, d, levels, rms, colors, linewidths, None)
    show_vectors(ax, d, kwsvec, xvec, yvec, None)
    show_lines(ax, line_c, line_a, line_r, kwsline, rmax)
    show_markers(ax, center, markpos, kwsmark)
    add_beam(ax, d, bmaj, bmin, bpa, beamcolor)
    add_label(ax, d, labels)
    add_scalebar(ax, d, scalebar)
    if not (grid is None): ax.grid(**grid)
    if not exfig:
        fig.tight_layout()
        if savefig != '':
            fig.patch.set_alpha(0)
            fig.savefig(savefig, bbox_inches='tight', transparent=False)
        if show == True: plt.show()
        plt.close()


def image_fits(fig=None, ax=None, sameaxes=True,
               xlabel='R.A. (arcsec)', ylabel='Dec. (arcsec)',
               colorfits=[], cmap='cubehelix', Tbcolor=False, logcolor=False,
               bunit='c', cbticks=[], cbticklabels=[], cbformat='%.2f',
               nancolor='white', cmin=None, cmax=None, alpha=1, cfactor=1,
               contourfits=[], colors='gray', linewidths=1.8, rms='neg',
               levels=[-24, -12, -6, -3, 3, 6, 12, 24, 48, 96, 198, 384],
               Tbcontour=False,
               vec0fits=[], vec1fits=[], vecskip=1, vecrot=0,
               vectype='ampang', kwsvec={'color':'gray', 'width':0.007},
               center='', rmax=0, xoff=0, yoff=0, nskipx=1,
               labels=[{'x':0.1, 'y':0.1, 's':'', 'color':'gray'}],
               scalebar={'length':0, 'label':'', 'color':'gray',
                         'fontsize':32, 'linewidth':3},
               beamcolor='gray', savefig='', show=True, grid=None,
               xticks=[], yticks=[], xticklabels=[], yticklabels=[],
               xticksminor=[], yticksminor=[],
               line_c=[], line_a=[], line_r=[], kwsline={'color':'gray'},
               markpos=[], markrel=True,
               kwsmark=[{'marker':'+', 'ms':50, 'mfc':'gray',
                         'mec':'gray', 'mew':3, 'alpha':1}]):

    vec0array,    x, bvec = fits2data(vec0fits,    False,     center)
    vec1array,    x, bvec = fits2data(vec1fits,    False,     center)
    contourarray, x, bcon = fits2data(contourfits, Tbcontour, center)
    colorarray,   x, bcol = fits2data(colorfits,   Tbcolor,   center)
    if len(x) > 0: x, y, v = x[0]
    bmaj, bmin, bpa = np.array(bcol + bcon + bvec).T
    bunit = read_bunit(colorfits, bunit)
    markpos = set_markpos(markpos, markrel, center,
                          colorfits, contourfits, vec1fits)

    image_array(vectype=vectype, center=[xoff, yoff],
                fig=fig, ax=ax, sameaxes=sameaxes,
                x=x, y=y, xlabel=xlabel, ylabel=ylabel,
                bmaj=bmaj, bmin=bmin, bpa=bpa, 
                colorarray=colorarray, cfactor=cfactor, cmap=cmap,
                contourarray=contourarray, colors=colors,
                linewidths=linewidths, rms=rms, levels=levels,
                vec0array=vec0array, vec1array=vec1array,
                kwsvec=kwsvec, vecskip=vecskip, vecrot=vecrot,
                rmax=rmax, nskipx=nskipx,
                beamcolor=beamcolor, savefig=savefig,
                show=show, grid=grid, nancolor=nancolor, bunit=bunit,
                logcolor=logcolor, cbticks=cbticks,
                cbticklabels=cbticklabels, cbformat=cbformat,
                cmin=cmin, cmax=cmax, alpha=alpha,
                xticks=xticks, yticks=yticks,
                xticklabels=xticklabels, yticklabels=yticklabels,
                xticksminor=xticksminor, yticksminor=yticksminor,
                line_c=line_c, line_a=line_a, line_r=line_r, kwsline=kwsline,
                labels=labels, scalebar=scalebar,
                markpos=markpos, kwsmark=kwsmark)

