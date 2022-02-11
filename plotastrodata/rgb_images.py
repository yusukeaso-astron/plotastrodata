from fits_utils import fits2data
from other_utils import listing



def rgb_array(fig=None, ax=None, x=None, y=None, xlabel='x', ylabel='y',
                bmaj=0, bmin=0, bpa=0,
                rarray=[], garray=[], barray=[], logcolor=False,
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

    set_rcparams(24, 'k', 'inout')

    colorarray = set_rgb(rarray, garray, barray, logcolor)
    x, y = set_grid(colorarray, contourarray, vec1array, x, y, None, chan=False)
    d = DataForPlot(1)
    d.gen_color(colorarray)
    d.gen_contour(contourarray)
    d.gen_vector(vec0array, vec1array, vectype, vecrot)
    d.gen_grid(x, y, [])

    d, iskip = sub_grid(d, center, rmax, None, None, nskipx, None, chan=False)
    colors, linewidths, rms, levels = \
        set_contour(d, colors, linewidths, rms, levels)
    kwsvec = set_vector(d, kwsvec, vecskip)
    d, xvec, yvec = skipping(d, iskip, vecskip)

    if fig is None:
        fig = plt.figure(figsize=(9, 9))
        exfig = False
    else:
        exfig = True
    if ax is None: ax = fig.add_subplot(1, 1, 1)
    set_axes(ax, d, xticks, yticks, xticksminor, yticksminor,
             xticklabels, yticklabels, xlabel, ylabel, True, 0)
    show_rgb(ax, d)
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


def rgb_fits(fig=None, ax=None,
               xlabel='R.A. (arcsec)', ylabel='Dec. (arcsec)',
               rfits='', gfits='', bfits='', logcolor=False,
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
    rarray,       x, brgb = fits2data(rfits,       False,     center)
    garray,       x, brgb = fits2data(gfits,       False,     center)
    barray,       x, brgb = fits2data(bfits,       False,     center)

    if len(x) > 0: x, y, v = x[0]
    bmaj, bmin, bpa = np.array(brgb + bcon + bvec).T
    markpos = set_markpos(markpos, markrel, center,
                          rfits, contourfits, vec1fits)

    rgb_array(vectype=vectype, center=[xoff, yoff],
              fig=fig, ax=ax, x=x, y=y, xlabel=xlabel, ylabel=ylabel,
              bmaj=bmaj, bmin=bmin, bpa=bpa, 
              rarray=rarray, garray=garray, barray=barray, nskipx=nskipx,
              contourarray=contourarray, colors=colors,
              linewidths=linewidths, rms=rms, levels=levels,
              vec0array=vec0array, vec1array=vec1array,
              kwsvec=kwsvec, vecskip=vecskip, vecrot=vecrot,
              rmax=rmax,
              beamcolor=beamcolor, savefig=savefig,
              show=show, grid=grid,
              logcolor=logcolor,
              xticks=xticks, yticks=yticks,
              xticklabels=xticklabels, yticklabels=yticklabels,
              xticksminor=xticksminor, yticksminor=yticksminor,
              line_c=line_c, line_a=line_a, line_r=line_r, kwsline=kwsline,
              labels=labels, scalebar=scalebar,
              markpos=markpos, kwsmark=kwsmark)
