import subprocess
import shlex
import numpy as np
import matplotlib.pyplot as plt
from fits_utils import fits2data, read_bunit
from plotastrodata.settings import *



def chan_array(x=None, y=None, v=None, xlabel='x', ylabel='y',
               bmaj=0, bmin=0, bpa=0,
               colorarray=[], cmap='cubehelix', logcolor=False, cfactor=1,
               bunit='c', cbticks=[], cbticklabels=[], cbformat='%f',
               nancolor='white', cmin=None, cmax=None, alpha=1,
               contourarray=[], colors='gray', linewidths=1.0, rms='edge',
               levels=[-24, -12, -6, -3, 3, 6, 12, 24, 48, 96, 198, 384],
               vec0array=[], vec1array=[], vecskip=1, vecrot=0,
               vectype='nochange', kwsvec={'color':'gray', 'width':0.007},
               nrows=4, ncols=6, center=[0, 0, 0], rmax=0,
               nskipx=1, nskipv=1, vmin=-1e4, vmax=1e4,
               labels=[{'x':0.1, 'y':0.1, 'v':'all', 's':'', 'color':'gray'}],
               scalebar={'length':0, 'label':'', 'color':'gray',
                         'corner':'bottom right', 'fontsize':12, 'linewidth':2},
               beamcolor='gray', savefig='', show=True, grid=None, veldigit=2,
               xticks=[], yticks=[], xticksminor=[], yticksminor=[],
               xticklabels=[], yticklabels=[],
               line_c=[], line_a=[], line_r=[], kwsline={'color':'gray'}):

    set_rcparams(15, nancolor, 'inout')

    x, y, v = set_grid(colorarray, contourarray, vec1array, x, y, v, chan=True)
    d = DataForPlot(len(v))
    d.gen_color(colorarray)
    d.gen_contour(contourarray)
    d.gen_vector(vec0array, vec1array, vectype, vecrot)
    d.gen_grid(x, y, v)
    if d.ncol > 0: d.color *= cfactor

    d, iskip = sub_grid(d, center, rmax, vmin, vmax, nskipx, nskipv, chan=True)
    d, cmin, cmax, bunit = set_color(d, cmin, cmax, logcolor, bunit)
    colors, linewidths, rms, levels = \
        set_contour(d, colors, linewidths, rms, levels)
    kwsvec = set_vector(d, kwsvec, vecskip)
    d, xvec, yvec = skipping(d, iskip, vecskip)

    npages = int(np.ceil(len(d.v) / (nrows * ncols)))
    for l in range(npages):
      fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                             sharex=True, sharey=True,
                             figsize=(ncols * 2, max(nrows, 1.5) * 2))
      if nrows == 1: ax = np.array([ax])
      fig.subplots_adjust(hspace=0, wspace=0, right=0.87, top=0.87)
      for i in range(nrows):
        for j in range(ncols):
          k = l * nrows * ncols + i * ncols + j
          ax[i, j].tick_params(top=True, right=True)
          set_axes(ax[i, j], d, xticks, yticks, xticksminor, yticksminor,
                   xticklabels, yticklabels, xlabel, ylabel, True, 30)
          if i < nrows - 1 and j < ncols - 1:  # I don't know why.
              if len(xticks) == 0 and len(yticks) == 0:
                  ax[i, j].set_yticks(ax[i, j].get_xticks())
                  ax[i, j].set_xlim(x[0], x[-1])
                  ax[i, j].set_ylim(y[0], y[-1])
          if i == nrows - 1 and j == 0:
              ax[i, j].tick_params(labelleft=True, labelbottom=True)
              add_beam(ax[i, j], d, bmaj, bmin, bpa, beamcolor)
              add_scalebar(ax[i, j], d, scalebar)
          else:
              ax[i, j].tick_params(labelleft=False, labelbottom=False)
              ax[i, j].set_xlabel('')
              ax[i, j].set_ylabel('')
          if k >= len(d.v): continue
          _, pcol = show_color(ax[i, j], d, cmap, cmin, cmax, bunit,
                               cbformat, alpha, k)
          show_contour(ax[i, j], d, levels, rms, colors, linewidths, k)
          show_vectors(ax[i, j], d, kwsvec, xvec, yvec, k)
          show_lines(ax[i, j], line_c, line_a, line_r, kwsline, rmax)
          ax[i, j].text(0.955 * d.x[0] + 0.045 * d.x[-1],
                        0.146 * d.y[0] + 0.854 * d.y[-1],
                        (r'${0:.' + str(int(veldigit)) + 'f}$').format(d.v[k]),
                        color='black', backgroundcolor='white', zorder=5)
          add_label(ax[i, j], d, labels, k)
          if not (grid is None): ax[i, j].grid(**grid)
      if len(colorarray) > 0:
          cax = plt.axes([0.88, 0.105, 0.015, 0.77])
          cb = [plt.colorbar(p, cax=cax, label=l, format=cbformat) \
                for p, l in zip(pcol, bunit)]
          set_colorbar(cb, cbticks, cbticklabels, logcolor, chan=True)
      if savefig != '':
          if npages > 1: s = savefig.rstrip('.pdf') + '_' + str(l) + '.pdf'
          else: s = savefig
          fig.patch.set_alpha(0)
          fig.savefig(s, bbox_inches='tight', transparent=False)
      if show == True: plt.show()
    if savefig != '' and npages > 1:
        cmd = 'pdfunite '
        for l in range(npages):
            cmd += savefig.rstrip('.pdf') + '_' + str(l) + '.pdf '
        cmd += savefig
        subprocess.call(shlex.split(cmd))
        for l in range(npages):
            subprocess.call(shlex.split('rm ' + savefig.rstrip('.pdf') 
                                        + '_' + str(l) + '.pdf'))
    plt.close()


def chan_fits(xlabel='R.A. (arcsec)', ylabel='Dec. (arcsec)',
              colorfits='', cmap='cubehelix', Tbcolor=False, logcolor=False,
              bunit=None, cbticks=[], cbticklabels=[], cbformat='%.1e',
              nancolor='white', cmin=None, cmax=None, alpha=1, cfactor=1,
              contourfits=[], colors='gray', linewidths=1.0, rms='edge',
              levels=[-24, -12, -6, -3, 3, 6, 12, 24, 48, 96, 198, 384],
              Tbcontour=False,
              vec0fits=[], vec1fits=[], vecskip=1, vecrot=0,
              vectype='ampang', kwsvec={'color':'gray', 'width':0.007},
              nrows=4, ncols=6, center='', rmax=0, xoff=0, yoff=0, voff=0,
              nskipx=1, nskipv=1, vmin=-1e4, vmax=1e4, 
              labels=[{'x':0.1, 'y':0.1, 'v':'all', 's':'', 'color':'gray'}],
              scalebar={'length':0, 'label':'', 'color':'gray',
                        'corner':'bottom right', 'fontsize':32, 'linewidth':3},
              beamcolor='gray', savefig='', show=True, grid=None, veldigit=2,
              xticks=[], yticks=[], xticksminor=[], yticksminor=[],
              xticklabels=[], yticklabels=[],
              line_c=[], line_a=[], line_r=[], kwsline={}):

    vec0array,    xvec, bvec = fits2data(vec0fits,    False,     center)
    vec1array,    xvec, bvec = fits2data(vec1fits,    False,     center)
    contourarray, xcon, bcon = fits2data(contourfits, Tbcontour, center)
    colorarray,   xcol, bcol = fits2data(colorfits,   Tbcolor,   center)
    x, y, v = [a for a in [xcol, xcon, xvec] if len(a) > 0][0][0]
    bmaj, bmin, bpa = np.array(bcol + bcon + bvec).T
    bunit = read_bunit(colorfits, bunit)

    chan_array(x=x, y=y, v=v, xlabel=xlabel, ylabel=ylabel,
               bmaj=bmaj, bmin=bmin, bpa=bpa, vectype=vectype,
               center=[xoff, yoff, voff],
               colorarray=colorarray, cmap=cmap, nskipx=nskipx, nskipv=nskipv,
               contourarray=contourarray, colors=colors, cfactor=cfactor,
               linewidths=linewidths, rms=rms, levels=levels,
               vec0array=vec0array, vec1array=vec1array,
               kwsvec=kwsvec, vecskip=vecskip, vecrot=vecrot,
               nrows=nrows, ncols=ncols, rmax=rmax,
               labels=labels, scalebar=scalebar,
               vmin=vmin, vmax=vmax, beamcolor=beamcolor, savefig=savefig,
               show=show, grid=grid, nancolor=nancolor, bunit=bunit,
               logcolor=logcolor, cbticks=cbticks,
               cbticklabels=cbticklabels, cbformat=cbformat,
               cmin=cmin, cmax=cmax, alpha=alpha, veldigit=veldigit,
               xticks=xticks, yticks=yticks,
               xticksminor=xticksminor, yticksminor=yticksminor,
               xticklabels=xticklabels, yticklabels=yticklabels,
               line_c=line_c, line_a=line_a, line_r=line_r, kwsline=kwsline)
