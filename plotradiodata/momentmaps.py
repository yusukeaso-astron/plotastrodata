from fits_utils import fits2data, read_bunit
from plotradiodata.images import image_array



def mom_array(fig=None, ax=None, x=None, y=None, v=None,
              xlabel='offset', ylabel='velocity',
              bmaj=0, bmin=0, bpa=0,
              vlim=None, masksigma=3, mom1sigma=2,
              cubearray=None, cmap='coolwarm', bunit=r'km s$^{-1}$',
              cbticks=[], cbticklabels=[], cbformat='%.1f',
              nancolor='white', cmin=None, cmax=None,
              contourarray=[], colors='gray', linewidths=1.0, rms=None,
              levels=[-24, -12, -6, -3, 3, 6, 12, 24, 48, 96, 198, 384],
              vec0array=[], vec1array=[], vecskip=1, vecrot=0,
              vectype='nochange', kwsvec={'color':'gray', 'width':0.007},
              rmax=0, center=[0, 0], nskipx=1,
              labels=[{'x':0.1, 'y':0.1, 's':'', 'color':'gray'}],
              scalebar={'length':0, 'label':'', 'color':'gray',
                        'fontsize':32, 'linewidth':3},
              beamcolor='gray', savefig='', show=True, grid=None, 
              xticks=[], yticks=[], xticksminor=[], yticksminor=[],
              xticklabels=[], yticklabels=[],
              line_c=[], line_a=[], line_r=[], kwsline={'color':'gray'},
              markpos=[], kwsmark=[{'marker':'+', 'color':'gray'}]):


    rmscube = np.nanstd(cubearray[[0, -1]])
    if vlim is None:
        ic, jc = np.argmin(np.abs(x)), np.argmin(np.abs(y))
        profile = np.nansum(cubearray[:, jc - 20:jc + 20, ic - 20:ic + 20],
                            axis=(1, 2))
        rmsprofile = rmscube * 40
        profile[profile < 3 * rmsprofile] = np.nan
        if np.all(np.isnan(profile)):
            print('No signal.')
            return
        vmean = np.nansum(profile * v) / np.nansum(profile)
        vstd = np.nansum(profile * (v - vmean)**2) / np.nansum(profile)
        vstd = np.sqrt(vstd)
        vmin, vmax = vmean - 2 * vstd, vmean + 2 * vstd
        print(f'vmin, vmax = {vmin:.2f}, {vmax:.2f}')
        vlim = [vmin, vmax]
    klim = [np.argmin(np.abs(v - vt)) for vt in vlim]
    if klim[0] > klim[1]: klim = klim[::-1]
    dv = np.abs(v[1] - v[0])
    ca, vv = np.empty_like(cubearray[0:1]), np.empty_like(v[0:1])
    for i in range(len(klim) // 2):
        ca = np.concatenate((ca, cubearray[klim[2 * i]:klim[2 * i + 1] + 1]))
        vv = np.concatenate((vv,         v[klim[2 * i]:klim[2 * i + 1] + 1]))
    cubearray, v = ca[1:], vv[1:]
    vcube = np.moveaxis(np.full((len(y), len(x), len(v)), v), -1, 0)
    mom0 = np.nansum(cubearray, axis=0) * dv
    mom0[mom0 == 0] = 1e-10
    cubearray[cubearray < mom1sigma * rmscube] = np.nan
    mom1 = np.nansum(cubearray * vcube, axis=0) / mom0 * dv
    rmsmom0 = rmscube * np.sqrt(len(v)) * dv
    mom1[mom0 < masksigma * rmsmom0] = np.nan
    if not(type(rms) in [float, int, str, list]): rms = rmsmom0
    if len(contourarray) == 0:
        contourarray = mom0
    elif len(contourarray) == len(y):
        contourarray = [mom0, contourarray]
    else:
        contourarray = np.concatenate(([mom0], contourarray))

    image_array(vectype=vectype, center=center[:2],
                fig=fig, ax=ax, sameaxes=True,
                x=x, y=y, xlabel=xlabel, ylabel=ylabel,
                bmaj=bmaj, bmin=bmin, bpa=bpa, 
                colorarray=mom1, cmap=cmap, nskipx=nskipx,
                contourarray=contourarray, colors=colors,
                linewidths=linewidths, rms=rms, levels=levels,
                vec0array=vec0array, vec1array=vec1array,
                kwsvec=kwsvec, vecskip=vecskip, vecrot=vecrot,
                rmax=rmax,
                beamcolor=beamcolor, savefig=savefig,
                show=show, grid=grid, nancolor=nancolor, bunit=bunit,
                logcolor=False, cbticks=cbticks,
                cbticklabels=cbticklabels, cbformat=cbformat,
                cmin=cmin, cmax=cmax, alpha=1,
                xticks=xticks, yticks=yticks,
                xticklabels=xticklabels, yticklabels=yticklabels,
                xticksminor=xticksminor, yticksminor=yticksminor,
                line_c=line_c, line_a=line_a, line_r=line_r, kwsline=kwsline,
                labels=labels, scalebar=scalebar,
                markpos=markpos, kwsmark=kwsmark)


def mom_fits(fig=None, ax=None,
             xlabel='R.A (arcsec)', ylabel='Dec. (arcsec)',
             vlim=None, masksigma=3, mom1sigma=2,
             cubefits='', cmap='coolwarm', bunit=r'km s$^{-1}$',
             cbticks=[], cbticklabels=[], cbformat='%.1f',
             nancolor='white', cmin=None, cmax=None,
             contourfits=[], colors='gray', linewidths=1.0, rms=None,
             levels=[-24, -12, -6, -3, 3, 6, 12, 24, 48, 96, 198, 384],
             Tbcontour=False,
             vec0fits=[], vec1fits=[], vecskip=1, vecrot=0,
             vectype='ampang', kwsvec={'color':'gray', 'width':0.007},
             rmax=0, center='', xoff=0, yoff=0, voff=0, nskipx=1,
             labels=[{'x':0.1, 'y':0.1, 's':'', 'color':'gray'}],
             scalebar={'length':0, 'label':'', 'color':'gray',
                       'fontsize':32, 'linewidth':3},
             beamcolor='gray', savefig='', show=True, grid=None,
             xticks=[], yticks=[], xticksminor=[], yticksminor=[],
             xticklabels=[], yticklabels=[],
             line_c=[], line_a=[], line_r=[], kwsline={'color':'gray'},
             markpos=[], markrel=True,
             kwsmark=[{'marker':'+', 'color':'gray'}]):

    vec0array,    x, bvec = fits2data(vec0fits,    False,     center)
    vec1array,    x, bvec = fits2data(vec1fits,    False,     center)
    contourarray, x, bcon = fits2data(contourfits, Tbcontour, center)
    cubearray,    x, bmom = fits2data(cubefits,    False,     center)

    if len(x) > 0: x, y, v = x[0]
    bmaj, bmin, bpa = np.array(bmom + bcon + bvec).T
    markpos = set_markpos(markpos, markrel, center, cubefits, contourfits, vec1fits)

    mom_array(vectype=vectype, center=[xoff, yoff],
              fig=fig, ax=ax, bmaj=bmaj, bmin=bmin, bpa=bpa,
              x=x, y=y, v=v - voff, xlabel=xlabel, ylabel=ylabel,
              cubearray=cubearray[0], cmap=cmap, bunit=bunit,
              contourarray=contourarray,
              colors=colors, linewidths=linewidths, rms=rms, levels=levels,
              vec0array=vec0array, vec1array=vec1array,
              kwsvec=kwsvec, vecskip=vecskip, vecrot=vecrot,
              nskipx=nskipx, vlim=vlim,
              masksigma=masksigma, mom1sigma=mom1sigma,
              beamcolor=beamcolor, savefig=savefig, show=show, grid=grid,
              nancolor=nancolor, cbticks=cbticks,
              cbticklabels=cbticklabels, cbformat=cbformat,
              cmin=cmin, cmax=cmax, rmax=rmax,
              xticks=xticks, yticks=yticks,
              xticksminor=xticksminor, yticksminor=yticksminor,
              xticklabels=xticklabels, yticklabels=yticklabels,
              labels=labels, scalebar=scalebar,
              markpos=markpos, kwsmark=kwsmark,
              line_c=line_c, line_a=line_a, line_r=line_r, kwsline=kwsline)
