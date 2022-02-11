from plotastrodata.channels import chan_fits
from plotastrodata.momentmaps import mom_fits
from plotastrodata.pvdiagrams import pv_fits
from plotastrodata.lineprofiles import lineprofile


chan_fits(colorfits='test_input.fits',
          contourfits='test_input.fits',
          rmax=0.8, nrows=3, ncols=5, nskipv=4, vmin=-4.7, vmax=4.7, 
          scalebar={'length':50 / 140, 'label':'50 au'},
          savefig='test_chan.png', show=True)

mom_fits(vlim=(-5, 5), cubefits='test_input.fits',
         rmax=0.8, scalebar={'length':50 / 140, 'label':'50 au'},
         savefig='test_mom01.png', show=True)

pv_fits(length=1.6, width=0, pa=60,
        colorfits='test_input.fits',
        contourfits='test_input.fits',
        savefig='test_pv.png', show=True)

lineprofile(fitsimage='test_input.fits', coords=['04h04m43.1s 26d18m56.2s'],
            radius='point', unit='K',
            ylabel='K', savefig='test_prof.png', show=True)
