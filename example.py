from plotastrodata.analysis_utils import AstroData, AstroFrame
from plotastrodata.plot_utils import PlotAstroData as pad
from plotastrodata.plot_utils import plotprofile, plotslice, plot3d

pre = 'testFITS/'

# 2D case
p = pad(rmax=0.8, center='04h04m43.07s 26d18m56.20s')
d = AstroData(fitsimage=pre+'test2D.fits', Tb=True, sigma=5e-3)
f = AstroFrame(rmax=0.8, center='04h04m43.07s 26d18m56.20s')
f.read(d)
p.add_color(**d.todict(), cblabel='Tb (K)')
p.add_contour(fitsimage=pre+'test2D_2.fits', colors='r', sigma=5e-3)
p.add_contour(fitsimage=pre+'test2D.fits', xskip=2, yskip=2, sigma=5e-3)
p.add_segment(ampfits=pre+'test2Damp.fits',
              angfits=pre+'test2Dang.fits', xskip=3, yskip=3)
p.add_scalebar(length=50 / 140, label='50 au')
p.add_text([0.3, 0.3], slist='text')
p.add_marker('04h04m43.07s 26d18m56.20s')
p.add_line([[0.5, 0.5], [0.6, 0.6]], anglelist=[60, 60], rlist=[0.5, 0.5])
p.add_arrow([0.4, 0.4], anglelist=150, rlist=0.5)
p.add_region('ellipse', [0.2, 0.8], majlist=0.4, minlist=0.2, palist=45)
p.set_axis_radec(nticksminor=5, title={'label': '2D image', 'loc': 'right'})
p.savefig('test2D.png', show=True)

# 3D case
p = pad(rmax=0.8, fitsimage=pre+'test3D.fits', vmin=-5, vmax=5, vskip=2)
p.add_color(fitsimage=pre+'test3D.fits', stretch='log')
p.add_contour(fitsimage=pre+'test3D.fits', colors='r')
p.add_contour(fitsimage=pre+'test2D.fits', colors='b', sigma=5e-3)
p.add_segment(ampfits=pre+'test2Damp.fits',
              angfits=pre+'test2Dang.fits', xskip=3, yskip=3)
p.add_scalebar(length=50 / 140, label='50 au')
p.add_text([[0.3, 0.3]], slist=['text'], include_chan=[0, 1, 2])
p.add_marker([0.7, 0.7])
p.add_line([[0.5, 0.5], [0.6, 0.6]], anglelist=[60, 60], rlist=[0.5, 0.5],
           include_chan=[6, 7, 8])
p.add_arrow([[0.4, 0.4]], anglelist=[150], rlist=[0.5],
            include_chan=[9, 10, 11])
p.add_region('rectangle', [[0.2, 0.8]],
             majlist=[0.4], minlist=[0.2], palist=[45],
             include_chan=[12, 13, 14])
p.set_axis(grid={}, title='3D channel maps')
p.savefig('test3D.png', show=True)

# PV case
p = pad(rmax=0.8, pv=True, swapxy=True, vmin=-5, vmax=5, figsize=(6, 7))
p.add_color(fitsimage=pre+'testPV.fits', Tb=True, cblabel='Tb (K)',
            cblocation='top')
p.add_contour(fitsimage=pre+'testPV.fits', colors='r', sigma=1e-3)
p.add_text([0.3, 0.3], slist='text')
p.add_marker([[0.5, 0.5]])
p.set_axis(title='PV diagram')
p.savefig('testPV.png', show=True)

# log log PV case
p = pad(rmax=0.8 * 140, pv=True, quadrants='13', vmin=-5, vmax=5, dist=140)
p.add_color(fitsimage=pre+'testPV.fits', Tb=True, cblabel='Tb (K)')
p.add_contour(fitsimage=pre+'testPV.fits', colors='r', sigma=1e-3)
p.set_axis(title='loglog PV diagram', loglog=20)
p.savefig('testloglogPV.png', show=True)

# RGB case
p = pad(rmax=0.8, center='04h04m43.07s 26d18m56.20s')
p.add_rgb(fitsimage=[pre+'test'+c+'.fits' for c in ['R', 'G', 'B']],
          sigma=[5e-3, 5e-3, 5e-3])
p.add_contour(fitsimage=pre+'test2D_2.fits', colors='r', sigma=5e-3)
p.add_contour(fitsimage=pre+'test2D.fits', xskip=2, yskip=2, sigma=5e-3)
p.add_segment(ampfits=pre+'test2Damp.fits',
              angfits=pre+'test2Dang.fits', xskip=3, yskip=3)
p.add_scalebar(length=50 / 140, label='50 au')
p.add_text([0.3, 0.3], slist='text')
p.add_marker('04h04m43.07s 26d18m56.20s')
p.add_line([[0.5, 0.5], [0.6, 0.6]], anglelist=[60, 60], rlist=[0.5, 0.5])
p.add_arrow([0.4, 0.4], anglelist=150, rlist=0.5)
p.add_region('ellipse', [0.2, 0.8], majlist=0.4, minlist=0.2, palist=45)
p.set_axis_radec(nticksminor=5, title={'label': '2D RGB', 'loc': 'right'})
p.savefig('test2Drgb.png', show=True)

# Line profile
plotprofile(fitsimage=pre+'test3D.fits', ellipse=[[0.2, 0.2, 0]] * 2, flux=True,
            coords=['04h04m43.045s 26d18m55.766s', '04h04m43.109s 26d18m56.704s'],
            gaussfit=True, savefig='testprofile.png', show=True, width=2)

# Spatial slice
plotslice(length=1.6, pa=270, fitsimage=pre+'test2D.fits',
          center='04h04m43.07s 26d18m56.20s', sigma=5e-3,
          savefig='testslice.png', show=True)

# Rotatable 3D cube in html
plot3d(rmax=0.8, vmin=-5, vmax=5, fitsimage=pre+'test3D.fits',
       outname='test3D', levels=[3, 6, 9], show=False)
