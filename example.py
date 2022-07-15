from plotastrodata.plot_utils import PlotAstroData as pad
from plotastrodata.plot_utils import profile

pre = 'testFITS/'

# 2D case
f = pad(rmax=0.8, center='04h04m43.07s 26d18m56.20s')
f.add_color(fitsimage=pre+'test2D.fits', Tb=True, cblabel='Tb (K)')
f.add_contour(fitsimage=pre+'test2D_2.fits', colors='r', sigma=5e-3)
f.add_contour(fitsimage=pre+'test2D.fits', skip=2, sigma=5e-3)
f.add_segment(ampfits=pre+'test2Damp.fits',
              angfits=pre+'test2Dang.fits', skip=3)
f.add_scalebar(length=50 / 140, label='50 au')
f.add_text([0.3, 0.3], slist='text')
f.add_marker('04h04m43.07s 26d18m56.20s')
f.add_line([[0.5, 0.5], [0.6, 0.6]], anglelist=[60, 60], rlist=[0.5, 0.5])
f.add_arrow([0.4, 0.4], anglelist=150, rlist=0.5)
f.add_region('ellipse', [0.2, 0.8], majlist=0.4, minlist=0.2, palist=45)
f.set_axis_radec(nticksminor=5, title={'label':'2D image', 'loc':'right'})
f.savefig('test2D.png', show=True)

# 3D case
f = pad(rmax=0.8, fitsimage=pre+'test3D.fits', vmin=-5, vmax=5, vskip=2)
f.add_color(fitsimage=pre+'test3D.fits', stretch='log')
f.add_contour(fitsimage=pre+'test3D.fits', colors='r')
f.add_contour(fitsimage=pre+'test2D.fits', colors='b')
f.add_segment(ampfits=pre+'test2Damp.fits',
              angfits=pre+'test2Dang.fits', skip=3)
f.add_scalebar(length=50 / 140, label='50 au')
f.add_text([[0.3, 0.3]], slist=['text'], include_chan=[0,1,2])
f.add_marker([0.7, 0.7])
f.add_line([[0.5, 0.5], [0.6, 0.6]], anglelist=[60, 60], rlist=[0.5, 0.5],
           include_chan=[6,7,8])
f.add_arrow([[0.4, 0.4]], anglelist=[150], rlist=[0.5],
            include_chan=[9,10,11])
f.add_region('rectangle', [[0.2, 0.8]],
             majlist=[0.4], minlist=[0.2], palist=[45],
             include_chan=[12,13,14])
f.set_axis(grid={}, title='3D channel maps')
f.savefig('test3D.png', show=True)

# PV case
f = pad(rmax=0.8, pv=True, vmin=-5, vmax=5)
f.add_color(fitsimage=pre+'testPV.fits', Tb=True, cblabel='Tb (K)')
f.add_contour(fitsimage=pre+'testPV.fits', colors='r', sigma=5e-3)
f.add_text([0.3, 0.3], slist='text')
f.add_marker([[0.5, 0.5]])
f.set_axis(title='PV diagram')
f.savefig('testPV.png', show=True)

# log log PV case
f = pad(rmax=0.8 * 140, pv=True, quadrants='13', vmin=-5, vmax=5, dist=140)
f.add_color(fitsimage=pre+'testPV.fits', Tb=True, cblabel='Tb (K)')
f.add_contour(fitsimage=pre+'testPV.fits', colors='r', sigma=5e-3)
f.set_axis(title='loglog PV diagram', loglog=20)
f.savefig('testloglogPV.png', show=True)

# Line profile
profile(fitsimage=pre+'test3D.fits', radius=0.2, flux=True,
        coords=['04h04m43.045s 26d18m55.766s', '04h04m43.109s 26d18m56.704s'],
        gaussfit=True, savefig='testprofile.png', show=True, width=2)

# RGB case
f = pad(rmax=0.8, center='04h04m43.07s 26d18m56.20s')
f.add_rgb(fitsimage=[pre+'test'+c+'.fits' for c in ['R', 'G', 'B']])
f.add_scalebar(length=50 / 140, label='50 au')
f.set_axis_radec(nticksminor=5, title={'label':'2D image', 'loc':'right'})
f.savefig('test2Drgb.png', show=True)
