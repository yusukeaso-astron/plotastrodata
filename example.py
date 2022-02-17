from plotastrodata.plot_utils import plotastrodata as pad

pre = 'testFITS/'
# 2D case
pa = pad(rmax=0.8, center='04h04m43.07s 26d18m56.20s')
pa.add_color(fitsimage=pre+'test2D.fits', Tb=True, cblabel='Tb (K)')
pa.add_contour(fitsimage=pre+'test2D_2.fits', colors='r', sigma=5e-3)
pa.add_contour(fitsimage=pre+'test2D.fits', skip=2, sigma=5e-3)
pa.add_segment(ampfits=pre+'test2Damp.fits',
               angfits=pre+'test2Dang.fits', skip=3)
pa.add_scalebar(length=50 / 140, label='50 au')
pa.add_label([0.3, 0.3], slist='label')
pa.add_marker('04h04m43.07s 26d18m56.20s')
pa.add_line([[0.5, 0.5], [0.6, 0.6]], anglelist=[60, 60], rlist=[0.5, 0.5])
pa.add_arrow([0.4, 0.4], anglelist=150, rlist=0.5)
pa.add_ellipse([0.2, 0.8], majlist=0.4, minlist=0.2, palist=45)
pa.set_axis()
pa.savefig()
pa.show()

# 3D case
pa = pad(rmax=0.8, fitsimage=pre+'test3D.fits', vmin=-5, vmax=5, vskip=2)
pa.add_color(fitsimage=pre+'test3D.fits', log=True)
pa.add_contour(fitsimage=pre+'test3D.fits', colors='r')
pa.add_contour(fitsimage=pre+'test2D.fits', colors='b')
pa.add_segment(ampfits=pre+'test2Damp.fits',
               angfits=pre+'test2Dang.fits', skip=3)
pa.add_scalebar(length=50 / 140, label='50 au')
pa.add_label([[0.3, 0.3]], slist=['label'], include_chan=[0,1,2])
pa.add_marker([0.7, 0.7])
pa.add_line([[0.5, 0.5], [0.6, 0.6]], anglelist=[60, 60], rlist=[0.5, 0.5], include_chan=[6,7,8])
pa.add_arrow([[0.4, 0.4]], anglelist=[150], rlist=[0.5], include_chan=[9,10,11])
pa.add_ellipse([[0.2, 0.8]], majlist=[0.4], minlist=[0.2], palist=[45], include_chan=[12,13,14])
pa.set_axis(grid={})
pa.savefig()
pa.show()
