import matplotlib.pyplot as plt
from plotastrodata.plot_utils import plotastro2D, plotastro3D


pa = plotastro2D(rmax=0.8, center='04h04m43.07s 26d18m56.20s')
pa.add_color(fitsimage='test2D.fits', Tb=True, cblabel='Tb (K)', cmin=0)
pa.add_contour(fitsimage='test2D_2.fits', colors='r', sigma=5e-3)
pa.add_contour(fitsimage='test2D.fits', skip=2, sigma=5e-3)
pa.add_vector(ampfits='test2Damp.fits', angfits='test2Dang.fits', skip=3)
pa.add_scalebar(length=50 / 140, label='50 au', color='k')
pa.add_label([[0.3, 0.3]], slist=['label'])
pa.add_marker([[0.7, 0.7]])
pa.add_line([[0.5, 0.5], [0.6, 0.6]], anglelist=[60, 60], rlist=[0.5, 0.5])
pa.add_arrow([[0.4, 0.4]], anglelist=[150], rlist=[0.5])
pa.add_ellipse([[0.2, 0.8]], majlist=[0.4], minlist=[0.2], palist=[45])
pa.set_axis()
pa.savefig()
pa.show()

pa = plotastro3D(rmax=0.8, center='04h04m43.07s 26d18m56.20s',
                 fitsimage='test3D.fits', vmin=-5, vmax=5, vskip=2)
pa.add_color(fitsimage='test3D.fits', log=True)
pa.add_contour(fitsimage='test3D.fits', colors='r')
pa.add_contour(fitsimage='test2D_2.fits', colors='b')
pa.add_vector(ampfits='test2Damp.fits', angfits='test2Dang.fits', skip=3)
pa.add_scalebar(length=50 / 140, label='50 au', color='k')
pa.add_label([0,1,2], [[0.3, 0.3]], slist=['label'])
pa.add_marker([3,4,5], ['04h04m43.07s 26d18m56.20s'])
pa.add_line([6,7,8], [[0.5, 0.5], [0.6, 0.6]], anglelist=[60, 60], rlist=[0.5, 0.5])
pa.add_arrow([9,10,11], [[0.4, 0.4]], anglelist=[150], rlist=[0.5])
pa.add_ellipse([12,13,14], [[0.2, 0.8]], majlist=[0.4], minlist=[0.2], palist=[45])
pa.set_axis(xticks=[-0.8,-0.4,0,0.4,0.8], yticks=[-0.8,-0.4,0,0.4,0.8])
pa.savefig()
pa.show()