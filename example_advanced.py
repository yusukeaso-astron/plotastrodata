import matplotlib.pyplot as plt
import matplotlib.animation as animation
from plotastrodata.plot_utils import PlotAstroData as pad

pre = 'testFITS/'
nchans = 31

def update_plot(i):
    f = pad(rmax=0.8, fitsimage=pre+'test3D.fits', vmin=-5, vmax=5, vskip=2,
            channelnumber=i, fig=fig)
    f.add_color(fitsimage=pre+'test3D.fits', stretch='log')
    f.add_contour(fitsimage=pre+'test3D.fits', colors='r')
    f.add_contour(fitsimage=pre+'test2D.fits', colors='b')
    f.add_segment(ampfits=pre+'test2Damp.fits',
                  angfits=pre+'test2Dang.fits', xskip=3, yskip=3)
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
    f.set_axis_radec(grid={}, title='3D channel maps')
    f.fig.tight_layout()

fig = plt.figure()
ani = animation.FuncAnimation(fig, update_plot, frames=nchans, interval=50)
Writer = animation.writers['ffmpeg']  # for mp4
#Writer = animation.writers['pillow']  # for gif
writer = Writer(fps=10, bitrate=128)  # frame per second
ani.save('test_animation.mp4', writer=writer)
#ani.save('test_animation.gif', writer=writer)
