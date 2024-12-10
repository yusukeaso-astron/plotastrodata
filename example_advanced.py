import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from plotastrodata.plot_utils import PlotAstroData as pad
from plotastrodata.los_utils import sys2obs, polarvel2losvel

# The following introduces a way for making an animation file of a given FITS cube.
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
    f.add_text([[0.3, 0.3]], slist=['text'], include_chan=[0, 1, 2])
    f.add_marker([0.7, 0.7])
    f.add_line([[0.5, 0.5], [0.6, 0.6]], anglelist=[60, 60], rlist=[0.5, 0.5],
               include_chan=[6, 7, 8])
    f.add_arrow([[0.4, 0.4]], anglelist=[150], rlist=[0.5],
                include_chan=[9, 10, 11])
    f.add_region('rectangle', [[0.2, 0.8]],
                 majlist=[0.4], minlist=[0.2], palist=[45],
                 include_chan=[12, 13, 14])
    f.set_axis_radec(grid={}, title='3D channel maps')
    f.fig.tight_layout()


fig = plt.figure()
ani = animation.FuncAnimation(fig, update_plot, frames=nchans, interval=50)
Writer = animation.writers['ffmpeg']  # for mp4
# Writer = animation.writers['pillow']  # for gif
writer = Writer(fps=10, bitrate=128)  # frame per second
ani.save('test_animation.mp4', writer=writer)
# ani.save('test_animation.gif', writer=writer)
plt.close()

# The following introduces a way for plotting the projected morphology and the line-of-sight velocity of a streamer.
incl = 60
phi0 = 0
theta0 = 90

xscale = np.sin(np.radians(theta0))**2
vscale = 1 / np.sin(np.radians(theta0))

xsys = np.linspace(0, 3 / xscale, 200)
ysys = np.sqrt(2 * xsys + 1)
zsys = np.zeros_like(xsys)

r = np.hypot(xsys, ysys)
theta = np.zeros_like(r) + np.pi / 2
phi = np.arctan2(ysys, xsys)

v_r = -np.sqrt(1 / r * (2 - 1 / r))
v_theta = np.zeros_like(v_r)
v_phi = 1 / r

xsys = xsys * xscale
ysys = ysys * xscale
zsys = zsys * xscale

v_r = v_r * vscale
v_theta = v_theta * vscale
v_phi = v_phi * vscale

xobs, yobs, zobs = sys2obs(xsys=xsys, ysys=ysys, zsys=zsys,
                           incl=incl, phi0=phi0, theta0=theta0)
vlos = polarvel2losvel(v_r=v_r, v_theta=v_theta, v_phi=v_phi,
                       theta=theta, phi=phi,
                       incl=incl, phi0=phi0, theta0=theta0)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
m = ax.scatter(xobs, yobs, c=vlos, cmap='coolwarm', vmin=-1.5, vmax=1.5)
fig.colorbar(m, ax=ax, label=r'$V_\mathrm{los} / \sqrt{GM_{*}/r_{c}}$')
ax.set_xlim(3, -3)
ax.set_ylim(-3, 3)
ax.set_aspect(1)
ax.set_xlabel(r'$x / r_{c}$')
ax.set_ylabel(r'$y / r_{c}$')
ax.grid()
fig.savefig('streamer.png')
plt.close()
