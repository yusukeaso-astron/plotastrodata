# plotastrodata
Python package to make figures from radio astronomical data.

Basic rules for the main class plotastrodata ---
    For 3D data, a 1D velocity array or a FITS file with a velocity axis must be given to set up channels in each page. len(v)=1 (default) means to make a 2D figure. Lengths are in the unit of arcsec. Angles are in the unit of degree. For ellipse, line, arrow, label, and marker, a single input must be listed like poslist=[[0.2, 0.3]], and each element of poslist supposes a text coordinate like '01h23m45.6s 01d23m45.6s' or a list of relative x and y like [0.2, 0.3] (0 is left or bottom, 1 is right or top). Parameters for original methods in matplotlib.axes.Axes can be used as kwargs; see the default kwargs0 for reference.
