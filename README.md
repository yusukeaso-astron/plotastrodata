# plotastrodata
Python package to make figures from radio astronomical data.

The file example.py shows will help to find how to use this package.

Basic rules ---

For 3D data, a 1D velocity array or a FITS file with a velocity axis must be given to set up channels in each page.
len(v)=1 (default) means to make a 2D figure.
Spatial lengths are in the unit of arcsec, or au if dist (!= 1) is given.
Angles are in the unit of degree.
For ellipse, line, arrow, label, and marker, a single input can be treated without a list, e.g., anglelist=60, as well as anglelist=[60].
Each element of poslist supposes a text coordinate like '01h23m45.6s 01d23m45.6s' or a list of relative x and y like [0.2, 0.3] (0 is left or bottom, 1 is right or top).
Parameters for original methods in matplotlib.axes.Axes can be used as kwargs; see the default kwargs0 for reference.
Position-velocity diagrams (pv=True) does not yet suppot ellipse, line, arrow, and segment because the units of abscissa and ordinate are different.
The parameter sigma can be one of the methods of ['edge', 'neg', 'med', 'iter', 'out'] as well as a specific value.
