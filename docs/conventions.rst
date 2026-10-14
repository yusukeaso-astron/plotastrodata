Conventions and usage notes
===========================

Runtime outputs
---------------

Computed scalar results use Python ``float``, ``int``, and ``bool``.
Extended-precision NumPy floating scalars, such as ``np.longdouble``, are
preserved. Arrays, including zero-dimensional arrays, remain arrays; the
normalization policy does not change their dtypes. Numerical operations such
as brightness-temperature conversion may promote their input dtype.
Generic container and pass-through helpers do not recursively convert inputs.
Integer constants remain integers. ``Mrot3d`` returns a NumPy array.
Internal calculations need no scalar normalization. Plotly HTML export uses
a separate payload with extended-precision values converted to float64;
``plot3d(return_data_layout=True)`` preserves their in-memory precision.

Data and coordinates
--------------------

The usual array orders are ``(y, x)`` for images, ``(v, y, x)`` for cubes,
and ``(v, x)`` for position-velocity (PV) diagrams. Coordinate arrays must
match the corresponding dimensions. Selecting ``channelnumber`` retains a
singleton channel axis, ``(1, ny, nx)``, in the plotting data.

Spatial offsets are in arcseconds for ``dist=1``, or au when ``dist`` is the
source distance in parsecs. Velocities are in km/s and rest frequencies in Hz.
FITS writing converts spatial units back to degrees even without a sky center.

``AstroFrame(swapxy=True).read(d)`` transposes the last two data axes,
exchanges the processed x/y coordinates, and transforms the stored beam angle
to ``90 - bpa``. It reports the changed axis order once per read.
``AstroData`` does not track or undo swaps: later analysis and FITS writing
apply their normal axis assumptions to the current data. In PV mode, ``v``
and ``dv`` follow processed y even if it now contains positions.

For PV plots, an effective beam uses the actual velocity spacing before
its angle is transformed. ``beam_org`` preserves the original sky beam.
Most external uses need one dataset; PlotAstroData also uses multiple datasets
internally, whose processed coordinate grids must agree.
``todict()`` does not copy contained lists or arrays.

Overlays and display
--------------------

Angles are degrees from the original data +y axis toward +x. Apply additional
rotation in that frame before ``swapxy``, which exchanges direction components
(equivalently, ``90 - angle``). ``xflip`` and ``yflip`` affect only the display.
Regions, lines, arrows, and polarization segments follow this convention.
``add_beam`` always accepts processed-frame angles, including explicit beams;
an already transformed beam must not be swapped again.

Sky-coordinate positions require a center, are scaled by ``dist``, and are
exchanged under ``swapxy``. Relative numeric positions refer to the displayed
axes: 0 is left/bottom and 1 is right/top. They are never exchanged, including
``beampos``. RGB display uses lower origin and pixel-edge extents to align
with color and contour maps.

``add_scalebar`` requires a finite positive ``length`` in horizontal data-axis
units. PV diagrams do not support scalebars, regions, lines, arrows, or segment
maps. Relative markers and text remain available.
``add_segment`` uses a dimensionless quiver scale relative to the displayed
x pixel spacing; see its API description for amplitude normalization.

Supply both ``fig`` and ``ax`` to ``PlotAstroData``, or omit both. Supplied
axes support one panel on one page. In an animation, clear the figure (including
colorbar axes), create a new axes, and pass the pair for each frame, as in the
gallery example.

R.A./Dec. tick formatting requires ``dist=1`` and non-PV data; ``swapxy`` is
supported. ``plot3d`` requires ``dist=1``, ``swapxy=False``, a non-PV cube,
uniformly spaced axes, and a finite positive scalar sigma. Its coordinates
are ascending to retain a right-handed system; display flips do not reverse
its axes.

Noise, fitting, and FITS regeneration
-------------------------------------

Histogram fitting supports unweighted histograms with equally spaced bins.
Both ``density=True`` and ``density=False`` are supported; fitting uses the
underlying counts in either case. Explicit uneven edges and weights are
rejected. Equal widths use ``rtol=1e-5, atol=0`` to accommodate float32 rounding.
The ``edges`` argument of ``noise_utils.normalize`` is keyword-only.

``gaussfit1d`` and ``gaussfit2d`` default to ``sigma=None``: they print a notice
and estimate noise from residuals of an initial fit. Returned scalar sigma is
a float, preserving extended precision; array-valued sigma retains its dtype.

``FitsData.get_data`` returns cached data. Its arguments are used only if data
have not yet been generated. Call ``gen_data`` to change conversion options::

    fd.gen_data(Tb=True)
    temperature = fd.get_data()
    fd.gen_data(Tb=False)
    original_units = fd.get_data()

Temperature and logarithmic conversions leave the source HDU array unchanged.
Repeated temperature conversion starts from that source and does not compound
the factor. Integer input is promoted when needed for temperature conversion.

For inverse real FFTs, supply the original x-axis sample count via ``nx`` when
needed to distinguish odd and even lengths; the half-spectrum alone does not
uniquely specify the original length. See the FFT API for individual signatures.
