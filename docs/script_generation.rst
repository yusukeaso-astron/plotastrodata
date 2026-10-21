Script generation
=================

``plotastrodata.script_utils.generate_image_script`` writes a short, editable
script following the "2D image" section of ``example.py``. It calls existing
plotastrodata APIs directly. The generated script has no helper functions,
configuration block or automatic source selection. A noise histogram and
a 2D Gaussian fit precede the 2D image.

.. code-block:: python

   from plotastrodata.script_utils import generate_image_script

   script = generate_image_script(
       "image.fits", output="image.py",
   )

Review the values where they appear in the script, then run:

.. code-block:: bash

   python image.py

The script uses the existing noise and image plotting methods:

.. code-block:: python

   from plotastrodata.noise_utils import Noise
   from plotastrodata.analysis_utils import AstroData, AstroFrame
   from plotastrodata.plot_utils import PlotAstroData

   # Read the image
   d = AstroData(fitsimage="image.fits", sigma=None)  # Estimate noise below.
   f = AstroFrame()
   f.read(d)

   # Noise visualization
   n = Noise(data=d.data, sigma="hist-pbcor")
   n.gen_histogram()
   n.fit_histogram()
   print(f"Noise (mean, std): {n.mean:.3e}, {n.std:.3e}")
   n.plot_histogram(savefig="image_noise.png", show=False)

   d.sigma = n.std  # A float number in the data unit is acceptable.

   # 2D Gaussian fitting
   d_fit = AstroData(fitsimage="image.fits", sigma=d.sigma)
   f_fit = AstroFrame(rmax=3.0)  # Estimated from PB-weighted emission in arcsec; adjust the fit region here.
   f_fit.read(d_fit)
   fit = d_fit.gaussfit2d(progressbar=True)
   print("Gaussian center:", fit["center"])
   print("Center uncertainties (east, north; arcsec):", fit["perr"][1:3])

   # 2D image
   p = PlotAstroData(rmax=3.0, center=fit["center"])  # Estimated from PB-weighted emission in arcsec; adjust as needed.
   p.add_color(**d.todict(), stretch="log", cblabel=d.bunit)
   p.add_contour(**d.todict(), colors="r")
   p.set_axis_radec(nticksminor=5)
   p.savefig("image.png", show=False)

The default beam and sigma contour levels come from the existing plotting
methods. No text, markers, lines, arrows, or regions are added. Fonts keep the
library defaults. No brightness-temperature conversion is applied.

The generator estimates rmax and writes it into the fitting and plotting
frames as an inline number rounded to three significant digits (3.0 above
is illustrative). Edit these values directly to change the regions.
Generation reads the input data and fits noise to estimate ranges; it does
not execute the generated script or make its figures.

The generated script reads the full image with AstroFrame for noise
estimation. Noise uses ``sigma="hist-pbcor"``, and its fitted standard deviation
is assigned to ``d.sigma``. Noise mean and standard deviation print in
``.3e`` format. A second image read applies the smaller fitting region and
reuses that noise level.

The Gaussian fit uses AstroData.gaussfit2d with its existing beam noise
correlation correction and enables the MCMC progress bar. It reports sky
coordinates and fitted position uncertainties along east and north
(RA multiplied by cos(Dec), and Dec), in arcsec. The 2D image is centered on the fitted sky position. NaN and infinite
pixels are rejected by the fitter. Fitting large images can take substantial time;
edit the fitting frame's rmax to choose a smaller region if needed.

Input must be a primary-HDU RA/Dec image with at least two pixels on each
spatial axis; higher axes must be singleton. Input and output paths in the
script are absolute. The generator returns the absolute script path as a
``pathlib.Path``. An existing script requires ``overwrite=True`` to replace it.

Both PNGs are saved beside the script: ``<script stem>_noise.png`` and
``<script stem>.png``. Figure overwriting follows the existing plotting
methods independently of the generator's overwrite option.
The script writes ``stretch="log"`` and ``show=False`` directly;
edit the plotting calls for another intensity scale or interactive display.


Cube scripts
------------

``plotastrodata.script_utils.generate_cube_script`` writes a noise
visualization, an integrated line profile, and 3D channel maps using the
corresponding sections of ``example.py``:

.. code-block:: python

   from plotastrodata.script_utils import generate_cube_script

   script = generate_cube_script("cube.fits", output="cube.py")

Review cube.py, then run it. Generation checks the primary FITS header and
reads cube pixels to estimate the plotting ranges.
The input must have RA/Dec spatial axes with at least two pixels each, at
least two spectral channels on axis 3 (frequency in Hz or velocity in m/s
or km/s), and only singleton higher axes. Running the flux-profile section
assumes Jy/beam intensities and requires complete beam metadata and spatial
pixel sizes. Generation fits noise and selects emission ranges, but does
not execute the script or make its figures.

The script reads the full cube with AstroFrame.read and estimates noise using
``Noise(data=d.data, sigma="edge,hist-pbcor")``. This selects the first and last
channels and fits the PB-corrected histogram model. Edit that selection if
the edge channels contain emission. The script saves ``<script stem>_noise.png``,
prints mean and standard deviation with .3e formatting, and reuses the fitted
standard deviation for the channel-map colors and contours.

Between the noise and channel maps, the script crops a separate AstroData
instance that shares the loaded data to the estimated spatial and velocity
ranges using AstroFrame.read, then calls
plotprofile with ``flux=True``. A circle with diameter four times the crop's
rmax encloses the entire square, integrating flux from all cropped pixels.
This assumes Jy/beam intensities and requires beam metadata and pixel sizes.
The profile uses the intrinsic channel spacing in one panel and is saved
as ``<script stem>_profile.png``. Its crop can be edited independently of
the channel-map field. The full cube remains available for the channel maps.

The profile and channel-map sections use the existing plotting functions:

.. code-block:: python

   from plotastrodata.plot_utils import PlotAstroData, plotprofile

   # Line profile
   d_profile = AstroData(**d.todict())
   f_profile = AstroFrame(
       rmax=3.0,  # Estimated spatial half-width (arcsec); adjust as needed.
       vmin=1.0, vmax=8.0,  # Estimated velocity range (km/s); adjust as needed.
   )
   f_profile.read(d_profile)
   plotprofile(**d_profile.todict(), xlist=[0], ylist=[0],
               ellipse=[4 * f_profile.rmax, 4 * f_profile.rmax, 0],  # Circle enclosing the cropped square.
               flux=True,
               savefig="cube_profile.png",
               show=False,
               )

   # 3D channel maps
   p = PlotAstroData(
       v=d.v, center=d.center,
       rmax=3.0,  # Estimated spatial half-width (arcsec); adjust as needed.
       vmin=1.0, vmax=8.0,  # Estimated velocity range (km/s); adjust as needed.
   )
   p.add_color(**d.todict(), stretch="log", cblabel=d.bunit)
   p.add_contour(**d.todict(), colors="r")
   p.set_axis()
   p.savefig("cube.png", show=False)

The generator estimates rmax, vmin, and vmax and writes them directly
into both frames, rounded to three significant digits with inline comments.
The example values above are illustrative. Edit them to change the profile
or channel-map region. Changing the script's noise fit does not recompute
the fixed range estimates.

Both plots use the estimated velocity interval at native channel spacing,
without skipping or binning. The library supplies the default panel layout,
velocity labels, fonts, beam, and contour levels. No grid, title, or
annotation overlays are added. Figures plot the original PB-corrected data
about the FITS reference center. Multiple channel-map pages are saved as
cube_0.png, cube_1.png, and so on; one page is saved as cube.png.

The generator returns an absolute pathlib.Path and writes absolute input
and figure paths in the script. It accepts only the input
path, optional output path, and keyword-only overwrite flag. An existing
script requires ``overwrite=True`` to replace it; figure overwriting follows
the existing plotting methods independently. Change plotting and noise
choices directly in the generated script.

An unsuitable cube header, output suffix, or centered finite region raises
ValueError during generation. Non-boolean overwrite values raise TypeError;
only a Python bool is accepted. Generation writes the script but does not
execute it. The returned Path is the sole public result: numeric estimates
are serialized as Python float literals, and the input arrays and FITS file
are not modified.
