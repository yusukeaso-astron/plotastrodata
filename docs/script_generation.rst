Simple 2D script generation
===========================

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
   f_fit = AstroFrame(rmax=3.0)  # One quarter of the valid half-width in arcsec; adjust the fit region here.
   f_fit.read(d_fit)
   fit = d_fit.gaussfit2d(progressbar=True)
   print("Gaussian center:", fit["center"])
   print("Center uncertainties (east, north; arcsec):", fit["perr"][1:3])

   # 2D image
   p = PlotAstroData(rmax=3.0, center=fit["center"])  # One quarter of the valid half-width in arcsec; adjust as needed.
   p.add_color(**d.todict(), stretch="log", cblabel=d.bunit)
   p.add_contour(**d.todict(), colors="r")
   p.set_axis_radec(nticksminor=5)
   p.savefig("image.png", show=False)

The default beam and sigma contour levels come from the existing plotting
methods. No text, markers, lines, arrows, or regions are added. Fonts keep the
library defaults. No brightness-temperature conversion is applied.

The generator starts with half the larger image dimension, using projected
WCS pixel scales in arcsec. If nonfinite pixels exist, it reduces this
half-width to a square centered on the FITS reference position that excludes
them, leaving a one-pixel margin for trimming. It then divides the half-width
by four. The resulting full field spans one quarter of that valid square's
width. This is a footprint estimate, not a measurement of source emission.

The estimated rmax is written as a Python float rounded to eight significant
digits in both the fitting frame and PlotAstroData (3.0 above is illustrative).
Edit these numbers directly to change the fitting or plotting region.
Generation reads the image pixels to estimate the crop, but does not run
noise estimation, Gaussian fitting, or plotting.

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
