# plotastrodata
Python package to make figures from radio astronomical data by astropy and matplotlib.
The API, examples, and gallery are available in [readthedocs](https://plotastrodata.readthedocs.io/en/latest/).
The PDF manual is available [here](https://github.com/yusukeaso-astron/plotastrodata/blob/main/plotastrodata_manual.pdf).


## Highlights

* Read, write, and process astronomical FITS images, spectral cubes, position-velocity diagrams, and NumPy arrays, including WCS coordinates, spectral axes, beam information, and brightness-temperature conversion.
* Create 2D maps, channel maps, position-velocity diagrams, RGB composites, line profiles, spatial slices, and interactive 3D isosurface visualizations.
* Overlay color, contour, polarization-vector, and RGB data from different spatial grids, with astronomical coordinates, beams, scale bars, regions, markers, and other annotations.
* Prepare data through trimming, interpolation, binning, masking, centering, rotation, deprojection, beam circularization, and profile extraction.
* Estimate image noise, including support for primary-beam-corrected data.
* Fit Gaussian models to line profiles and two-dimensional image components.
* Use supporting tools for coordinate and line-of-sight transformations, centered Fourier transforms and visibility sampling, and Bayesian or grid-based parameter estimation.

## Installation

Install from conda-forge:

```bash
conda install conda-forge::plotastrodata
```

or from PyPI:

```bash
pip install plotastrodata
```

To install the latest source from GitHub:

```bash
git clone https://github.com/yusukeaso-astron/plotastrodata
cd plotastrodata
python -m pip install .
```

To update a source installation, run the following commands in the cloned repository:

```bash
git pull
python -m pip install --upgrade .
```

## Demo and Usage

The repository includes [`example.py`](example.py), which demonstrates the main plotting and analysis features. After cloning the repository and installing the package, run:

```bash
python example.py
```

More examples and the complete API reference are available in the [online documentation](https://plotastrodata.readthedocs.io/en/latest/).
 
## Requirement

* Python >= 3.11
* astropy >= 7.2
* corner (only for fitting)
* dynesty (only for fitting)
* emcee >= 3.1.5 (only for fitting)
* ffmpeg (only for movie)
* matplotlib
* numpy >= 2.0
* pillow (only for RGB figures)
* plotly (only for html cube)
* ptemcee (only for fitting)
* pydantic >= 2
* scikit-image (only for html cube)
* scipy
* tqdm (only for fitting)

## Conventions and Limitations

* Array axes are ordered as `(y, x)` for a 2D image, `(v, y, x)` for a spectral cube, and `(v, x)` for a position-velocity diagram. With NumPy-array input, supply matching one-dimensional coordinate arrays; a spectral cube requires `v` to construct channel maps. FITS input obtains these coordinates from its header.
* Spatial coordinates are offsets from `center`. They are expressed in arcseconds when `dist=1`. When `dist` is set to the source distance in parsecs, spatial offsets and beam sizes are expressed in au.
* By default, the R.A. axis increases toward the left (`xflip=True`), following the usual astronomical image convention. Use `xflip=False` to reverse this behavior.
* Velocity coordinates and `vsys` are in km/s, while `restfreq` is in Hz. A frequency axis read from FITS is converted to velocity relative to `vsys`.
* Sky positions may be given as coordinate strings, such as `'ICRS 01h23m45.6s 01d23m45.6s'`. Numeric positions supplied through `poslist`, such as `[0.2, 0.3]`, are relative plotting positions: 0 is the left or bottom edge and 1 is the right or top edge. They are not spatial offsets.
* Angles are in degrees unless an API explicitly states otherwise. Sky position angles are measured from north toward east. The polar-coordinate arrays `theta` and `phi` used by the line-of-sight utilities are in radians.
* A beam is represented by `[bmaj, bmin, bpa]`, where the first two values use the current spatial unit and `bpa` is in degrees. `Tb=True` converts data from Jy/beam to brightness temperature and requires a valid rest frequency and spatial-resolution metadata.
* `sigma` may be a numeric RMS value, a string selecting a noise-estimation method, or `None` to disable noise-dependent behavior. See the API documentation for the available estimation methods.
* Annotation methods accept a single value or a list of values. `include_chan` selects displayed channels by zero-based index; `None` applies the annotation to all channels.
* Position-velocity mode (`pv=True`) does not support regions, lines, arrows, or segment maps because its spatial and velocity axes use different units. Markers and text may still be placed using relative plotting positions.
* Constants in `const_utils` are plain numerical values in SI units, not `astropy.units.Quantity` objects.
 
## Author
 
* Name: Yusuke Aso
* Affiliation: Korea Astronomy and Space Science Institute
* E-mail: yaso@kasi.re.kr
 
## License
 
"plotastrodata" is under [GNU General Public License Version 3](https://www.gnu.org/licenses/gpl-3.0.html).
