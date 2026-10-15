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

Validated integer count arguments accept Python and NumPy integers and reject booleans,
floats (including `2.0`), and zero-dimensional arrays. Most counts must be positive;
`veldigit`, MCMC `nburnin`, and automatic `plotprofile(nrows=0)` allow zero.
`nwalkersperdim` and inverse-real-FFT `nx` must be at least two. Invalid counts
raise `ValueError`.

Scalar-only numeric checks also reject booleans, complex values, and arrays;
validation preserves extended floating precision. Range and optional-value
rules depend on the argument. `AstroData.slice` requires an explicit positive
`length`; zero is rejected.

See [Conventions and usage notes](docs/conventions.rst) for data formats, coordinate and angle conventions, runtime types, and supported usage.

## Author
 
* Name: Yusuke Aso
* Affiliation: Korea Astronomy and Space Science Institute
* E-mail: yaso@kasi.re.kr
 
## License
 
"plotastrodata" is under [GNU General Public License Version 3](https://www.gnu.org/licenses/gpl-3.0.html).
