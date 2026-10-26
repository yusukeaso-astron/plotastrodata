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

### Released version

Install from conda-forge:

```bash
conda install conda-forge::plotastrodata
```

or from PyPI:

```bash
pip install plotastrodata
```

### Development version from GitHub

Clone the repository to install the latest development version, which may
include changes not yet available in the published releases:

```bash
git clone https://github.com/yusukeaso-astron/plotastrodata
(cd plotastrodata)
(python -m pip install .)
```

To update the development version, run the following commands in the cloned repository:

```bash
git pull
(python -m pip install --upgrade .)
```

## Demo and Usage

The repository includes [`example.py`](example.py), which demonstrates the main plotting and analysis features. After cloning the repository and installing the package, run:

```bash
python example.py
```

More examples and the complete API reference are available in the [online documentation](https://plotastrodata.readthedocs.io/en/latest/).
 
## Generate plotting scripts

```python
from plotastrodata.script_utils import generate_image_script

script = generate_image_script(
    "image.fits", output="image.py",
)
```

Review the inline values, then run `python image.py`. The script uses existing
library APIs to visualize a hist-pbcor noise fit, fit a 2D Gaussian and report
its sky center and east/north uncertainties, and plot a logarithmic image
with contours, a beam, and RA/Dec axes. It saves separate noise and image PNGs.
Noise estimation uses the full image. Fitting and plotting ranges are estimated automatically and written inline
with three significant digits for editing.
For a spectral cube, use `generate_cube_script("cube.fits", output="cube.py")`
from the same module. It generates an edge-channel noise visualization, a flux-integrated line
profile and 3D channel maps over estimated spatial and velocity ranges,
with intrinsic channel spacing. Two further single-panel figures show moments
1 and 2 in linear colors with moment-0 contours and colorbar ticks to one
decimal place; moment 1 uses a blue-red colormap. The moments use the shared crop and are saved as
`<script stem>_mom1.png` and `<script stem>_mom2.png`. The flux profile assumes Jy/beam data with beam metadata.
Noise and range estimation in both generators assume PB-corrected intensities.
Both generators return an absolute pathlib.Path and write the script
without executing it.
See the [script-generation guide](docs/script_generation.rst).

Ranges can also be obtained from loaded data with
`rmax, vmin, vmax = d.estimate_ranges(pbcor=True)` for PB-corrected data.
For uniform-noise data, use `d.estimate_ranges()`. Noise is fitted when
`d.sigma=None`; a supplied positive scalar assumes a zero baseline unless
`pbcor=True`. For PB-corrected data, supplied sigma describes noise before
PB correction. The array-based function is `noise_utils.estimate_ranges`,
with the same `pbcor` option and optional `sigma`. Here, `rmax` is a symmetric
spatial half-width; 2D images return `None` for both velocity limits.

## Combining saved figures

Use `concat_pngs` or `concat_pdfs` to place saved figures in a grid:

```python
from plotastrodata.pngpdf_utils import concat_pngs, concat_pdfs

concat_pngs(["a.png", "b.png"], "grid.png", columns=2, padding=10)
concat_pdfs(["a.pdf", "b.pdf"], "grid.pdf", columns=2, padding=10)
```

Both use direct scaling, whole-unit rounding, and centered placement. Padding
is measured in pixels for PNG and points for PDF; PNG DPI metadata is ignored.
Matching numeric source dimensions produce matching grid geometry. Optional
`scale_factors` accepts one multiplier per input or a nested grid with missing
entries defaulting to 1. PDF grids use only the first page of each unencrypted
input and preserve vector content. PNG supports transparent backgrounds.
Pillow and pypdf are installed with plotastrodata.

## Requirement

* Python >= 3.11
* astropy >= 7.2
* corner (only for fitting)
* dynesty (only for fitting)
* emcee >= 3.1.5 (only for fitting)
* ffmpeg (only for movie)
* matplotlib
* numpy >= 2.0
* pillow (for RGB figures and PNG grids)
* plotly (only for html cube)
* ptemcee (only for fitting)
* pydantic >= 2
* pypdf >= 6.21.2 (for PDF grids)
* scikit-image (only for html cube)
* scipy
* tqdm (only for fitting)

## Conventions and Limitations

See [Conventions and usage notes](docs/conventions.rst) for data formats, coordinate and angle conventions, runtime types, input validation, and usage limitations.

## Author
 
* Name: Yusuke Aso
* Affiliation: Korea Astronomy and Space Science Institute
* E-mail: yaso@kasi.re.kr
 
## License
 
"plotastrodata" is under [GNU General Public License Version 3](https://www.gnu.org/licenses/gpl-3.0.html).
