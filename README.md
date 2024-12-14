# plotastrodata
Python package to make figures from radio astronomical data by astropy and matplotlib.


## Demo and Usage
For the installation, conda and are available.
```bash
conda install conda-forge::plotastrodata
```
or
```bash
pip install plotastrodata
```
The following is the way to install plotastrodata manually. The file example.py will help you find out how to use this package.
```bash
git clone https://github.com/yusukeaso-astron/plotastrodata
cd plotastrodata
python example.py
```
To keep the package updated, always type the command below in the directory plotastrodata before you use it.
```bash
git pull
```
Also, setting the path in .bashrc (or .zshrc, etc.) will be useful.
```bash
export PYTHONPATH=${PYTHONPATH}:/YOUR_PATH_TO/plotastrodata
```
Or directly in your script,
```Python
import  sys
sys.path.append( "/YOUR_PATH_TO/plotastrodata" )
```
The Sphinx html document is available from docs/_build/index.html or [readthedocs](https://plotastrodata.readthedocs.io/en/latest/).
 
## Features
 
plotastrodata can do the following things.
* Make 3D channel maps, 3D rotatable html cube, or 2D images including position-velocity diagrams.
* Color scale can be linear, log, and asinh.
* Make a figure of line profiles with Gaussian fitting.
* Plot images to externally given fig and ax (2D images only).
* Combine color, contour, segment, and RGB maps using images with different spatial grids.
* Input fits files or 2D/3D numpy arrays.
* Select the R.A.-Dec. style or the offset style as the x/y tick labels.
* Fill channel maps with a 2D image.
* Add line, arrow, ellipse, rectangle, text, and marker in specified channels.
* Use original arguments of matplotlib (pcolormesh, contour, quiver, plot, text, Ellipse, Rectangle).
* Other functions for plotting line profiles and a spatial 1D slice.
* example_advanced.py includes how to make a movie of channel maps.
* There are other utilities for Fourier transform and fitting.
 
## Requirement

* astropy
* corner (only for fitting)
* dynesty (only for fitting)
* ffmpeg (only for movie)
* matplotlib
* multiprocess (only for fitting)
* numpy
* pillow (only for RGB figures)
* plotly (only for html cube)
* ptemcee (only for fitting)
* scikit-image (only for html cube)
* scipy
* tqdm (only for fitting)

 
## Installation
 
Download from https://github.com/yusukeaso-astron/plotastrodata or git clone.
```bash 
git clone https://github.com/yusukeaso-astron/plotastrodata
```
 
## Note

* For 3D data, a 1D velocity array or a FITS file with a velocity axis must be given to set up channels on each page.
* For 2D/3D data, the spatial center can be read from a FITS file or manually given.
* len(v)=1 (default) means to make a 2D figure.
* Spatial lengths are in the unit of arcsec, or au if dist (!= 1) is given.
* Angles are in the unit of degree.
* For region, line, arrow, label, and marker, a single input can be treated without a list, e.g., anglelist=60, as well as anglelist=[60].
* Each element of poslist supposes a text coordinate like '01h23m45.6s 01d23m45.6s' or a list of relative x and y like [0.2, 0.3] (0 is left or bottom, 1 is right or top).
* Parameters for original methods in matplotlib.axes.Axes can be used as kwargs; see the default kwargs0 for reference.
* Position-velocity diagrams (pv=True) do not yet support region, line, arrow, and segment because the units of abscissa and ordinate are different.
 
## Author
 
* Name: Yusuke Aso
* Affiliation: Korea Astronomy and Space Science Institute
* E-mail: yaso@kasi.re.kr
 
## License
 
"plotastrodata" is under [GNU General Public License Version 3](https://www.gnu.org/licenses/gpl-3.0.html).
