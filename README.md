# plotastrodata
Python package to make figures from radio astronomical data by astropy and matplotlib.


## Demo and Usage
 
The file example.py will help to find how to use this package.
```bash
git clone https://github.com/yusukeaso-astron/plotastrodata
cd plotastrodata
python example.py
```
To keep the package updated, type the command below in the directory plotastrodata, always before you use.
```bash
git pull
```
Also, setting the path in .bashrc (or .zshrc etc.) will be useful.
```bash
export PYTHONPATH=${PYTHONPATH}:/YOUR_PATH_TO/plotastrodata
```
The Sphix html document is available from docs/build/index.html.
 
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
 
## Requirement

* astropy
* matplotlib
* numpy
* PIL (only for RGB figures)
* plotly (only for html cube)
* skimage (only for html cube)
* re
* scipy
* warnings

 
## Installation
 
Download from https://github.com/yusukeaso-astron/plotastrodata or git clone.
```bash 
git clone https://github.com/yusukeaso-astron/plotastrodata
```
 
## Note

* For 3D data, a 1D velocity array or a FITS file with a velocity axis must be given to set up channels in each page.
* For 2D/3D data, the spatial center can be read from a FITS file or manually given.
* len(v)=1 (default) means to make a 2D figure.
* Spatial lengths are in the unit of arcsec, or au if dist (!= 1) is given.
* Angles are in the unit of degree.
* For region, line, arrow, label, and marker, a single input can be treated without a list, e.g., anglelist=60, as well as anglelist=[60].
* Each element of poslist supposes a text coordinate like '01h23m45.6s 01d23m45.6s' or a list of relative x and y like [0.2, 0.3] (0 is left or bottom, 1 is right or top).
* Parameters for original methods in matplotlib.axes.Axes can be used as kwargs; see the default kwargs0 for reference.
* Position-velocity diagrams (pv=True) does not yet suppot region, line, arrow, and segment because the units of abscissa and ordinate are different.
 
## Author
 
* Name: Yusuke Aso
* Affiliation: Korea Astronomy and Space Science Institute
* E-mail: yaso@kasi.re.kr
 
## License
 
"plotastrodata" is under [GNU General Public License Version 3](https://www.gnu.org/licenses/gpl-3.0.html).
