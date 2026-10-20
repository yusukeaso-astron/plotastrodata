"""Generate short, editable plotting scripts using existing plotastrodata APIs."""
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from plotastrodata.analysis_utils import AstroData, AstroFrame


def _get_image_radius(fitsimage: str | Path) -> float:
    """Validate a primary-HDU 2D RA/Dec image and return its half-width.

    Read only the header. The result is half the larger image dimension
    in arcsec, using projected WCS pixel scales, as a Python float.
    Raise ValueError for unsuitable image axes or spatial pixel scales.
    """
    with fits.open(fitsimage, memmap=True) as hdus:
        h = hdus[0].header
        n = int(h.get('NAXIS', 0))
        if (n < 2 or h.get('NAXIS1', 0) < 2 or h.get('NAXIS2', 0) < 2
                or any(h[f'NAXIS{i}'] != 1 for i in range(3, n + 1))
                or not str(h.get('CTYPE1', '')).upper().startswith('RA')
                or not str(h.get('CTYPE2', '')).upper().startswith('DEC')):
            raise ValueError('Expected a 2D RA/Dec image'
                             + ' in the primary FITS HDU.')
        scales = proj_plane_pixel_scales(WCS(h, naxis=2, fix=False).celestial)
        rmax = max(h['NAXIS1'] * scales[0], h['NAXIS2'] * scales[1]) * 3600 / 2
        if not np.isfinite(rmax) or rmax <= 0:
            raise ValueError('Expected finite positive spatial pixel scales.')
    return float(rmax)


def generate_image_script(fitsimage: str | Path,
                          output: str | Path | None = None,
                          *, overwrite: bool = False) -> Path:
    """Write an editable noise, Gaussian-fit, and 2D image plotting script.

    Generation reads the primary FITS image to estimate a finite fitting
    region, but does not execute the script or modify the input file.
    The generated script uses the existing AstroData, AstroFrame, Noise,
    and PlotAstroData APIs directly, following example.py.

    Args:
        fitsimage (str | Path): A primary-HDU RA/Dec image with at least
            two pixels on each spatial axis. Any higher axes must be singleton.
        output (str | Path | None): Script path with a .py suffix. Defaults
            to <input stem>_image.py in the current working directory.
            Relative paths are resolved at generation time.
        overwrite (bool): Allow replacing an existing script. Defaults to
            False. This option does not control figure overwriting.

    Returns:
        Path: Absolute path of the written script.

    Raises:
        TypeError: overwrite is not a Python bool.
        ValueError: The output path, image axes, spatial pixel scales, or
            centered finite fitting region are unsuitable.
        FileExistsError: The script exists and overwrite is False.
        OSError: The input cannot be read or the script cannot be written.

    Notes:
        The initial half-width is half the larger image dimension in arcsec,
        using projected WCS pixel scales. If nonfinite pixels are present,
        it is reduced to a square around the FITS reference position that
        excludes them, with one spatial pixel of margin. That half-width
        is divided by four and rounded to eight significant digits for the
        inline Python float used by the fitting frame and plotting frame.
        Thus the full field width is one quarter of the initial valid width.
        This is a footprint estimate, not a measurement of source emission.

        The generated script reads the full image with sigma=None, fits
        a hist-pbcor noise model, and saves <script stem>_noise.png. It then
        reads a separate cropped image with the fitted noise level and calls
        gaussfit2d(progressbar=True). It prints the sky center and east/north
        position uncertainties in arcsec, then centers the 2D image on the
        fitted sky position and saves <script stem>.png.
        Both PNGs are beside the script and follow the existing plotting
        methods' overwrite behavior. Noise statistics print with .3e format.
        Plotting uses stretch='log', show=False, and the default fonts,
        beam, and contour levels. Edit these choices directly in the script.
    """
    if not isinstance(overwrite, bool):
        raise TypeError('overwrite must be a boolean.')
    fitsimage = Path(fitsimage).resolve()
    output = Path(output or f'{fitsimage.stem}_image.py').resolve()
    if output.suffix != '.py':
        raise ValueError('output must have a .py suffix.')
    rmax = _get_image_radius(fitsimage)
    image = AstroData(fitsimage=str(fitsimage), sigma=None)
    AstroFrame().read(image)
    invalid_y, invalid_x = np.nonzero(~np.isfinite(image.data))
    if invalid_x.size:
        nearest_invalid = np.min(np.maximum(np.abs(image.x[invalid_x]),
                                            np.abs(image.y[invalid_y])))
        margin = max(abs(image.dx), abs(image.dy))
        rmax = min(rmax, nearest_invalid - margin)
        if not np.isfinite(rmax) or rmax <= margin:
            raise ValueError('No centered finite image region large enough'
                             + ' for fitting.')
    rmax = float(f'{rmax / 4:.8g}')
    imports = ('from plotastrodata.analysis_utils import AstroData, AstroFrame\n'
               'from plotastrodata.noise_utils import Noise\n'
               'from plotastrodata.plot_utils import PlotAstroData\n')
    source = imports + f'''

# Read the image
d = AstroData(fitsimage={str(fitsimage)!r}, sigma=None)  # Estimate noise below.
f = AstroFrame()
f.read(d)

# Noise visualization
n = Noise(data=d.data, sigma='hist-pbcor')
n.gen_histogram()
n.fit_histogram()
print(f'Noise (mean, std): {{n.mean:.3e}}, {{n.std:.3e}}')
n.plot_histogram(savefig={str(output.with_name(output.stem + '_noise.png'))!r}, show=False)

d.sigma = n.std  # A float number in the data unit is acceptable.

# 2D Gaussian fitting
d_fit = AstroData(fitsimage={str(fitsimage)!r}, sigma=d.sigma)
f_fit = AstroFrame(rmax={rmax!r})  # One quarter of the valid half-width in arcsec; adjust the fit region here.
f_fit.read(d_fit)
fit = d_fit.gaussfit2d(progressbar=True)
print('Gaussian center:', fit['center'])
print('Center uncertainties (east, north; arcsec):', fit['perr'][1:3])

# 2D image
p = PlotAstroData(rmax={rmax!r}, center=fit['center'])  # One quarter of the valid half-width in arcsec; adjust as needed.
p.add_color(**d.todict(), stretch='log', cblabel=d.bunit)
p.add_contour(**d.todict(), colors='r')
p.set_axis_radec(nticksminor=5)
p.savefig({str(output.with_suffix('.png'))!r}, show=False)
'''
    compile(source, str(output), 'exec')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w' if overwrite else 'x') as stream:
        stream.write(source)
    return output
