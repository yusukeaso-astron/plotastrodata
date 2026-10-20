"""Generate short, editable plotting scripts using existing plotastrodata APIs."""
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from plotastrodata.analysis_utils import AstroData, AstroFrame


def _get_image_radius(fitsimage: str | Path, *, ndim: int = 2) -> float:
    """Validate a primary-HDU RA/Dec image or cube and return its half-width.

    Read only the header. The result is half the larger image dimension
    in arcsec, using projected WCS pixel scales, as a Python float.
    Internal callers use ndim=2 for an image or ndim=3 for a spectral
    cube. Higher axes must be singleton. Raise ValueError for unsuitable axes or spatial pixel scales.
    """
    with fits.open(fitsimage, memmap=True) as hdus:
        h = hdus[0].header
        n = int(h.get('NAXIS', 0))
        if (n < ndim or h.get('NAXIS1', 0) < 2 or h.get('NAXIS2', 0) < 2
                or any(h[f'NAXIS{i}'] != 1 for i in range(ndim + 1, n + 1))
                or not str(h.get('CTYPE1', '')).upper().startswith('RA')
                or not str(h.get('CTYPE2', '')).upper().startswith('DEC')):
            raise ValueError(f'Expected a {ndim}D RA/Dec image'
                             + ' in the primary FITS HDU.')
        if ndim == 3 and (
                h.get('NAXIS3', 0) < 2
                or not str(h.get('CTYPE3', '')).upper().startswith(
                    ('FREQ', 'VRAD', 'VELO', 'VOPT', 'FELO'))
                or str(h.get('CUNIT3', '')).strip().upper() not in
                ('HZ', 'M/S', 'KM/S')):
            raise ValueError('Expected at least two frequency'
                             + ' or velocity channels on the third FITS axis,'
                             + ' in Hz, m/s, or km/s.')
        scales = proj_plane_pixel_scales(WCS(h, naxis=2, fix=False).celestial)
        rmax = max(h['NAXIS1'] * scales[0], h['NAXIS2'] * scales[1]) * 3600 / 2
        if not np.isfinite(rmax) or rmax <= 0:
            raise ValueError('Expected finite positive spatial pixel scales.')
    return float(rmax)


def _get_valid_radius(image: AstroData, rmax: float) -> float:
    """Reduce a half-width to exclude nonfinite pixels, with a pixel margin.

    image must contain one 2D image or 3D cube already read by AstroFrame,
    with spatial coordinates and pixel spacings. rmax uses those same
    spatial units. For cubes, a
    spatial pixel is invalid if any channel is nonfinite. Return the
    half-width as a Python float; quarter-size scaling is left to the caller.
    Raise ValueError if no sufficiently large centered finite region remains.
    """
    if image.data.ndim == 2:
        valid = np.isfinite(image.data)
    else:
        valid = np.ones(image.data.shape[-2:], dtype=bool)
        for channel in image.data:
            valid &= np.isfinite(channel)
    invalid_y, invalid_x = np.nonzero(~valid)
    if invalid_x.size:
        nearest_invalid = np.min(np.maximum(np.abs(image.x[invalid_x]),
                                            np.abs(image.y[invalid_y])))
        margin = max(abs(image.dx), abs(image.dy))
        rmax = min(rmax, nearest_invalid - margin)
        if not np.isfinite(rmax) or rmax <= margin:
            raise ValueError('No centered finite spatial region large enough.')
    return float(rmax)


def _write_script(source: str, output: Path, *, overwrite: bool = False
                  ) -> None:
    """Check Python syntax and write source, creating parent directories.

    Compilation does not execute the source. Existing scripts raise
    FileExistsError unless overwrite is True. SyntaxError is raised before
    directories or files are created when source is invalid. Other I/O
    errors propagate. output is a Path; return None.
    """
    compile(source, str(output), 'exec')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w' if overwrite else 'x') as stream:
        stream.write(source)


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
    rmax = _get_valid_radius(image, rmax)
    rmax = float(f'{rmax / 4:.8g}')
    source = f'''
from plotastrodata.analysis_utils import AstroData, AstroFrame
from plotastrodata.noise_utils import Noise
from plotastrodata.plot_utils import PlotAstroData


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
    _write_script(source, output, overwrite=overwrite)
    return output


def generate_cube_script(fitsimage: str | Path,
                         output: str | Path | None = None,
                         *, overwrite: bool = False) -> Path:
    """Write noise, integrated line-profile, and 3D channel-map plotting code.

    Use the existing AstroData, AstroFrame, Noise, plotprofile, and
    PlotAstroData APIs, following the line-profile and 3D channel-map
    sections of example.py. Generation reads
    the FITS header and cube pixels, but does not execute the script.

    Args:
        fitsimage (str | Path): Primary-HDU RA/Dec spectral cube with at least
            two pixels on each spatial axis and two frequency or velocity
            channels on axis 3, in Hz, m/s, or km/s. Higher axes must be
            singleton. Running the generated flux profile assumes Jy/beam
            intensities and requires a complete beam and spatial pixel sizes.
        output (str | Path | None): Script path with a .py suffix. Defaults
            to <input stem>_cube.py in the current working directory.
            Relative input and output paths are resolved at generation time.
        overwrite (bool): Permit replacing an existing script. Defaults to
            False. This does not control figure overwriting.

    Returns:
        Path: Absolute path of the written script.

    Raises:
        TypeError: overwrite is not a Python bool.
        ValueError: The output suffix, cube axes, spatial pixel scales, or
            centered finite spatial region are unsuitable.
        FileExistsError: The script exists and overwrite is False.
        OSError: The input cannot be read or the script cannot be written.

    Notes:
        The script reads the full cube with sigma=None. Noise uses
        edge,hist-pbcor (first and last spectral channels); the fitted
        standard deviation is reused for colors and contours. Statistics
        print with .3e format and the histogram is saved as
        <script stem>_noise.png beside the script.

        A separate AstroData instance reuses the loaded cube and is read
        through AstroFrame to crop it to the spatial quarter-size.
        plotprofile integrates flux over that crop with a circle whose
        diameter is four times rmax, enclosing the entire square. Flux
        integration assumes Jy/beam intensities and requires beam metadata
        and spatial pixel sizes. The single-panel profile is saved as
        <script stem>_profile.png at intrinsic channel spacing.

        Channel maps use the full spectral range at intrinsic channel
        spacing, without skipping or binning. Spatial rmax starts from
        half the larger image dimension in arcsec. A spatial pixel is
        invalid if any channel there is nonfinite. If invalid pixels exist,
        the centered square is reduced to exclude them with one spatial
        pixel of margin. This valid half-width is divided by four and
        rounded to eight significant digits as an inline Python float
        for the profile crop and channel maps. The full field width is one
        quarter of the estimated valid width. This estimates the footprint,
        not the source emission. Both regions use the FITS reference center;
        noise uses the full cube.

        The script sets stretch='log' and show=False, with default fonts,
        beams, contours, and channel layout. Edit these choices directly.
        Channel-map pages are saved beside the script as <script stem>.png;
        multiple pages receive _0, _1, ... suffixes before .png, following
        PlotAstroData.savefig. Existing figures may be overwritten.
    """
    if not isinstance(overwrite, bool):
        raise TypeError('overwrite must be a boolean.')
    fitsimage = Path(fitsimage).resolve()
    output = Path(output or f'{fitsimage.stem}_cube.py').resolve()
    if output.suffix != '.py':
        raise ValueError('output must have a .py suffix.')
    rmax = _get_image_radius(fitsimage, ndim=3)
    cube = AstroData(fitsimage=str(fitsimage), sigma=None)
    AstroFrame().read(cube)
    rmax = _get_valid_radius(cube, rmax)
    rmax = float(f'{rmax / 4:.8g}')
    source = f"""
from plotastrodata.analysis_utils import AstroData, AstroFrame
from plotastrodata.noise_utils import Noise
from plotastrodata.plot_utils import PlotAstroData, plotprofile


# Read the cube
d = AstroData(fitsimage={str(fitsimage)!r}, sigma=None)  # Estimate noise below.
f = AstroFrame()
f.read(d)

# Noise visualization
n = Noise(data=d.data, sigma='edge,hist-pbcor')  # First and last channels; adjust if they contain emission.
n.gen_histogram()
n.fit_histogram()
print(f'Noise (mean, std): {{n.mean:.3e}}, {{n.std:.3e}}')
n.plot_histogram(savefig={str(output.with_name(output.stem + '_noise.png'))!r}, show=False)

d.sigma = n.std  # A float number in the data unit is acceptable.

# Line profile
d_profile = AstroData(**d.todict())
f_profile = AstroFrame(rmax={rmax!r})  # One quarter of the valid half-width in arcsec; adjust as needed.
f_profile.read(d_profile)
plotprofile(**d_profile.todict(), xlist=[0], ylist=[0],
            ellipse=[4 * f_profile.rmax, 4 * f_profile.rmax, 0],  # Circle enclosing the cropped square.
            flux=True,
            savefig={str(output.with_name(output.stem + '_profile.png'))!r},
            show=False,
            )

# 3D channel maps
p = PlotAstroData(v=d.v, rmax={rmax!r}, center=d.center)  # One quarter of the valid half-width in arcsec; adjust as needed.
p.add_color(**d.todict(), stretch='log', cblabel=d.bunit)
p.add_contour(**d.todict(), colors='r')
p.set_axis()
p.savefig({str(output.with_suffix('.png'))!r}, show=False)
"""
    _write_script(source, output, overwrite=overwrite)
    return output
