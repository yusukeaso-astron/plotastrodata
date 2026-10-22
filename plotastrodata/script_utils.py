"""Generate short, editable plotting scripts using existing plotastrodata APIs."""
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from plotastrodata.analysis_utils import AstroData, AstroFrame


def _validate_fits_image(fitsimage: str | Path, *, ndim: int = 2) -> None:
    """Validate the header of a primary-HDU RA/Dec image or spectral cube.

    Internal callers use ndim=2 for an image or ndim=3 for a cube.
    Higher axes must be singleton. Read only the header and return None.
    Raise ValueError for unsuitable axes or nonpositive/nonfinite spatial
    pixel scales. No data pixels are loaded or modified.
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
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError('Expected finite positive spatial pixel scales.')


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

    Generation reads the primary FITS image and calls
    AstroData.estimate_ranges(pbcor=True) to fit noise and estimate the
    spatial half-width. It does not execute the generated script or modify
    the input FITS file.
    The generated script uses the existing AstroData, AstroFrame, Noise,
    and PlotAstroData APIs directly, following example.py.

    Args:
        fitsimage (str | Path): A primary-HDU RA/Dec image with at least
            two pixels on each spatial axis. Any higher axes must be singleton.
            Noise and range estimation assume PB-corrected intensities.
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
        Generation estimates rmax and writes it directly into the fitting
        and plotting frames, rounded to three significant digits as a
        Python float literal with an inline comment. Edit it in the script
        to change either field; no runtime range-selection helper is added.

        The generated script reads the full image with sigma=None, fits
        a hist-pbcor noise model, and saves <script stem>_noise.png. It then
        reads a separate cropped image with the fitted noise level and calls
        gaussfit2d(progressbar=True). It prints the sky center and east/north
        position uncertainties in arcsec, then centers the 2D image on the
        fitted sky position and saves <script stem>.png.
        Both PNGs are beside the script and follow the existing plotting
        methods' overwrite behavior. Noise statistics print with .3e format.
        Changing the generated noise fit does not recompute the inline range.
        Plotting uses stretch='log', show=False, and the default fonts,
        beam, and contour levels. Edit these choices directly in the script.
    """
    if not isinstance(overwrite, bool):
        raise TypeError('overwrite must be a boolean.')
    fitsimage = Path(fitsimage).resolve()
    output = Path(output or f'{fitsimage.stem}_image.py').resolve()
    if output.suffix != '.py':
        raise ValueError('output must have a .py suffix.')
    _validate_fits_image(fitsimage)
    image = AstroData(fitsimage=str(fitsimage), sigma=None)
    AstroFrame().read(image)
    rmax, _, _ = image.estimate_ranges(pbcor=True)
    rmax = float(f'{rmax:.3g}')
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
f_fit = AstroFrame(rmax={rmax!r})  # Estimated from PB-weighted emission (arcsec); adjust the fit region here.
f_fit.read(d_fit)
fit = d_fit.gaussfit2d(progressbar=True)
print('Gaussian center:', fit['center'])
print('Center uncertainties (east, north; arcsec):', fit['perr'][1:3])

# 2D image
p = PlotAstroData(rmax={rmax!r}, center=fit['center'])  # Estimated from PB-weighted emission (arcsec); adjust as needed.
p.add_color(**d.todict(), stretch='log')
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
    sections of example.py. Generation reads the FITS header and cube
    pixels and calls AstroData.estimate_ranges(pbcor=True) to fit
    edge-channel noise and estimate emission ranges. It does
    not execute the generated script or modify the input FITS file.

    Args:
        fitsimage (str | Path): Primary-HDU RA/Dec spectral cube with at least
            two pixels on each spatial axis and two frequency or velocity
            channels on axis 3, in Hz, m/s, or km/s. Higher axes must be
            singleton. Noise and range estimation assume PB-corrected
            intensities. Running the generated flux profile assumes Jy/beam
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
        through AstroFrame to crop it to the estimated spatial and velocity
        ranges, written directly in the script with inline comments.
        plotprofile integrates flux over that crop with a circle whose
        diameter is four times rmax, enclosing the entire square. Flux
        integration assumes Jy/beam intensities and requires beam metadata
        and spatial pixel sizes. The single-panel profile is saved as
        <script stem>_profile.png at intrinsic channel spacing.

        Generation estimates rmax, vmin, and vmax and writes them into
        both frames as Python float literals rounded to three significant
        digits, with inline comments. Edit them in the generated script
        to change the profile or channel-map region. Changing its noise
        fit does not recompute these values.

        Both profile and channel maps use the selected ranges at intrinsic
        channel spacing, without skipping or binning. They plot original
        PB-corrected data about the FITS reference center. The generated
        noise visualization still uses the full spatial footprint.

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
    _validate_fits_image(fitsimage, ndim=3)
    cube = AstroData(fitsimage=str(fitsimage), sigma=None)
    AstroFrame().read(cube)
    rmax, vmin, vmax = cube.estimate_ranges(pbcor=True)
    rmax = float(f'{rmax:.3g}')
    vmin, vmax = float(f'{vmin:.3g}'), float(f'{vmax:.3g}')
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
f_profile = AstroFrame(rmax={rmax!r},  # Estimated spatial half-width (arcsec); adjust as needed.
                       vmin={vmin!r}, vmax={vmax!r},  # Estimated velocity range (km/s); adjust as needed.
                       )
f_profile.read(d_profile)
plotprofile(**d_profile.todict(), xlist=[0], ylist=[0],
            ellipse=[4 * f_profile.rmax, 4 * f_profile.rmax, 0],  # Circle enclosing the cropped square.
            flux=True,
            savefig={str(output.with_name(output.stem + '_profile.png'))!r},
            show=False,
            )

# 3D channel maps
p = PlotAstroData(v=d.v, center=d.center,
                  rmax={rmax!r},  # Estimated spatial half-width (arcsec); adjust as needed.
                  vmin={vmin!r}, vmax={vmax!r},  # Estimated velocity range (km/s); adjust as needed.
                  )
p.add_color(**d.todict(), stretch='log')
p.add_contour(**d.todict(), colors='r')
p.set_axis()
p.savefig({str(output.with_suffix('.png'))!r}, show=False)
"""
    _write_script(source, output, overwrite=overwrite)
    return output
