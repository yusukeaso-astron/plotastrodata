"""Generate short, editable plotting scripts using existing plotastrodata APIs."""
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy import ndimage

from plotastrodata._type_utils import RealScalar, RealSequence, _normalize_float
from plotastrodata._validation_utils import (
    _is_uniform_spacing,
    _require_positive_real_scalar,
)
from plotastrodata.analysis_utils import AstroData, AstroFrame
from plotastrodata.noise_utils import Noise


def _multiply_pb(data: np.ndarray, x: np.ndarray, y: np.ndarray,
                 R: float | np.integer | np.floating,
                 r_out: float | np.integer | np.floating) -> np.ndarray:
    """Multiply an image or cube by an approximate Gaussian primary beam.

    Args:
        data: Numeric array with shape (ny, nx) or (nchan, ny, nx).
        x: Finite real 1D x offsets from the assumed PB center, length nx.
        y: Finite real 1D y offsets from the assumed PB center, length ny.
        R: Positive dimensionless outer radius r_out / r_p from gauss_pbcor
            (Noise.popt[2]), where r_p is the primary-beam HWHM.
        r_out: Positive outer radius of the footprint used for the noise fit,
            in the same spatial units as x and y. It is supplied explicitly,
            not inferred from the array shape or the plotting crop.

    Returns:
        np.ndarray: New array of the same shape, multiplied by
        B(x, y) = 2**(-R**2 * ((x/r_out)**2 + (y/r_out)**2)).
        One spatial pattern is broadcast across all channels. Inputs are
        not modified and nonfinite data pixels are not filled or removed.
        NumPy determines the arithmetic dtype; integer data can become
        floating point, and extended precision is retained without casting.

    Raises:
        TypeError: data is not a NumPy array.
        ValueError: Shapes, coordinate values, or PB parameters are unsuitable.

    Notes:
        Applying this to PB-corrected data approximately restores the
        pre-correction intensities. The assumed circular beam and center
        are intended for range estimation, not calibrated PB reconstruction.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError('data must be a NumPy array.')
    if (data.ndim not in (2, 3) or 0 in data.shape
            or not np.issubdtype(data.dtype, np.number)):
        raise ValueError('data must be a nonempty numeric 2D or 3D array.')
    x, y = np.asarray(x), np.asarray(y)
    for coordinate, length, name in ((x, data.shape[-1], 'x'),
                                     (y, data.shape[-2], 'y')):
        if (coordinate.ndim != 1 or len(coordinate) != length
                or coordinate.dtype.kind not in 'iuf'
                or not np.all(np.isfinite(coordinate))):
            raise ValueError(f'{name} must be a finite real 1D coordinate array'
                             + ' matching the spatial data axis.')
    _require_positive_real_scalar(R, 'R')
    _require_positive_real_scalar(r_out, 'r_out')
    radius_squared = (x[None, :] / r_out)**2 + (y[:, None] / r_out)**2
    pattern = np.exp2(-R**2 * radius_squared)
    return data * pattern


def _validate_range_inputs(data: np.ndarray, x: np.ndarray, y: np.ndarray,
                           sigma: RealScalar, v: np.ndarray | None
                           ) -> tuple[np.ndarray,
                                      np.ndarray,
                                      np.ndarray | None]:
    """Validate range-estimator inputs and return the coordinate arrays.

    Preserve array dtypes and leave all inputs unchanged. Reject unsuitable
    dimensions, noise values, coordinates, and image/cube velocity arguments.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError('data must be a NumPy array.')
    if (data.ndim not in (2, 3) or data.dtype.kind not in 'iuf'
            or any(length < 2 for length in data.shape)):
        raise ValueError('data must be a real 2D or 3D array with at least'
                         + ' two samples on every axis.')
    _require_positive_real_scalar(sigma, 'sigma')
    x, y = np.asarray(x), np.asarray(y)
    axes = [(x, data.shape[-1], 'x'), (y, data.shape[-2], 'y')]
    if data.ndim == 3:
        if v is None:
            raise ValueError('v is required for a cube.')
        v = np.asarray(v)
        axes.append((v, data.shape[0], 'v'))
    elif v is not None:
        raise ValueError('v must be omitted for a 2D image.')
    for coordinate, length, name in axes:
        if (coordinate.ndim != 1 or len(coordinate) != length
                or coordinate.dtype.kind not in 'iuf'
                or not np.all(np.isfinite(coordinate))):
            raise ValueError(f'{name} must be a finite real 1D coordinate array'
                             + ' matching its data axis.')
        steps = np.diff(coordinate)
        if steps[0] == 0 or not _is_uniform_spacing(steps, steps[0]):
            raise ValueError(f'{name} must have nonzero uniform spacing.')
    return x, y, v


def _range_beam_settings(beam: RealSequence | None, dx: RealScalar, dy: RealScalar
                         ) -> tuple[RealScalar, RealScalar]:
    """Return minimum spatial footprint in pixels and spatial padding.

    Use a Gaussian beam area and the larger FWHM when beam lengths are
    provided, otherwise one pixel. Internal numeric precision is retained.
    """
    minimum_pixels = 1
    padding = max(dx, dy)
    if beam is not None:
        sizes = np.asarray(beam)
        if sizes.ndim != 1 or sizes.size not in (2, 3):
            raise ValueError('beam must contain [major, minor]'
                             + ' or [major, minor, pa].')
        for size in sizes[:2]:
            _require_positive_real_scalar(size, 'beam FWHM')
        minimum_pixels = np.ceil(np.pi * sizes[0] * sizes[1]
                                 / (4 * np.log(2) * dx * dy))
        padding = max(sizes[:2])

    return minimum_pixels, padding


def _find_emission_bounds(data: np.ndarray, sigma: RealScalar,
                          minimum_pixels: RealScalar
                          ) -> tuple[slice, ...] | None:
    """Return union bounding slices for qualified emission, or None.

    Keep all face-connected 3-sigma components with a 5-sigma seed and the
    minimum projected spatial footprint. Cubes also require two adjacent
    channels. Input arrays are not modified.
    """
    structure = ndimage.generate_binary_structure(data.ndim, 1)
    mask = np.isfinite(data) & (data >= 3 * sigma)
    labels, _ = ndimage.label(mask, structure=structure)
    del mask
    lower = list(data.shape)
    upper = [0] * data.ndim
    found = False
    for component_id, region in enumerate(ndimage.find_objects(labels), start=1):
        if region is None:
            continue
        component = labels[region] == component_id
        if not np.any((data[region] >= 5 * sigma) & component):
            continue
        if data.ndim == 3:
            if region[0].stop - region[0].start < 2:
                continue
            spatial = np.any(component, axis=0)
        else:
            spatial = component
        if np.count_nonzero(spatial) < minimum_pixels:
            continue
        found = True
        for axis, extent in enumerate(region):
            lower[axis] = min(lower[axis], extent.start)
            upper[axis] = max(upper[axis], extent.stop)

    if not found:
        return None
    return tuple(slice(lo, hi) for lo, hi in zip(lower, upper))


def _estimate_ranges(data: np.ndarray, x: np.ndarray, y: np.ndarray,
                     sigma: RealScalar, *, v: np.ndarray | None = None,
                     beam: RealSequence | None = None
                     ) -> tuple[float | np.longdouble,
                                float | np.longdouble | None,
                                float | np.longdouble | None]:
    """Estimate spatial and velocity limits from positive emission.

    Args:
        data: Real numeric image (ny, nx) or cube (nchan, ny, nx), with
            approximately uniform noise, for example after _multiply_pb.
            Nonfinite values are excluded from detection.
        x: Finite, uniformly spaced 1D x offsets from the intended center.
        y: Finite, uniformly spaced 1D y offsets from the intended center.
            Both spatial axes require at least two samples and may descend.
        sigma: Finite positive scalar noise level in the data units.
        v: Finite, uniformly spaced channel coordinates, required for a
            cube and omitted for an image. At least two channels are needed.
        beam: Optional [major, minor] FWHM lengths, or [major, minor, pa].
            Lengths use the same units as x and y; pa is unused. Without a
            beam, the minimum footprint is one pixel and padding is one
            spatial pixel rather than one beam FWHM.

    Returns:
        tuple: (rmax, vmin, vmax). Computed scalars are Python floats,
        except np.longdouble results retain extended precision. An image
        returns None for both velocity limits. A cube returns numerical
        velocity limits in ascending order even for a descending v axis.
        Inputs and their array dtypes are unchanged.

    Raises:
        TypeError: data is not a NumPy array.
        ValueError: Data, axes, noise, beam, or the centered finite region
            are unsuitable.

    Notes:
        Face-connected components at or above 3*sigma survive only when they
        contain a 5*sigma seed and cover at least one Gaussian beam area
        in projected spatial pixels. Cube components must also span two
        adjacent channels. All qualifying components are retained.

        For each spatial axis, average the absolute values of the two
        emission endpoints; take the larger x/y average and add the major
        beam FWHM. Thus asymmetric emission uses mean absolute extent,
        not maximum distance, and its farthest edge may lie outside the
        symmetric field. Clip rmax to the centered finite footprint.
        Velocity limits include two extra channels on each side, clipped
        to the available channels, without skipping or binning.

        With no qualifying component, use the centered finite footprint
        and full velocity range. Spatial pixels with any nonfinite cube
        channel restrict that footprint, with a pixel margin as in the
        script generators. Thresholds assume a zero noise baseline; subtract
        a fitted noise mean from data before calling if appropriate.
        This helper does not fit noise or apply a PB.
    """
    x, y, v = _validate_range_inputs(data, x, y, sigma, v)
    dx, dy = abs(x[1] - x[0]), abs(y[1] - y[0])
    minimum_pixels, padding = _range_beam_settings(beam, dx, dy)

    footprint = AstroData(data=data, x=x, y=y, sigma=None)
    footprint.dx, footprint.dy = dx, dy
    full_radius = max((abs(x[0]) + abs(x[-1]) + dx) / 2,
                      (abs(y[0]) + abs(y[-1]) + dy) / 2)
    available_radius = _get_valid_radius(footprint, full_radius)
    bounds = _find_emission_bounds(data, sigma, minimum_pixels)
    rmax = available_radius
    if bounds is not None:
        rx = (abs(x[bounds[-1].start]) + abs(x[bounds[-1].stop - 1])) / 2
        ry = (abs(y[bounds[-2].start]) + abs(y[bounds[-2].stop - 1])) / 2
        rmax = min(max(rx, ry) + padding, available_radius)
    rmax = _normalize_float(rmax, force_float=True)
    if v is None:
        return rmax, None, None
    first = max(bounds[0].start - 2, 0) if bounds is not None else 0
    last = (min(bounds[0].stop + 1, len(v) - 1)
            if bounds is not None else len(v) - 1)
    return (rmax,
            _normalize_float(min(v[first], v[last]), force_float=True),
            _normalize_float(max(v[first], v[last]), force_float=True))


def _estimate_pb_ranges(image: AstroData
                        ) -> tuple[float | np.longdouble,
                                   float | np.longdouble | None,
                                   float | np.longdouble | None]:
    """Fit PB-corrected noise and estimate emission ranges for loaded data.

    image must contain one real 2D image or 3D cube already read by
    AstroFrame, with spatial coordinates, pixel sizes, and cube velocities.
    Fit hist-pbcor to the full image or the first and last cube channels.
    Infer r_out from the equivalent circular area of the finite noise
    footprint, averaging the two sampled channel areas for a cube.

    Multiply a temporary array by the approximate PB from the fitted R,
    subtract the fitted noise mean, and pass it to _estimate_ranges with
    the fitted noise standard deviation and available beam metadata.
    Missing beam lengths select the estimator's pixel-based defaults.
    The AstroData object, its arrays, and the input FITS file are unchanged.

    Return (rmax, vmin, vmax), with None velocity limits for an image.
    Scalar results are Python floats, except np.longdouble results retain
    extended precision. The selection rules and fallback are those of
    _estimate_ranges. Errors from noise fitting, PB weighting, and range
    estimation propagate from the underlying helpers.
    """
    if not isinstance(image.data, np.ndarray) or image.data.ndim not in (2, 3):
        raise ValueError('Expected a loaded 2D image or 3D cube.')
    if image.data.ndim == 2:
        noise_data = image.data
        pixels = np.count_nonzero(np.isfinite(noise_data))
    else:
        noise_data = image.data[[0, -1]]
        pixels = np.mean(np.count_nonzero(np.isfinite(noise_data), axis=(1, 2)))
    noise = Noise(data=noise_data, sigma='hist-pbcor')
    noise.gen_histogram()
    noise.fit_histogram()
    r_out = np.sqrt(pixels * abs(image.dx * image.dy) / np.pi)
    del noise_data
    weighted = _multiply_pb(image.data, image.x, image.y,
                            R=noise.popt[2], r_out=r_out)
    weighted -= noise.mean
    beam = image.beam if None not in image.beam[:2] else None
    return _estimate_ranges(weighted, image.x, image.y, noise.std,
                            v=image.v if image.data.ndim == 3 else None,
                            beam=beam)


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


def _get_valid_radius(image: AstroData, rmax: RealScalar
                      ) -> float | np.longdouble:
    """Reduce a half-width to exclude nonfinite pixels, with a pixel margin.

    image must contain one 2D image or 3D cube already read by AstroFrame,
    with spatial coordinates and pixel spacings. rmax uses those same
    spatial units. For cubes, a
    spatial pixel is invalid if any channel is nonfinite. Return the
    half-width as a Python float, preserving np.longdouble results as an
    extended-precision exception; emission-size selection is left to the caller.
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
    return _normalize_float(rmax, force_float=True)


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

    Generation reads the primary FITS image, fits its noise histogram, and
    estimates an emission-based fitting region. It does not execute the
    generated script or modify the input FITS file.
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
    rmax, _, _ = _estimate_pb_ranges(image)
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
    pixels, fits edge-channel noise, and estimates emission ranges. It does
    not execute the generated script or modify the input FITS file.

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
    rmax, vmin, vmax = _estimate_pb_ranges(cube)
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
