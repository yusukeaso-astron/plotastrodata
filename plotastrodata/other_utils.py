import warnings
from os import PathLike
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator as RGI

from plotastrodata._type_utils import BeamInput, RealScalar, _normalize_float
from plotastrodata._validation_utils import (
    _is_uniform_spacing,
    _require_positive_real_scalar,
)


def isdeg(s: Any) -> bool:
    """Whether the given string means degree.

    Args:
        s (str): The string to be checked.

    Returns:
        bool: Whether the given string means degree.
    """
    if isinstance(s, str):
        return s.strip() in ['deg', 'DEG', 'degree', 'DEGREE']
    else:
        return False


def nearest_index(arr: np.ndarray, x: float = 0) -> int:
    """Get the index of the input arrary that gives a value nearest to
    the given value x. np.searchsorted() does not work with a descending
    array.

    Args:
        arr (np.ndarray): Sorted array.
        x (float, optional): Value to approach. Defaults to 0.

    Returns:
        int: The index that gives a value nearest to x.
    """
    return int(np.argmin(np.abs(arr - x)))


def trim(data: np.ndarray | None = None, x: np.ndarray | None = None,
         y: np.ndarray | None = None, v: np.ndarray | None = None,
         xlim: list[float | None] | None = None,
         ylim: list[float | None] | None = None,
         vlim: list[float | None] | None = None,
         pv: bool = False
         ) -> tuple[np.ndarray | None, list[np.ndarray | None]]:
    """Trim 2D or 3D data by given coordinates and their limits.

    Singleton axes are squeezed before selecting spatial/spectral axes.
    Data that squeeze to a scalar are rejected; data=None trims grids only.

    Args:
        data (np.ndarray, optional): 2D or 3D array. Defaults to None.
        x (np.ndarray, optional): 1D array. Defaults to None.
        y (np.ndarray, optional): 1D array. Defaults to None.
        v (np.ndarray, optional): 1D array. Defaults to None.
        xlim (list, optional): [xmin, xmax]. Defaults to None.
        ylim (list, optional): [ymin, ymax]. Defaults to None.
        vlim (list, optional): [vmin, vmax]. Defaults to None.

    Returns:
        tuple: Trimmed (data, [x,y,v]). An axis is left untrimmed when
        its limit is None or contains None.
    """
    def get_bounds(arr: np.ndarray | None,
                   lim: list[float | None] | None
                   ) -> tuple[np.ndarray | None, int, int | None]:
        if arr is None or lim is None or None in lim:
            return arr, 0, None
        lo = nearest_index(arr, max(np.min(arr), np.min(lim)))
        hi = nearest_index(arr, min(np.max(arr), np.max(lim)))
        lo, hi = sorted((lo, hi))
        return arr[lo:hi + 1], lo, hi + 1

    xout, i0, i1 = get_bounds(x, xlim)
    yout, j0, j1 = get_bounds(y, ylim)
    vout, k0, k1 = get_bounds(v, vlim)

    if data is None:
        return None, [xout, yout, vout]

    d = np.squeeze(data)

    if d.ndim == 0:
        raise ValueError('data must be a 2D or 3D array, not a scalar.')

    if d.ndim == 2:
        if pv:
            j0, j1 = k0, k1
        dataout = d[j0:j1, i0:i1]
    else:
        d = np.moveaxis(d, [-3, -2, -1], [0, 1, 2])
        d = d[k0:k1, j0:j1, i0:i1]
        d = np.moveaxis(d, [0, 1, 2], [-3, -2, -1])
        dataout = d

    return dataout, [xout, yout, vout]


def to4dim(data: np.ndarray) -> np.ndarray:
    """Change a 2D, 3D, or 4D array to a 4D array.

    Args:
        data (np.ndarray): Input data. 2D, 3D, or 4D.

    Returns:
        np.ndarray: Output 4D array.
    """
    ndim = np.ndim(data)
    if ndim not in [2, 3, 4]:
        raise ValueError('data must be a 2D, 3D, or 4D array.')
    if ndim == 2:
        d = np.array([[data]])
    elif np.ndim(data) == 3:
        d = np.array([data])
    else:
        d = np.array(data)
    return d


def reform_grid(v: np.ndarray | None = None,
                k0: int | None = None, k1: int | None = None,
                vmin: float | None = None, vmax: float | None = None
                ) -> np.ndarray | None:
    """Extend or cut the given 1D array based on the given range.

    Args:
        v (np.ndarray | None, optional): Input 1D array. Defaults to
            None.
        k0 (int | None, optional): How many channels are added before
            v[0]; the minus sign means extension. k0 has the priority
            over vmin. Defaults to None.
        k1 (int | None, optional): How many channels are added after
            v[-1]; the plus sign means extension. k1 has the priority
            over vmax. Defaults to None.
        vmin (float | None, optional): New minimum velocity. Defaults to
            None.
        vmax (float | None, optional): New maximum velocity. Defaults to
            None.

    Returns:
        np.ndarray: Extended or cut 1D array.
    """
    if v is None or len(v) <= 1:
        return v

    dv = v[1] - v[0]
    if k0 is None and vmin is not None:
        k0 = int(round((vmin - v[0]) / dv))
    if k0 is not None and k0 != 0:
        if k0 < 0:
            vpre = v[0] + dv * np.arange(k0, 0)
            v = np.concatenate((vpre, v))
        else:
            v = v[k0:]
    if k1 is None and vmax is not None:
        k1 = int(round((vmax - v[-1]) / dv))
    if k1 is not None and k1 != 0:
        if k1 > 0:
            vpost = v[-1] + dv * np.arange(1, k1 + 1)
            v = np.concatenate((v, vpost))
        else:
            v = v[:len(v) + k1]
    return v


def reform_data(c: np.ndarray, v_in: np.ndarray | None,
                nv: int, v_org: np.ndarray | None = None,
                vskip: int = 1) -> np.ndarray:
    """Skip and fill channels with nan.

    Args:
        c (np.ndarray): The input 2D or 3D arrays.
        v_in (np.ndarray): The input velocity 1D array.
        nv (int): The number of channels with a label.
        v (np.ndarray, optional): The velocity 1D array, including the
            channels with and without a label. Defaults to None.
        vskip (int, optional): How many channels are skipped. Defaults
            to 1.

    Returns:
        np.ndarray: 3D arrays skipped and filled with nan.
    """
    if v_org is None:
        return c

    ndim = np.ndim(c)
    if ndim not in [2, 3]:
        raise ValueError('c must be a 2D or 3D array.')

    if ndim == 2:
        d = np.full((nv, *np.shape(c)), c)
    elif v_in is not None:
        if len(v_in) < 2 or len(v_org) < 2:
            raise ValueError(
                '3D reform_data() requires at least two v_in and v_org values.')
        dv_org = v_org[1] - v_org[0]
        dv_in = (v_in[1] - v_in[0]) * vskip
        k0 = nearest_index(v_org, v_in[0])
        k1 = nearest_index(v_org, v_in[-1])
        if np.abs(dv_in - dv_org) / dv_org < 0.01:
            d = c
        else:
            warnings.warn('Velocity resolution mismatch (>1%).'
                          + ' The cube needs to be regridded'
                          + ' outside plotastrodata.',
                          UserWarning)
            n_valid = k1 - k0
            d = [None] * n_valid
            for k in range(n_valid):
                k_tmp = nearest_index(v_in, v_org[k])
                diffvel = np.abs(v_in[k_tmp] - v_org[k])
                nearby = diffvel < dv_org * 0.5
                d[k] = c[k_tmp] if nearby else c[0] * np.nan
            d = np.array(d)
        if k0 > 0:
            prenan = np.full((k0, *np.shape(d)[1:]), np.nan)
            d = np.concatenate((prenan, d))
        d = d[::vskip]
    else:
        raise ValueError('3D reform_data() requires v_in when v_org is given.')
    shape = np.shape(d)
    shape = (len(v_org) - shape[0], shape[1], shape[2])
    postnan = np.full(shape, np.nan)
    d = np.concatenate((d, postnan))
    return d


def RGIxy(y: np.ndarray, x: np.ndarray, data: np.ndarray,
          yxnew: tuple[np.ndarray, np.ndarray] | list[np.ndarray]
          | np.ndarray | None = None,
          **kwargs: Any) -> RGI | list[RGI] | list[list[RGI]] | np.ndarray:
    """RGI for x and y at each channel.

    Default keyword values:
        RegularGridInterpolator: ``bounds_error=False``,
        ``fill_value=np.nan``, and ``method='linear'``. User-supplied
        keyword arguments override these values.

    Args:
        y (np.ndarray): 1D array. Second coordinate.
        x (np.ndarray): 1D array. First coordinate.
        data (np.ndarray): 2D, 3D, or 4D array.
        yxnew (tuple, list, or np.ndarray, optional): (ynew, xnew), where
            ynew and xnew are 1D or 2D arrays. Defaults to None.

    Returns:
        RGI or list or np.ndarray: Without evaluation coordinates, an
        interpolator or a (possibly nested) list of interpolators. Otherwise,
        the interpolated array.
    """
    if np.ndim(data) not in [2, 3, 4]:
        raise ValueError('data must be a 2D, 3D, or 4D array.')

    _kw = {'bounds_error': False, 'fill_value': np.nan,
           'method': 'linear'}
    _kw.update(kwargs)
    c4d = to4dim(data)
    c4d[np.isnan(c4d)] = 0
    f = [[RGI((y, x), c2d, **_kw)
          for c2d in c3d] for c3d in c4d]
    if yxnew is None:
        if len(f) == 1:
            f = f[0]
        if len(f) == 1:
            f = f[0]
        return f
    else:
        return np.squeeze([[f2d(tuple(yxnew)) for f2d in f3d] for f3d in f])


def RGIxyv(v: np.ndarray, y: np.ndarray, x: np.ndarray, data: np.ndarray,
           vyxnew: tuple[np.ndarray, np.ndarray, np.ndarray]
           | list[np.ndarray] | np.ndarray | None = None,
           **kwargs: Any) -> RGI | list[RGI] | np.ndarray:
    """RGI in the x-y-v space.

    Default keyword values:
        RegularGridInterpolator: ``bounds_error=False``,
        ``fill_value=np.nan``, and ``method='linear'``. User-supplied
        keyword arguments override these values.

    Args:
        v (np.ndarray): 1D array. Third coordinate.
        y (np.ndarray): 1D array. Second coordinate.
        x (np.ndarray): 1D array. First coordinate.
        data (np.ndarray): 3D or 4D array.
        vyxnew (tuple, list, or np.ndarray, optional): (vnew, ynew, xnew),
            where vnew, ynew, and xnew are 1D or 2D arrays. Defaults to None.

    Returns:
        RGI or list or np.ndarray: Without evaluation coordinates, an
        interpolator or a list of interpolators. Otherwise, the interpolated
        array.
    """
    if np.ndim(data) not in [3, 4]:
        raise ValueError('data must be a 3D or 4D array.')

    _kw = {'bounds_error': False, 'fill_value': np.nan,
           'method': 'linear'}
    _kw.update(kwargs)
    c4d = to4dim(data)
    c4d[np.isnan(c4d)] = 0
    f = [RGI((v, y, x), c3d, **_kw) for c3d in c4d]
    if vyxnew is None:
        if len(f) == 1:
            f = f[0]
        return f
    else:
        return np.squeeze([f3d(tuple(vyxnew)) for f3d in f])


def close_figure(fig: Figure,
                 savefig: dict | str | PathLike[str] | None = None,
                 show: bool = False, tight: bool = True) -> None:
    """Save, show, and close the figure.

    If ``savefig`` is provided, the figure is saved with Matplotlib
    ``Figure.savefig``. Existing files with the same name are
    overwritten by Matplotlib. After optional saving/showing, the figure
    is closed with ``plt.close(fig)``.

    Default keyword values:
        Figure.savefig: ``bbox_inches='tight'`` and
        ``transparent=True``. Values in ``savefig`` override these
        defaults.

    Args:
        fig (Figure): Matplotlib figure to save, show, and close.
        savefig (dict, str, or os.PathLike, optional): For ``Figure.savefig``.
            Existing files may be overwritten, and the figure is closed
            after saving/showing. Defaults to None.
        show (bool, optional): True means doing plt.show(). Defaults to
            False.
        tight (bool, optional): True means doing fig.tight_layout().
            Defaults to True.
    """
    savefig0 = {'bbox_inches': 'tight', 'transparent': True}
    if tight:
        fig.tight_layout()
    if savefig is not None:
        s = ({'fname': savefig} if isinstance(savefig, (str, PathLike))
             else savefig)
        savefig0.update(s)
        fig.savefig(**savefig0)
    if show:
        plt.show()
    plt.close(fig)


def _validate_range_inputs(data: np.ndarray, x: np.ndarray, y: np.ndarray,
                           v: np.ndarray | None
                           ) -> tuple[np.ndarray,
                                      np.ndarray,
                                      np.ndarray | None]:
    """Validate data and coordinates and return the coordinate arrays.

    Preserve array dtypes and leave all inputs unchanged. Reject unsuitable
    dimensions, coordinates, and image/cube velocity arguments.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError('data must be a NumPy array.')
    if (data.ndim not in (2, 3) or data.dtype.kind not in 'iuf'
            or any(length < 2 for length in data.shape)):
        raise ValueError('data must be a real 2D or 3D array with at least'
                         + ' two samples on every axis.')
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


def _get_minarea_margin(
        beam: BeamInput | tuple[RealScalar | None, RealScalar | None] | None,
        dx: RealScalar, dy: RealScalar) -> tuple[RealScalar, RealScalar]:
    """Return minimum spatial footprint in pixels and spatial margin.

    Use a Gaussian beam area and the larger FWHM when beam lengths are
    provided, otherwise one pixel. Either missing length selects pixel-based
    defaults. Reject malformed shapes and invalid supplied lengths when both
    lengths are present; PA is unused. Internal numeric precision is retained.
    """
    min_area_pixels = 1
    spatial_margin = max(dx, dy)
    if beam is not None:
        sizes = np.asarray(beam)
        if sizes.ndim != 1 or sizes.size not in (2, 3):
            raise ValueError('beam must contain [major, minor]'
                             + ' or [major, minor, pa].')
        if any(size is None for size in sizes[:2]):
            return min_area_pixels, spatial_margin
        for size in beam[:2]:
            _require_positive_real_scalar(size, 'beam FWHM')
        min_area_pixels = np.ceil(np.pi * sizes[0] * sizes[1]
                                  / (4 * np.log(2) * dx * dy))
        spatial_margin = max(sizes[:2])

    return min_area_pixels, spatial_margin


def _find_emission_bounds(data: np.ndarray, sigma: RealScalar,
                          min_area_pixels: RealScalar
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
        if np.count_nonzero(spatial) < min_area_pixels:
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
                     min_area_pixels: RealScalar, spatial_margin: RealScalar,
                     ) -> tuple[float | np.longdouble,
                                float | np.longdouble | None,
                                float | np.longdouble | None]:
    """Estimate spatial and velocity limits from positive emission.

    Inputs and their array dtypes are unchanged. Thresholds assume a zero
    noise baseline; subtract a fitted noise mean beforehand if appropriate.
    The caller must validate data and coordinates with _validate_range_inputs
    and calculate min_area_pixels and spatial_margin with _get_minarea_margin.
    This function validates sigma, but does not fit noise or apply a PB.

    Args:
        data (np.ndarray): Real image (ny, nx) or cube (nchan, ny, nx)
            with approximately uniform noise. Nonfinite values are excluded
            from detection. Each axis requires at least two samples.
        x (np.ndarray): Finite, uniformly spaced 1D x offsets from the
            intended center, matching the spatial data axis. May descend.
        y (np.ndarray): Finite, uniformly spaced 1D y offsets from the
            intended center, matching the spatial data axis. May descend.
        sigma (int or float or np.integer or np.floating): Finite positive
            scalar noise level in the data units. Booleans are rejected.
        v (np.ndarray, optional): Finite, uniformly spaced channel
            coordinates. Required for cubes and omitted for images.
            At least two channels are needed; may descend. Defaults to None.
        min_area_pixels (int or float or np.integer or np.floating): Minimum
            projected spatial footprint in pixels, calculated by the caller.
        spatial_margin (int or float or np.integer or np.floating): Spatial
            margin in the units of x and y, calculated by the caller.

    Returns:
        tuple: (rmax, vmin, vmax). Spatial limits use the units of x and y;
        velocity limits use those of v and are returned in ascending order.
        An image returns None for both velocity limits. Computed scalars
        are Python floats, except np.longdouble results retain extended
        precision.

    Raises:
        ValueError: sigma is unsuitable, or no sufficiently large centered
            finite region remains.
    """
    _require_positive_real_scalar(sigma, 'sigma')
    dx, dy = abs(x[1] - x[0]), abs(y[1] - y[0])

    full_radius = max((abs(x[0]) + abs(x[-1]) + dx) / 2,
                      (abs(y[0]) + abs(y[-1]) + dy) / 2)
    available_radius = _get_valid_radius(data, x, y, full_radius)
    bounds = _find_emission_bounds(data, sigma, min_area_pixels)
    rmax = available_radius
    if bounds is not None:
        rx = (abs(x[bounds[-1].start]) + abs(x[bounds[-1].stop - 1])) / 2
        ry = (abs(y[bounds[-2].start]) + abs(y[bounds[-2].stop - 1])) / 2
        rmax = min(max(rx, ry) + spatial_margin, available_radius)
    rmax = _normalize_float(rmax, force_float=True)
    if v is None:
        return rmax, None, None
    first = max(bounds[0].start - 2, 0) if bounds is not None else 0
    last = (min(bounds[0].stop + 1, len(v) - 1)
            if bounds is not None else len(v) - 1)
    return (rmax,
            _normalize_float(min(v[first], v[last]), force_float=True),
            _normalize_float(max(v[first], v[last]), force_float=True))


def _get_valid_radius(data: np.ndarray, x: np.ndarray, y: np.ndarray,
                      rmax: RealScalar
                      ) -> float | np.longdouble:
    """Reduce a half-width to exclude nonfinite pixels, with a pixel margin.

    data contains one 2D image or 3D cube with spatial coordinates.
    rmax uses the same spatial units. For cubes, a spatial pixel is invalid
    if any channel is nonfinite. Return the half-width as a Python float,
    preserving np.longdouble results as an
    extended-precision exception; emission-size selection is left to the caller.
    Raise ValueError if no sufficiently large centered finite region remains.
    """
    if data.ndim == 2:
        valid = np.isfinite(data)
    else:
        valid = np.ones(data.shape[-2:], dtype=bool)
        for channel in data:
            valid &= np.isfinite(channel)
    invalid_y, invalid_x = np.nonzero(~valid)
    if invalid_x.size:
        nearest_invalid = np.min(np.maximum(np.abs(x[invalid_x]),
                                            np.abs(y[invalid_y])))
        margin = max(abs(x[1] - x[0]), abs(y[1] - y[0]))
        rmax = min(rmax, nearest_invalid - margin)
        if not np.isfinite(rmax) or rmax <= margin:
            raise ValueError('No centered finite spatial region large enough.')
    return _normalize_float(rmax, force_float=True)
