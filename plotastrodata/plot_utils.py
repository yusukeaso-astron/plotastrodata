from os import PathLike
from pathlib import Path
from dataclasses import dataclass, fields, field
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.collections import QuadMesh
from matplotlib.patches import Ellipse, Rectangle
from pydantic import ConfigDict, Field, field_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass
from typing import (Any, Callable, Literal,
                    TypeVar, TypedDict, TYPE_CHECKING)
from types import ModuleType

if TYPE_CHECKING:
    from plotly.graph_objs import Layout

from plotastrodata._validation_utils import (_require_integer,
                                             _is_finite_real_scalar,
                                             _is_positive_real_scalar,
                                             _require_positive_real_scalar,
                                             _is_zero_dimensional_array,
                                             _is_uniform_spacing)
from plotastrodata._type_utils import (_normalize_float, _internal_field,
                                       RealScalar, RealSequence,
                                       PositionPair, Positions,
                                       ValidatedFloat, ValidatedBeams)
from plotastrodata.analysis_utils import AstroData, AstroFrame
from plotastrodata.coord_utils import (coord2xy, xy2coord,
                                       get_hmdm, get_min, get_sec, _getframe)
from plotastrodata.fitting_utils import gaussian1d
from plotastrodata.noise_utils import estimate_rms
from plotastrodata.other_utils import close_figure, reform_grid, reform_data


plt.ioff()  # force to turn off interactive mode

T = TypeVar('T')
Stretch = Literal['linear', 'log', 'asinh', 'power']
AxisScale = Literal['linear', 'log', 'symlog', 'asinh', 'logit']

FloatOrList = ValidatedFloat | list[ValidatedFloat]
# Optional both as a whole setting and for individual per-stretch entries.
OptionalFloatOrList = ValidatedFloat | list[ValidatedFloat | None] | None
BeamPositions = list[ValidatedFloat] | list[list[ValidatedFloat] | None] | None
IndexOrArray = int | np.integer | np.ndarray

OverlayValues = RealScalar | RealSequence
ChannelIndices = list[int | np.integer] | tuple[int | np.integer, ...] | np.ndarray


def _overlay_values(value: Any, count: int, name: str
                    ) -> list | tuple | np.ndarray:
    """Broadcast scalars; require one sequence entry per position.

    Values are passed through without scalar normalization or dtype changes.
    A 0D array broadcasts its contained value, preserving the scalar dtype;
    sequence entries are not recursively unwrapped or converted.
    Empty positions produce no overlays, including when parameters are omitted.
    """
    if count == 0:
        return []
    if value is None:
        raise ValueError(f'{name} is required when positions are supplied.')
    if np.isscalar(value):
        return [value] * count
    if _is_zero_dimensional_array(value):
        return [value[()]] * count
    if np.ndim(value) != 1 or len(value) != count:
        raise ValueError(f'{name} must be a scalar or contain {count} values.')
    return value


def _positive_noise(value: Any, name: str) -> None:
    if not _is_positive_real_scalar(value):
        raise ValueError(f'{name} must be a finite positive scalar noise level.')


def _image_edges(grid: np.ndarray, name: str
                 ) -> tuple[float | np.floating, float | np.floating]:
    """Get pixel edges using the first interval as the spacing reference.

    Intermediate scalars and input arrays retain their precision.
    """
    steps = np.diff(grid)
    if (not np.all(np.isfinite(grid)) or steps[0] == 0
            or not _is_uniform_spacing(steps, steps[0])):
        raise ValueError(f'RGB {name} coordinates must be finite'
                         + ' and uniformly spaced.')
    return grid[0] - steps[0] / 2, grid[-1] + steps[0] / 2


def _validate_colorbar(cbformat: str, cbticks: RealSequence | None,
                       cbticklabels: list[str] | tuple[str, ...]
                       | np.ndarray | None, cblocation: str
                       ) -> np.ndarray | None:
    """Validate enabled colorbar options and prepare explicit tick positions."""
    if cblocation not in ('left', 'right', 'top', 'bottom'):
        raise ValueError('cblocation must be left, right,'
                         + ' top, or bottom.')
    try:
        formatter = (mpl.ticker.StrMethodFormatter(cbformat)
                     if '{' in cbformat
                     else mpl.ticker.FormatStrFormatter(cbformat))
        formatter(1.)
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        raise ValueError('cbformat must be a numeric printf-style'
                         ' or {x:...} format.') from exc
    if cbticks is not None:
        cbticks = np.asarray(cbticks)
        if cbticks.ndim != 1 or not np.all(np.isfinite(cbticks)):
            raise ValueError('cbticks must be a finite 1D sequence.')
    if (cbticklabels is not None
        and (cbticks is None or np.ndim(cbticklabels) != 1
             or len(cbticklabels) != len(cbticks))):
        raise ValueError('cbticklabels requires one label'
                         + ' per explicit cbtick.')
    return cbticks


def set_rcparams(fontsize: float = 18, nancolor: str | tuple[float, ...] = 'w',
                 dpi: float = 256) -> None:
    """Nice rcParams for figures.

    Args:
        fontsize (int, optional): plt.rcParams['font.size']. Defaults to
            18.
        nancolor (str, optional): plt.rcParams['axes.facecolor'].
            Defaults to 'w'.
        dpi (int, optional): plt.rcParams['savefig.dpi']. Defaults to
            256.
    """
    # plt.rcParams['font.family'] = 'arial'
    plt.rcParams['axes.facecolor'] = nancolor
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams['legend.fontsize'] = 15
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.direction'] = 'inout'
    plt.rcParams['ytick.direction'] = 'inout'
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['xtick.major.size'] = 10
    plt.rcParams['ytick.major.size'] = 10
    plt.rcParams['xtick.minor.size'] = 6
    plt.rcParams['ytick.minor.size'] = 6
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['xtick.minor.width'] = 1.5
    plt.rcParams['ytick.minor.width'] = 1.5


def _log_limits(vmin: float, vmax: float) -> None:
    """Validate a finite, positive, increasing logarithmic range."""
    if not (_is_finite_real_scalar(vmin) and _is_finite_real_scalar(vmax)
            and 0 < vmin < vmax):
        raise ValueError('Logarithmic limits must be finite and satisfy'
                         + ' 0 < vmin < vmax.')


def logticks(ticks: RealSequence, lim: RealSequence
             ) -> tuple[np.ndarray, list[str]]:
    """Make nice ticks for a log axis.

    Args:
        ticks (list): List of ticks.
        lim (list): [min, max].

    Returns:
        tuple: Numeric tick ndarray and a list of string labels.
    """
    if np.shape(lim) != (2,):
        raise ValueError('lim must contain exactly two limits.')
    _log_limits(*lim)
    order = int(np.floor((np.log10(lim[0]))))
    a = (lim[0] // 10**order + 1) * 10**order
    a = np.round(a, max(-order, 0))
    order = int(np.floor((np.log10(lim[1]))))
    b = (lim[1] // 10**order) * 10**order
    b = np.round(b, max(-order, 0))
    newticks = np.sort(np.unique(np.r_[a, ticks, b]))
    newlabels = [str(t if t < 1 else int(t)) for t in newticks]
    return newticks, newlabels


def logcbticks(vmin: float = 1e-3, vmax: float = 1e3
               ) -> tuple[np.ndarray, np.ndarray]:
    """Make nice ticks for a log color bar.

    Args:
        vmin (float, optional): Minimum value. Defaults to 1e-3.
        vmax (float, optional): Maximum value. Defaults to 1e3.

    Returns:
        tuple: Numeric tick ndarray and string-label ndarray.
    """
    _log_limits(vmin, vmax)
    i0 = int(np.floor(np.log10(vmin)))
    i1 = int(np.ceil(np.log10(vmax)))
    ticks = np.outer(np.logspace(i0, i1, i1 - i0 + 1), np.arange(1, 10))
    ticklabels = []
    for i in range(i0, i1 + 1):
        ii = np.abs(min(i, 0))
        ii = f'{ii:d}'
        for j in range(1, 10):
            jj = j * 10**i
            if j in [1, 2, 5]:
                s = f'{jj:.{ii}f}'
            else:
                s = ''
            ticklabels.append(s)
    ticks = np.ravel(ticks)
    ticklabels = np.ravel(ticklabels)
    cond = (vmin <= ticks) * (ticks <= vmax)
    return ticks[cond], ticklabels[cond]


def get_figsize(xmin: float, xmax: float, ymin: float, ymax: float,
                figsize: RealSequence | None = None,
                ncols: int | np.integer = 1, nrows: int | np.integer = 1,
                nchan: int | np.integer = 1
                ) -> tuple[float | np.longdouble,
                           float | np.longdouble] | RealSequence:
    """Get a nice figsize (tuple) with the given x and y ranges.

    ncols, nrows, and nchan must be positive Python/NumPy integers, even
    when figsize is supplied. Booleans, floats, and arrays are rejected.

    Args:
        xmin (float): The figsize is based on the aspect ratio of (ymax
            - ymin) / (xmax - xmin).
        xmax (float): The figsize is based on the aspect ratio of (ymax
            - ymin) / (xmax - xmin).
        ymin (float): The figsize is based on the aspect ratio of (ymax
            - ymin) / (xmax - xmin).
        ymax (float): The figsize is based on the aspect ratio of (ymax
            - ymin) / (xmax - xmin).
        figsize (tuple | None, optional): If this is not None, this will
            be the output as is. Defaults to None.
        ncols (int or np.integer, optional): The number of columns for the channel
            map. Defaults to 1.
        nrows (int or np.integer, optional): The number of rows for the channel map.
            Defaults to 1.
        nchan (int or np.integer, optional): The number of total channels for the
            channel map. Defaults to 1.

    Returns:
        tuple or list or ndarray: Supplied figsize is returned unchanged.
        Computed sizes contain Python floats or extended-precision NumPy floats.
    """
    for name, value in (('nrows', nrows), ('ncols', ncols), ('nchan', nchan)):
        _require_integer(value, name)
    if figsize is not None:
        return figsize
    if (not all(_is_finite_real_scalar(a) for a in (xmin, xmax, ymin, ymax))
            or xmin == xmax or ymin == ymax):
        raise ValueError('Automatic figure sizing requires finite limits'
                         + ' and nonzero x and y spans.')

    sqrt_a = (ymax - ymin) / (xmax - xmin)
    sqrt_a = np.sqrt(np.abs(sqrt_a))
    if nchan == 1:
        figsize = (7 / sqrt_a, 5 * sqrt_a)
    else:
        figsize = (ncols * 2 / sqrt_a, max(nrows*2, 3) * sqrt_a)
    return tuple(_normalize_float(a) for a in figsize)


def _get_gridwidth(mode: Literal['ra', 'dec'], rmax: float, cos_dec: float
                   ) -> tuple[float | np.floating, int]:
    if mode not in ('ra', 'dec'):
        raise ValueError("mode must be 'ra' or 'dec'.")
    _require_positive_real_scalar(rmax, 'rmax')
    if mode == 'ra' and not _is_positive_real_scalar(cos_dec):
        raise ValueError('cos_dec must be finite and positive for R.A.')
    # Length in the units of s for R.A. and " for Dec., respectively.
    length = 2 * rmax / (15 * cos_dec if mode == 'ra' else 1)
    p = np.floor(np.log10(length))
    b = length / 10**p
    if b <= 2:
        base, order = 5, p - 1
    elif b <= 4:
        base, order = 1, p
    else:
        base, order = 2, p
    return base * 10**order, int(order)


def _get_v(p: Any, v: RealSequence | None = None,
           restfreq: float | None = None,
           vskip: int = 1) -> np.ndarray:
    _require_integer(vskip, 'vskip')
    if p.fitsimage is not None and v is None:
        p.read(d := AstroData(fitsimage=p.fitsimage,
                              restfreq=restfreq, sigma=None))
        v = d.v
    if v is None:
        v = np.array([0])
    v = np.asarray(v)
    if v.ndim != 1 or v.size == 0:
        raise ValueError('v must be a nonempty 1D sequence.')
    if len(v) > 1:
        lower = p.vmin if getattr(p, '_explicit_vmin', True) else np.min(v)
        upper = p.vmax if getattr(p, '_explicit_vmax', True) else np.max(v)
        start, stop = (lower, upper) if v[-1] > v[0] else (upper, lower)
        v = reform_grid(v=v, vmin=start, vmax=stop)
        v = v[::vskip]
    if v.size == 0:
        raise ValueError('The velocity limits select no channels.')
    return v


def _get_nij2ch(nrows: int | np.integer = 1, ncols: int | np.integer = 1
                ) -> Callable[[IndexOrArray, IndexOrArray, IndexOrArray], int | np.ndarray]:
    _require_integer(nrows, 'nrows')
    _require_integer(ncols, 'ncols')

    def nij2ch(n: IndexOrArray, i: IndexOrArray, j: IndexOrArray) -> int | np.ndarray:
        result = n*nrows*ncols + i*ncols + j
        if any(isinstance(a, np.ndarray) for a in (n, i, j)):
            return np.asarray(result)
        return int(result)
    return nij2ch


def _get_ch2nij(nrows: int | np.integer = 1, ncols: int | np.integer = 1
                ) -> Callable[[IndexOrArray], tuple[int | np.ndarray,
                                                    int | np.ndarray,
                                                    int | np.ndarray]]:
    _require_integer(nrows, 'nrows')
    _require_integer(ncols, 'ncols')

    def ch2nij(ch: IndexOrArray) -> tuple[int | np.ndarray,
                                          int | np.ndarray,
                                          int | np.ndarray]:
        n = ch // (nrows*ncols)
        i = (ch - n*nrows*ncols) // ncols
        j = ch % ncols
        if isinstance(ch, np.ndarray):
            return np.asarray(n), np.asarray(i), np.asarray(j)
        return int(n), int(i), int(j)
    return ch2nij


def _get_vskipfill(nv: int, v_org: np.ndarray, vskip: int,
                   channelnumber: int | np.integer | None
                   ) -> Callable[[np.ndarray, np.ndarray | None], np.ndarray]:
    """Return a 3D channel sampler, with shape (1, ny, nx) for a selection.

    nv and vskip must be positive integers. Selection accepts Python and
    NumPy integers, including negative indices, with normal bounds checking.
    """
    _require_integer(nv, 'nv')
    _require_integer(vskip, 'vskip')
    if channelnumber is not None and (
            isinstance(channelnumber, (bool, np.bool_))
            or not isinstance(channelnumber, (int, np.integer))):
        raise ValueError('channelnumber must be an integer or None.')

    def vskipfill(c: np.ndarray, v_in: np.ndarray | None) -> np.ndarray:
        c = reform_data(c=c, v_in=v_in, nv=nv, v_org=v_org, vskip=vskip)
        if channelnumber is not None:
            c = c[channelnumber][np.newaxis]
        return c
    return vskipfill


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class Stretcher():
    """Arguments and methods related to the stretch in
    PlotAstroData.add_color() and add_rgb().

    Scalar settings broadcast across stretches; lists must match their count.
    Ordinary numeric settings become Python floats; extended precision is
    preserved. Configured bounds remain in linear units.

    Args:
        stretch (str, optional): 'log', 'asinh', 'power', or
            'linear'. Other names are rejected. 'log' means the
            mapped data are logarithmic. 'asinh' means the mapped
            data are arc sin hyperbolic. 'power' means the mapped
            data are power-law (see also stretchpower). Defaults to
            'linear'.
        stretchscale (float or list, optional): The output is
            asinh(data / stretchscale). Defaults to None.
        stretchpower (float or list, optional): The output is
            data**stretchpower / stretchpower. 1 means the linear
            scale, while 0 means the logarithmic scale. Defaults to
            0.5.
        vmin (float or list, optional): The minimum value for
            Axes.pcolormesh() of matplotlib. Defaults to None.
        vmax (float or list, optional): The maximum value for
            Axes.pcolormesh() of matplotlib. Defaults to None.
        sigma (float, list, or None, optional): Noise level. None
            means no noise level is used. Defaults to 0.
    """
    stretch: Stretch | list[Stretch] = 'linear'
    stretchscale: OptionalFloatOrList = None
    stretchpower: FloatOrList = 0.5
    vmin: OptionalFloatOrList = None
    vmax: OptionalFloatOrList = None
    sigma: OptionalFloatOrList = 0

    n: int = field(init=False, default=1)

    def __post_init__(self) -> None:
        self.n = 1 if isinstance(self.stretch, str) else len(self.stretch)
        if self.n == 0:
            raise ValueError('stretch must not be empty.')
        for name in ('stretch', 'stretchscale', 'stretchpower',
                     'vmin', 'vmax', 'sigma'):
            value = getattr(self, name)
            if isinstance(value, list):
                if len(value) != self.n:
                    raise ValueError(f'{name} must contain {self.n} values.')
                if self.n == 1:
                    setattr(self, name, value[0])
        # Resolve defaults without changing scalar parameters into arrays.
        scales, minima = [], []
        for i in range(self.n):
            st, scale, sigma = (self._parameter(name, i) for name in
                                ('stretch', 'stretchscale', 'sigma'))
            scale = sigma if scale is None else scale
            if st == 'asinh' and not _is_positive_real_scalar(scale):
                raise ValueError('asinh requires a finite positive'
                                 + ' stretchscale or sigma.')
            minimum = self._parameter('vmin', i)
            if minimum is None and st in ('log', 'power'):
                minimum = sigma
            scales.append(scale)
            minima.append(minimum)
        self.stretchscale = scales[0] if self.n == 1 else scales
        self.vmin = minima[0] if self.n == 1 else minima

    def _parameter(self, name: str, i: int) -> Any:
        value = getattr(self, name)
        return value[i] if isinstance(value, list) else value

    def do(self, x: float | np.floating | RealSequence, i: int = 0
           ) -> int | float | np.longdouble | np.ndarray:
        """Stretch values; arrays (including 0D) remain arrays.

        Scalar inputs return Python scalars, preserving extended precision.
        Lists and tuples return arrays. i selects a configured stretch.
        """
        st = self._parameter('stretch', i)
        t = np.array(x)
        if st == 'log':
            t = np.log10(t)
        elif st == 'asinh':
            t = np.arcsinh(t / self._parameter('stretchscale', i))
        elif st == 'power':
            power = self._parameter('stretchpower', i)
            power = 1e-6 if power == 0 else power
            t = t**power / power
        result = _normalize_float(t, array_output=not np.isscalar(x))
        if isinstance(result, (np.integer, np.bool_)):
            return result.item()
        return result

    def undo(self, x: float | np.floating | RealSequence, i: int = 0
             ) -> int | float | np.longdouble | np.ndarray:
        """Invert do(), preserving arrays and extended-precision scalars."""
        st = self._parameter('stretch', i)
        t = np.array(x)
        if st == 'log':
            t = 10.**t
        elif st == 'asinh':
            t = np.sinh(t) * self._parameter('stretchscale', i)
        elif st == 'power':
            power = self._parameter('stretchpower', i)
            power = 1e-6 if power == 0 else power
            t = (t * power)**(1 / power)
        result = _normalize_float(t, array_output=not np.isscalar(x))
        if isinstance(result, (np.integer, np.bool_)):
            return result.item()
        return result

    def set_minmax(self, data: np.ndarray | list[np.ndarray]
                   ) -> tuple[np.ndarray | list[np.ndarray],
                              float | np.longdouble | list[float | np.longdouble],
                              float | np.longdouble | list[float | np.longdouble]]:
        """Return stretched data and bounds without changing configured bounds.

        One stretch returns an array and scalar bounds. Multiple stretches
        return lists of arrays and bounds. Ordinary computed scalar bounds
        become Python floats; extended precision is preserved. Input arrays
        and input lists are not modified.
        """
        datasets = [data] if self.n == 1 else data
        if len(datasets) != self.n:
            raise ValueError('Provide one dataset per stretch.')
        output, minima, maxima = [], [], []
        for i, c in enumerate(datasets):
            lo, hi = self._parameter('vmin', i), self._parameter('vmax', i)
            if lo is not None and hi is not None and lo > hi:
                raise ValueError('vmin must not exceed vmax.')
            c = np.asarray(c)
            clipped = c if lo is None and hi is None else c.clip(lo, hi)
            stretched = self.do(clipped, i)
            output.append(stretched)
            minimum = (np.asarray(np.nanmin(stretched)).item()
                       if lo is None else self.do(lo, i))
            maximum = (np.asarray(np.nanmax(stretched)).item()
                       if hi is None else self.do(hi, i))
            minima.append(minimum if isinstance(minimum, np.longdouble)
                          else float(minimum))
            maxima.append(maximum if isinstance(maximum, np.longdouble)
                          else float(maximum))
        if self.n == 1:
            return output[0], minima[0], maxima[0]
        return output, minima, maxima


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class Beam():
    """Arguments for PlotAstroData.add_beam().

    Ordinary numeric settings become Python floats; extended precision
    is preserved.

    Args:
        show_beam (bool, optional): Defaults to True.
        beam (list or np.ndarray, optional): One
            ``[bmaj, bmin, bpa]`` triple or a list of triples.
            ``bmaj`` and ``bmin`` must be positive; ``bpa`` is in
            degrees. ``[None, None, None]`` means no beam is
            available. Partial triples are not supported. Defaults
            to ``[None, None, None]``.
        beamcolor (str or list of str, optional): Matplotlib color.
            One color is used for every beam; when a list is given,
            provide one color per beam. Defaults to ``'gray'``.
        beampos (list or list of list, optional): One relative
            ``[x, y]`` position is used for every beam; when a list
            of positions is given, provide one position per beam.
            Each coordinate must be from 0 (left or bottom) to 1
            (right or top). None selects the automatic position.
            Defaults to None.
        beam_kwargs (dict, optional): Additional Matplotlib patch
            arguments. Defaults to {}.
    """
    show_beam: bool = True
    beam: ValidatedBeams = Field(default_factory=lambda: [None] * 3)
    beamcolor: str | list[str] = 'gray'
    beampos: BeamPositions = None
    beam_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator('beam', mode='before')
    @classmethod
    def _convert_beam_arrays(cls, value: Any) -> Any:
        """Convert documented NumPy beam inputs to the canonical list
        form.

        A single NumPy triple becomes a list, and NumPy triples inside a
        list of beams are converted individually. This keeps subsequent
        validation and plotting code independent of the input container.
        """
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, list):
            return [item.tolist() if isinstance(item, np.ndarray) else item
                    for item in value]
        return value

    @field_validator('beam')
    @classmethod
    def _validate_beam(cls, value: ValidatedBeams) -> ValidatedBeams:
        beams = value if value and isinstance(value[0], list) else [value]
        for beam in beams:
            if len(beam) != 3:
                raise ValueError('Each beam must contain [bmaj, bmin, bpa].')
            if all(component is None for component in beam):
                continue
            if any(component is None for component in beam):
                raise ValueError('A beam must be complete or all None.')
            if not all(np.isfinite(component) for component in beam):
                raise ValueError('Beam values must be finite.')
            if beam[0] <= 0 or beam[1] <= 0:
                raise ValueError('bmaj and bmin must be positive.')
        return value

    @field_validator('beampos')
    @classmethod
    def _validate_beampos(cls, value: BeamPositions) -> BeamPositions:
        if value is None:
            return value
        positions = [value] if cls._is_position(value) else value
        for position in positions:
            if position is None:
                continue
            if len(position) != 2:
                raise ValueError('Each beam position must contain [x, y].')
            if not all(0 <= coordinate <= 1 for coordinate in position):
                raise ValueError('Beam positions must be between 0 and 1.')
        return value

    @staticmethod
    def _is_position(value: Any) -> bool:
        return (isinstance(value, list) and len(value) == 2
                and all(isinstance(coordinate, (float, np.floating))
                        for coordinate in value))

    def validate_display(self) -> None:
        """Validate color and position lists against the number of
        beams.
        """
        multi = self.beam and isinstance(self.beam[0], list)
        beams = self.beam if multi else [self.beam]
        nbeams = len(beams)
        if isinstance(self.beamcolor, list) and len(self.beamcolor) != nbeams:
            raise ValueError('Provide one beamcolor for each beam.')
        if (self.beampos is not None
                and not self._is_position(self.beampos)
                and len(self.beampos) != nbeams):
            raise ValueError('Provide one beampos for each beam.')

    def todict(self) -> dict[str, Any]:
        """Return beam display settings as a dictionary.

        Returns:
            dict: Keyword arguments for PlotAstroData.add_beam().
        """
        tmp = {'show_beam': self.show_beam,
               'beam': self.beam,
               'beamcolor': self.beamcolor,
               'beampos': self.beampos}
        tmp.update(self.beam_kwargs)
        return tmp


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class PlotAxes2D():
    """Use Axes.set_* to adjust x and y axes.

    Ordinary numeric settings become Python floats; extended precision
        is preserved.

    Args:
        samexy (bool, optional): True supports same ticks between x and
            y. Defaults to True.
        loglog (float, optional): A positive value plots on a log-log
            plane, with xim=(xmax / loglog, xmax) and similarly for
            ylim. Defaults to None.
        xscale (str, optional): ``'log'`` labels decade ticks and ticks
            near the limits; other intermediate ticks are minor and
            unlabeled. Defaults to ``'linear'``.
        yscale (str, optional): ``'log'`` labels decade ticks and ticks
            near the limits; other intermediate ticks are minor and
            unlabeled. Defaults to ``'linear'``.
        xlim (list, optional): Defaults to None.
        ylim (list, optional): Defaults to None.
        xlabel (str, optional): Defaults to None.
        ylabel (str, optional): Defaults to None.
        xticks (list, optional): Explicit major ticks, overriding the
            default locator. Defaults to None.
        yticks (list, optional): Explicit major ticks, overriding the
            default locator. Defaults to None.
        xticklabels (list, optional): Defaults to None.
        yticklabels (list, optional): Defaults to None.
        xticksminor (list, int, or np.integer, optional): Explicit minor-tick
            positions, or a positive integer number of subdivisions per major
            interval. Counts reject booleans, floats, and arrays. Generation
            requires at least two major ticks. Defaults to None.
        yticksminor (list, int, or np.integer, optional): Same rule for y-axis
            minor ticks. Defaults to None.
        grid (dict or bool, optional): A bool toggles the grid;
            a dict supplies options. Defaults to None.
        aspect (dict, float, or str, optional): Also accepts 'equal' and
            'auto'. Defaults to None.
    """
    samexy: bool = True
    loglog: ValidatedFloat | None = Field(default=None, gt=0)
    xscale: AxisScale = 'linear'
    yscale: AxisScale = 'linear'
    xlim: list[ValidatedFloat] | None = None
    ylim: list[ValidatedFloat] | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    xticks: list[ValidatedFloat] | np.ndarray | None = None
    yticks: list[ValidatedFloat] | np.ndarray | None = None
    xticklabels: list | None = None
    yticklabels: list | None = None
    xticksminor: list | int | None = None
    yticksminor: list | int | None = None
    grid: dict | bool | None = None
    aspect: dict | ValidatedFloat | Literal['equal', 'auto'] | None = None
    ax: Axes | None = _internal_field()

    @field_validator('xlim', 'ylim')
    @classmethod
    def _validate_limits(cls, value: list[ValidatedFloat] | None
                         ) -> list[ValidatedFloat] | None:
        if (value is not None
                and (len(value) != 2
                     or not all(np.isfinite(v) for v in value))):
            raise ValueError('Axis limits must contain two finite values.')
        return value

    @field_validator('xticksminor', 'yticksminor', mode='before')
    @classmethod
    def _validate_minor(cls, value: Any) -> Any:
        if np.isscalar(value) or _is_zero_dimensional_array(value):
            _require_integer(value, 'minor tick subdivisions')
            return int(value)
        return value

    def _set_scale(self) -> None:
        ax = self.ax
        if self.loglog is not None:
            self.xscale = self.yscale = 'log'
            self.samexy = True
            for axis in ['x', 'y']:
                attr = f'{axis}lim'
                lim = getattr(self, attr)
                if lim is not None:
                    lim[0] = lim[1] / self.loglog
                setattr(self, attr, lim)
        ax.set_xscale(self.xscale)
        ax.set_yscale(self.yscale)

    def _init_ticks(self, axis: str) -> None:
        ax = self.ax
        ticks_attr = f'{axis}ticks'
        scale = getattr(self, f'{axis}scale')
        ticks = getattr(self, ticks_attr)
        if ticks is None:
            if scale == 'log':
                axis_obj = getattr(ax, f'{axis}axis')
                lim = getattr(self, f'{axis}lim')
                if lim is None:
                    lim = axis_obj.get_view_interval()
                lim = np.sort(lim)
                major_ticks = mpl.ticker.LogLocator(base=10).tick_values(*lim)
                major_ticks, major_labels = logticks(major_ticks, lim)
                axis_obj.set_ticks(major_ticks)
                axis_obj.set_ticklabels(major_labels)
                ll = mpl.ticker.LogLocator(base=10, subs=np.arange(2.0, 10.0))
                axis_obj.set_minor_locator(ll)
                axis_obj.set_minor_formatter(mpl.ticker.NullFormatter())
            else:
                ticks = getattr(ax, f'get_{axis}ticks')()
                setattr(self, ticks_attr, ticks)

    def _make_ticks(self, ticks: RealSequence, ticksminor: int
                    ) -> np.ndarray:
        if len(ticks) < 2:
            raise ValueError('Minor subdivisions require at least'
                             + ' two major ticks.')
        dt = ticks[1] - ticks[0]
        t = np.r_[ticks[0] - dt, ticks, ticks[-1] + dt]
        num = ticksminor * (len(t) - 1) + 1
        return np.linspace(t[0], t[-1], num)

    def _set_ticks(self, axis: str) -> None:
        ax = self.ax
        attr = f'{axis}ticks'
        ticks = getattr(self, attr)
        if ticks is not None:
            getattr(ax, f'set_{attr}')(ticks)
        ticksminor = getattr(self, f'{attr}minor')
        if ticksminor is not None:
            if isinstance(ticksminor, int):
                if ticks is None:
                    ticks = getattr(ax, f'get_{axis}ticks')()
                major_ticks = (ticks)
                ticksminor = self._make_ticks(major_ticks, ticksminor)
            getattr(ax, f'set_{attr}')(ticksminor, minor=True)

    def _apply_if_not_none(self, axis: str, attr: str) -> None:
        ax = self.ax
        method = getattr(ax, f'set_{axis}{attr}')
        value = getattr(self, f'{axis}{attr}')
        if value is not None:
            if attr == 'lim':
                method(*value)
            else:
                method(value)

    def set_xyaxes(self, ax: Axes) -> None:
        """Apply stored x- and y-axis settings to a Matplotlib axes
        object.

        Args:
            ax (object): Matplotlib axes object.
        """
        self.ax = ax
        self._set_scale()
        for axis in ['x', 'y']:
            # Applying 'lim' is required before and after the ticks.
            self._apply_if_not_none(axis, 'lim')
        if self.samexy:
            ax.set_xticks(ax.get_yticks())
            ax.set_yticks(ax.get_xticks())
            ax.set_aspect(1)
        for axis in ['x', 'y']:
            self._init_ticks(axis)
            self._set_ticks(axis)
            for attr in ['ticklabels', 'label', 'lim']:
                self._apply_if_not_none(axis, attr)
        if self.grid is not None:
            if isinstance(self.grid, dict):
                ax.grid(**self.grid)
            else:
                ax.grid(self.grid)
        if self.aspect is not None:
            if isinstance(self.aspect, dict):
                ax.set_aspect(**self.aspect)
            else:
                ax.set_aspect(self.aspect)


def kwargs2instance(cls: type[T], kw: dict[str, Any]) -> T:
    """Construct a dataclass using its init fields and remove consumed kwargs.

    Retain fitsimage/center for AstroFrame, vmin/vmax for Stretcher, and beam
    for Beam. Other unrecognized keywords remain in the original dictionary.

    Args:
        cls (class): Class to make the instance.
        kw (dict): Parameters to make the instance.

    Returns:
        instance: an instance of cls made from the parameters in kwargs.
    """
    exkeys = set()
    if cls == AstroFrame:
        exkeys = {'fitsimage', 'center'}
    elif cls == Stretcher:
        exkeys = {'vmin', 'vmax'}
    elif cls == Beam:
        exkeys = {'beam'}
    keys = {f.name for f in fields(cls) if f.init}
    tmp = {k: kw[k] for k in keys if k in kw}
    for k in keys - exkeys:
        kw.pop(k, None)
    return cls(**tmp)


def _normalize_rgb_channel(data: np.ndarray, st: Stretcher,
                           i: int) -> np.ndarray:
    """Stretch one RGB channel, preserving invalid pixels and array precision."""
    lo, hi = st._parameter('vmin', i), st._parameter('vmax', i)
    if lo is not None and hi is not None and lo > hi:
        raise ValueError('vmin must not exceed vmax.')
    clipped = data if lo is None and hi is None else data.clip(lo, hi)
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        stretched = st.do(clipped, i)
    valid = np.isfinite(data) & np.isfinite(stretched)
    normalized = np.full(stretched.shape, np.nan,
                         dtype=np.result_type(stretched.dtype,
                                              np.float32))
    if np.any(valid):
        low = np.min(stretched[valid]) if lo is None else st.do(lo, i)
        high = np.max(stretched[valid]) if hi is None else st.do(hi, i)
        if not np.isfinite(low) or not np.isfinite(high):
            raise ValueError('RGB stretch bounds must be finite'
                             + ' after transformation.')
        normalized[valid] = ((stretched[valid] - low) / (high - low)
                             if high > low else 0)
    return normalized


_AxisTicks = tuple[np.ndarray, np.ndarray, list[str]]


def _radec_ticks(center: str, frame_name: str, dec_center: RealScalar,
                 rmax: RealScalar, nticksminor: int
                 ) -> tuple[_AxisTicks, _AxisTicks]:
    """Build original-frame R.A./Dec. tick positions and labels.

    Round the tick grids around the unchanged center before any display swap.
    """
    def sky_at(xy: list | np.ndarray) -> str | list | np.ndarray:
        # Keep coordinate parsing helpers free of the frame prefix while
        # performing offsets in the supplied frame, not implicit ICRS.
        return _getframe(xy2coord(xy, center, frameorg=frame_name))[0]

    # Round tick locations, never the coordinate reference itself.
    on_min_scale = rmax >= 60.0

    def get_tickvalues(ticks: np.ndarray, mode: str, no_sec: bool
                       ) -> np.ndarray:
        xy = [np.zeros_like(ticks), ticks / 3600.]
        if mode == 'ra':
            xy.reverse()
        tickvalues = sky_at(xy)
        getter = get_min if no_sec else get_sec
        tickvalues = [getter(t, mode) for t in tickvalues]  # str
        tickvalues = np.array(tickvalues, dtype=float)
        # 7-digit precision for practical use.
        tickvalues = np.round(tickvalues, 7)
        return tickvalues

    units = {'ra': {'h': r'$^\mathrm{h}$',
                    'm': r'$^\mathrm{m}$',
                    's': r'.$\hspace{-0.4}^\mathrm{s}$'},
             'dec': {'d': r'$^{\circ}$',
                     'm': r'$^{\prime}$',
                     's': r'.$\hspace{-0.4}^{\prime\prime}$'}}
    # Zero declination needs a nonzero direction for its tick grid.
    sign_dec = -1 if dec_center < 0 else 1
    cos_dec = np.cos(np.radians(dec_center))
    intgrid = np.array([-3, -2, -1, 0, 1, 2, 3])

    def makegrid(mode: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        second = float(get_sec(center, mode))
        no_sec = on_min_scale and (mode == 'dec')
        # gridwidth is a float like 2 x 10^order (arcsec).
        gridwidth, order = _get_gridwidth(mode, rmax, cos_dec)
        if no_sec:
            # Minute-only labels require whole-minute spacing; a 50-arcsec
            # step, for example, would truncate distinct ticks to one label.
            gridwidth = max(60., np.ceil(gridwidth / 60.) * 60.)
        decimals = str(max(-order, 0))
        # Offset a rounded grid from the unchanged center's seconds.
        rounded = round(second / gridwidth) * gridwidth
        factor = 15 * cos_dec if mode == 'ra' else sign_dec
        ticks = (intgrid * gridwidth - second + rounded) * factor
        ticksminor = np.linspace(ticks[0], ticks[-1], 6*nticksminor + 1)
        tickvalues = get_tickvalues(ticks, mode, no_sec)
        whole, frac = np.divmod(tickvalues, 1)
        u = units[mode]['m' if no_sec else 's']
        ticklabels = [f'{int(i):02d}{u}' + f'{j:.{decimals}f}'[2:]
                      for i, j in zip(whole % 60, frac)]
        return ticks, ticksminor, ticklabels

    xticks, xticksminor, xticklabels = makegrid('ra')
    yticks, yticksminor, yticklabels = makegrid('dec')
    i_ref = np.where(np.abs(xticks) < rmax)[0][-1]
    idx_top = -1 if sign_dec > 0 else 0
    j_ref = np.where(np.abs(yticks) < rmax)[0][idx_top]
    ra_hm = get_hmdm(sky_at([xticks[i_ref] / 3600., 0]), 'ra')
    dec_dm = get_hmdm(sky_at([0, yticks[j_ref] / 3600.]), 'dec')
    if on_min_scale:
        dec_dm = dec_dm.split('d')[0] + 'd'
    ra_hm = ra_hm.translate(str.maketrans(units['ra']))
    dec_dm = dec_dm.translate(str.maketrans(units['dec']))
    textpad = ' ' * 12  # To shift the tick label to left.
    xticklabels[i_ref] = ra_hm + xticklabels[i_ref] + textpad
    yticklabels[j_ref] = dec_dm + '\n' + yticklabels[j_ref]
    return ((xticks, xticksminor, xticklabels),
            (yticks, yticksminor, yticklabels))


@dataclass(frozen=True)
class _PanelLayout:
    """Channel selection and panel dimensions, before any figures are created."""
    v: np.ndarray
    nv: int
    nrows: int
    ncols: int
    npages: int
    nchan: int
    channelnumber: int | None
    animation: bool


def _plan_panels(v: np.ndarray, pv: bool, channelnumber: int | np.integer | None,
                 nrows: int, ncols: int) -> _PanelLayout:
    """Select an animation channel or pad the velocity grid for complete pages."""
    nv = len(v)
    animation = isinstance(channelnumber, (int, np.integer))
    if channelnumber is not None:
        channelnumber = int(channelnumber)
        if not -nv <= channelnumber < nv:
            raise ValueError('channelnumber must be'
                             + f'between {-nv} and {nv - 1}.')
        channelnumber %= nv
    if pv or len(v) == 1 or animation:
        nrows = ncols = npages = nchan = 1
    else:
        npages = int(np.ceil(nv / nrows / ncols))
        nchan = npages * nrows * ncols
        v = reform_grid(v, k1=nchan - nv)
    return _PanelLayout(v, nv, nrows, ncols, npages, nchan,
                        channelnumber, animation)


def _create_panels(plan: _PanelLayout, fig: Figure | None, ax: Axes | None,
                   figsize: RealSequence, rmax: RealScalar, veldigit: int
                   ) -> tuple[list[Figure], np.ndarray]:
    """Create shared axes and velocity labels, or reuse a supplied single panel."""
    internalfig = fig is None
    internalax = ax is None
    nrows, ncols = plan.nrows, plan.ncols
    nij2ch = _get_nij2ch(nrows=nrows, ncols=ncols)
    ch2nij = _get_ch2nij(nrows=nrows, ncols=ncols)
    axes = (np.empty(plan.nchan, dtype=object) if internalax
            else np.array([ax], dtype=object))
    need_vlabel = plan.nchan > 1 or plan.animation
    figs = []
    for ch in range(plan.nchan):
        n, i, j = ch2nij(ch)
        if n == len(figs):
            if internalfig:
                fig = plt.figure(figsize=figsize)
            figs.append(fig)
            if need_vlabel:
                fig.subplots_adjust(hspace=0, wspace=0,
                                    right=0.87, top=0.87)
        else:
            fig = figs[n]
        if internalax:
            sharex = axes[nij2ch(n, i - 1, j)] if i > 0 else None
            sharey = axes[nij2ch(n, i, j - 1)] if j > 0 else None
            axes[ch] = fig.add_subplot(nrows, ncols, i*ncols + j + 1,
                                       sharex=sharex, sharey=sharey)
        if need_vlabel and ch < plan.nv:
            vlabel = plan.v[ch if plan.channelnumber is None else plan.channelnumber]
            axes[ch].text(0.9 * rmax, 0.7 * rmax,
                          rf'${vlabel:.{veldigit}f}$', color='black',
                          backgroundcolor='white', zorder=20)
    return figs, axes


class PlotAstroData(AstroFrame):
    """Make a figure from 2D/3D FITS files or 2D/3D arrays.

    Basic rules ---
    For 3D data, a 1D velocity array or a FITS file with a velocity axis
    must be given to set up channels in each page.
    For 2D/3D data, the spatial center can be read from a FITS file or
    manually given.
    len(v)=1 (default) means to make a 2D figure.
    Spatial lengths are in the unit of arcsec, or au if dist (!= 1) is
    given.
    Angles are in degrees, from original data +y toward +x. Regions,
    beams, lines, arrows, and polarization segments follow this convention.
    swapxy maps angles to 90 - angle; xflip and yflip affect only the display.
    Relative numeric positions remain fractions of the displayed axes.
    For region, line, arrow, label, and marker, a single input can be
    treated without a list, e.g., anglelist=60, as well as
    anglelist=[60].
    Each element of poslist supposes a text coordinate like '01h23m45.6s
    01d23m45.6s' or a list of relative x and y like [0.2, 0.3] (0 is
    left or bottom, 1 is right or top).
    Parameters for original methods in matplotlib.axes.Axes can be used
    as kwargs; see the default _kw for reference.
    Position-velocity diagrams (pv=True) do not yet support region,
    line, arrow, and segment because the units of abscissa and ordinate
    are different.

    kwargs is the arguments of AstroFrame to define plotting ranges.

    Args:
        v (np.ndarray, optional): Used to set up channels if fitsimage
            not given. Defaults to None.
        vskip (int or np.integer, optional): How many channels are skipped.
            Defaults to 1.
        veldigit (int or np.integer, optional): How many digits after
            the decimal point. Defaults to 2.
        restfreq (float, optional): Used for velocity and brightness T.
            Defaults to None.
        channelnumber (int or np.integer, optional): Specify a channel number
            to make 2D maps. Defaults to None.
        nrows (int or np.integer, optional): Used for channel maps.
            Defaults to 4.
        ncols (int or np.integer, optional): Used for channel maps.
            Defaults to 6.
        fontsize (int, optional): rcParams['font.size']. None means 18
            (2D) or 12 (3D). Defaults to None.
        nancolor (str, optional): Color for masked regions. Defaults to
            white.
        dpi (int, optional): Dot per inch for plotting an image.
            Defaults to 256.
        figsize (tuple, optional): Defaults to None.
        fig (Figure, optional): External figure. Supply together with ax,
            for a single panel only. Defaults to None.
        ax (Axes, optional): Axes belonging to fig. Defaults to None.

    Unspecified velocity limits use the available grid for channel setup;
    explicit limits may extend it. channelnumber indexes the resulting grid
    after vskip, accepts NumPy integers and negative indices, and is stored
    as a nonnegative Python integer. nrows, ncols, and vskip must be positive
    Python/NumPy integers; veldigit must be a nonnegative integer. These counts
    reject booleans, floats, and arrays. Invalid layout or selection inputs
    raise ValueError before figures are created.
    """
    fig: Figure | None
    figs: list[Figure]
    ax: np.ndarray
    rowcol: int
    npages: int
    allchan: np.ndarray
    bottomleft: np.ndarray
    channelnumber: int | None
    animation: bool
    v: np.ndarray
    vskipfill: Callable[[np.ndarray, np.ndarray | None], np.ndarray]
    _kw: dict[str, Any]
    sigma: float | np.longdouble | None | list[float | np.longdouble | None]
    beam: ValidatedBeams

    def __init__(self,
                 v: RealSequence | None = None, vskip: int | np.integer = 1,
                 veldigit: int | np.integer = 2,
                 restfreq: float | np.floating | None = None,
                 channelnumber: int | np.integer | None = None,
                 nrows: int | np.integer = 4, ncols: int | np.integer = 6,
                 fontsize: float | None = None,
                 nancolor: str | tuple[float, ...] = 'w', dpi: float = 256,
                 figsize: RealSequence | None = None,
                 fig: Figure | None = None, ax: Axes | None = None,
                 **kwargs: Any) -> None:
        _require_integer(nrows, 'nrows')
        _require_integer(ncols, 'ncols')
        _require_integer(vskip, 'vskip')
        nrows, ncols, vskip = int(nrows), int(ncols), int(vskip)
        _require_integer(veldigit, 'veldigit', minimum=0)
        if channelnumber is not None and (
                isinstance(channelnumber, (bool, np.bool_))
                or not isinstance(channelnumber, (int, np.integer))):
            raise ValueError('channelnumber must be an integer or None.')
        if (fig is None) != (ax is None):
            raise ValueError('Provide fig and ax together, or omit both.')
        if fig is not None and (not isinstance(fig, Figure)
                                or not isinstance(ax, Axes)
                                or ax.figure is not fig):
            raise ValueError('fig and ax must be a Figure and its own Axes.')
        super().__init__(**kwargs)
        self._explicit_vmin = 'vmin' in kwargs
        self._explicit_vmax = 'vmax' in kwargs
        internalfig = fig is None
        v = _get_v(p=self, v=v, restfreq=restfreq, vskip=vskip)
        plan = _plan_panels(v, self.pv, channelnumber, nrows, ncols)
        if ax is not None and plan.nchan != 1:
            raise ValueError('External fig and ax support a single panel only;'
                             + ' select channelnumber for a cube.')
        if fontsize is None:
            fontsize = 18 if plan.nchan == 1 else 12
        set_rcparams(fontsize=fontsize, nancolor=nancolor, dpi=dpi)
        figsize = get_figsize(xmin=self.xmin, xmax=self.xmax,
                              ymin=self.ymin, ymax=self.ymax,
                              figsize=figsize, ncols=plan.ncols,
                              nrows=plan.nrows, nchan=plan.nchan)
        figs, axes = _create_panels(plan, fig, ax, figsize, self.rmax, veldigit)
        self.fig = None if internalfig else fig
        self.figs = figs
        self.ax = axes
        self.rowcol = plan.nrows * plan.ncols
        self.npages = plan.npages
        self.allchan = np.arange(plan.nv)
        nij2ch = _get_nij2ch(nrows=plan.nrows, ncols=plan.ncols)
        self.bottomleft = nij2ch(np.arange(plan.npages), plan.nrows - 1, 0)
        self.channelnumber = plan.channelnumber
        self.animation = plan.animation
        self.v = plan.v
        self.vskipfill = _get_vskipfill(plan.nv, plan.v, vskip,
                                        plan.channelnumber)

    def _map_init(self, kw: dict[str, Any]
                  ) -> tuple[np.ndarray | list[np.ndarray | None],
                             np.ndarray | None,
                             np.ndarray | None,
                             np.ndarray | None,
                             float | np.longdouble | None | list[float | np.longdouble | None],
                             str | None | list[str | None],
                             dict[str, Any], bool]:
        """
        Common process for add_color, add_contour, add_segment, and
        add_rgb.
        xskip and yskip (int) mean spatial pixel skips, which defaults
        to 1. Consumes configuration keywords, updates plotting state, and
        draws the beam. Internal multiple-dataset use returns lists for
        data, sigma, and brightness units.

        Args:
            kw (dict): kwargs input for each method.

        Returns:
            tuple: (data, x, y, v, sigma, bunit, plotting kwargs, singlepix).
        """
        b = kwargs2instance(Beam, kw)
        self._kw.update(kw)
        xskip = self._kw.pop('xskip', 1)
        yskip = self._kw.pop('yskip', 1)
        d = kwargs2instance(AstroData, self._kw)
        self.read(d, xskip, yskip)
        self.sigma = d.sigma
        singlepix = d.dx is None or d.dy is None
        if len(d.beam) == 4:
            beam = next((onebeam for onebeam in d.beam
                         if all(a is not None for a in onebeam)),
                        [None, None, None],)
        else:
            beam = d.beam
        b.beam = self.beam = Beam(beam=beam).beam
        self.add_beam(**b.todict())
        return (d.data, d.x, d.y, d.v, d.sigma, d.bunit,
                self._kw, singlepix)

    def _validchan(self, include_chan: list[int | np.integer]
                   | tuple[int | np.integer, ...] | np.ndarray | None
                   ) -> np.ndarray:
        """Return axes indices; None includes every available channel."""
        chans = (self.allchan if include_chan is None
                 else np.asarray(include_chan))
        if self.animation:
            return np.array([0] if self.channelnumber in chans
                            else [], dtype=int)
        return chans

    def add_region(self, patch: Literal['ellipse', 'rectangle'] = 'ellipse',
                   poslist: Positions | None = None,
                   majlist: OverlayValues | None = None,
                   minlist: OverlayValues | None = None,
                   palist: OverlayValues | None = None,
                   include_chan: ChannelIndices | None = None,
                   **kwargs: Any) -> None:
        """Use add_patch() and Rectangle or Ellipse of matplotlib.

        Default keyword values:
            Matplotlib patch: ``facecolor='none'``,
            ``edgecolor='gray'``, ``linewidth=1.5``, and ``zorder=10``.
            User-supplied keyword arguments override these values.

        Args:
            patch (str, optional): 'ellipse' or 'rectangle'. Defaults to
                'ellipse'.
            poslist (str, pair, or sequence, optional): Sky coordinate or
                relative [x, y] pair, or a list/tuple/array of positions.
                None or an empty sequence draws nothing.
            majlist (scalar or sequence, optional): Height (major-axis length).
                Scalars broadcast; sequences must match the position count.
                Required when positions are supplied.
            minlist (scalar or sequence, optional): Width (minor-axis length).
                Scalars broadcast; sequences must match the position count.
                Required when positions are supplied.
            palist (scalar or sequence, optional): Position angle in degrees.
                Zero is original data +y; positive angles turn toward original
                data +x. swapxy maps it to 90 - palist. Axis flips change only
                the display, so regions remain aligned with the data.
                Scalars broadcast; sequences must match the position count.
                Required when positions are supplied.
            include_chan (list, tuple, or ndarray, optional): Integer channel
                indices. None means all.
        """
        _kw = {'facecolor': 'none', 'edgecolor': 'gray',
               'linewidth': 1.5, 'zorder': 10}
        _kw.update(kwargs)
        if patch not in ['rectangle', 'ellipse']:
            raise ValueError("patch must be 'rectangle' or 'ellipse'.")

        x, y = self.pos2xy([] if poslist is None else poslist)
        widths = _overlay_values(minlist, len(x), 'minlist')
        heights = _overlay_values(majlist, len(x), 'majlist')
        angles = _overlay_values(palist, len(x), 'palist')
        for x, y, width, height, angle in zip(x, y, widths, heights, angles):
            for ch, axnow in enumerate(self.ax):
                if ch not in self._validchan(include_chan):
                    continue
                # Matplotlib rotates the width edge from +x; palist describes
                # the height edge from original data +y toward +x.
                theta = -(90 - angle if self.swapxy else angle)
                if patch == 'rectangle':
                    a = np.radians(theta)
                    xp = x - (width*np.cos(a) - height*np.sin(a)) / 2.
                    yp = y - (width*np.sin(a) + height*np.cos(a)) / 2.
                    p = Rectangle
                else:
                    xp, yp = x, y
                    p = Ellipse
                p = p((xp, yp), width=width, height=height,
                      angle=theta, **_kw)
                axnow.add_patch(p)

    def add_beam(self, **kwargs: Any) -> None:
        """Use add_region() to plot the beam.

        kwargs may include the arguments of Beam, except for
        beam_kwargs, to specify the beam appearance. A single
        ``beamcolor`` or ``beampos`` value applies to every beam;
        per-beam lists must provide one value for each beam. A
        ``[None, None, None]`` beam is skipped. bpa is measured from the
        processed data +y axis toward +x, as stored by AstroFrame.read.
        This applies to both automatic and explicitly supplied beams,
        including PV beams. The angle is not swapped again; axis flips
        reflect the beam with the data.
        Relative beampos coordinates refer to the displayed axes and are not
        exchanged by swapxy.

        Default keyword values:
            Beam patch: ``facecolor=beamcolor`` and ``edgecolor=None``.
            Other keyword arguments override these values and are passed
            to ``add_region``.
        """
        b = kwargs2instance(Beam, kwargs)
        show_beam, beamcolor, beampos = b.show_beam, b.beamcolor, b.beampos
        beam = b.beam
        kwargs.pop('beam', None)
        b.validate_display()
        if not show_beam:
            return

        include_chan = self.allchan if self.animation else self.bottomleft
        patch = 'rectangle' if self.pv else 'ellipse'
        blist = [beam] if np.ndim(beam) == 1 else beam
        n = len(blist)
        bclist = beamcolor if isinstance(beamcolor, list) else [beamcolor] * n
        islist = beampos is not None and not b._is_position(beampos)
        bplist = beampos if islist else [beampos] * n
        for (bmaj, bmin, bpa), bc, bp in zip(blist, bclist, bplist):
            if None in [bmaj, bmin, bpa]:
                print('No beam to plot.')
                continue
            a = max(0.35 * bmaj / self.rmax, 0.1)
            if bp is None:
                bp = [a, 0.1 if self.pv else a]
            _kw = {'facecolor': bc, 'edgecolor': None}
            _kw.update(kwargs)
            # add_region accepts original-data angles; undo the stored
            # beam's transformation at this interface so it is drawn once.
            if self.swapxy:
                bpa = 90 - bpa
            self.add_region(patch=patch, poslist=bp,
                            majlist=bmaj, minlist=bmin, palist=bpa,
                            include_chan=include_chan,
                            **_kw)

    def add_marker(self, poslist: Positions | None = None,
                   include_chan: ChannelIndices | None = None,
                   **kwargs: Any) -> None:
        """Use Axes.plot of matplotlib.

        Default keyword values:
            Matplotlib: ``marker='+'``, ``ms=10``, ``mfc='gray'``,
            ``mec='gray'``, ``mew=2``, ``alpha=1``, and ``zorder=10``.
            User-supplied keyword arguments override these values.

        Args:
            poslist (str, pair, or sequence, optional): Sky coordinate or
                relative [x, y] pair, or a list/tuple/array of positions.
                None or an empty sequence draws nothing.
            include_chan (list, tuple, or ndarray, optional): Integer channel
                indices. None means all.
        """
        _kw = {'marker': '+', 'ms': 10, 'mfc': 'gray',
               'mec': 'gray', 'mew': 2, 'alpha': 1, 'zorder': 10}
        _kw.update(kwargs)
        for ch, axnow in enumerate(self.ax):
            if ch not in self._validchan(include_chan):
                continue
            for x, y in zip(*self.pos2xy([] if poslist is None else poslist)):
                axnow.plot(x, y, **_kw)

    def add_text(self, poslist: Positions | None = None,
                 slist: str | list[str] | tuple[str, ...]
                 | np.ndarray | None = None,
                 include_chan: ChannelIndices | None = None,
                 **kwargs: Any) -> None:
        """Use Axes.text of matplotlib.

        Default keyword values:
            Matplotlib: ``color='gray'``, ``fontsize=15``,
            ``ha='center'``, ``va='center'``, and ``zorder=10``.
            User-supplied keyword arguments override these values.
            Aliases: ``horizontalalignment`` and ``verticalalignment``
            are accepted as aliases for ``ha`` and ``va``.

        Args:
            poslist (str, pair, or sequence, optional): Sky coordinate or
                relative [x, y] pair, or a list/tuple/array of positions.
                None or an empty sequence draws nothing.
            slist (str or sequence, optional): Text to draw. A string broadcasts;
                sequences must match the position count. Required with positions.
            include_chan (list, tuple, or ndarray, optional): Integer channel
                indices. None means all.
        """
        _kw = {'color': 'gray', 'fontsize': 15, 'ha': 'center',
               'va': 'center', 'zorder': 10}
        subkeys = {'ha': 'horizontalalignment',
                   'va': 'verticalalignment'}
        for short, long in subkeys.items():
            if long in kwargs:
                kwargs[short] = kwargs.pop(long)
        _kw.update(kwargs)
        x, y = self.pos2xy([] if poslist is None else poslist)
        labels = _overlay_values(slist, len(x), 'slist')
        for ch, axnow in enumerate(self.ax):
            if ch not in self._validchan(include_chan):
                continue
            for xp, yp, label in zip(x, y, labels):
                axnow.text(x=xp, y=yp, s=label, **_kw)

    def add_line(self, poslist: Positions | None = None,
                 anglelist: OverlayValues | None = None,
                 rlist: OverlayValues | None = None,
                 include_chan: ChannelIndices | None = None,
                 **kwargs: Any) -> None:
        """Use Axes.plot of matplotlib.

        Default keyword values:
            Matplotlib: ``color='gray'``, ``linewidth=1.5``,
            ``linestyle='-'``, and ``zorder=10``. User-supplied keyword
            arguments override these values.

        Args:
            poslist (str, pair, or sequence, optional): Sky coordinate or
                relative [x, y] pair, or a list/tuple/array of positions.
                None or an empty sequence draws nothing.
            anglelist (scalar or sequence, optional): Angle in degrees
                from original data +y toward +x. swapxy maps it to 90 - angle;
                axis flips reflect the direction with the data.
                Scalars broadcast; sequences must match the position count.
                Required when positions are supplied.
            rlist (scalar or sequence, optional): Length in data-coordinate units.
                Scalars broadcast; sequences must match the position count.
                Required when positions are supplied.
            include_chan (list, tuple, or ndarray, optional): Integer channel
                indices. None means all.
        """
        _kw = {'color': 'gray', 'linewidth': 1.5,
               'linestyle': '-', 'zorder': 10}
        _kw.update(kwargs)
        x, y = self.pos2xy([] if poslist is None else poslist)
        angles = _overlay_values(anglelist, len(x), 'anglelist')
        radii = _overlay_values(rlist, len(x), 'rlist')
        for ch, axnow in enumerate(self.ax):
            if ch not in self._validchan(include_chan):
                continue
            for xp, yp, angle, r in zip(x, y, angles, radii):
                a = np.radians(90 - angle if self.swapxy else angle)
                axnow.plot([xp, xp + r * np.sin(a)],
                           [yp, yp + r * np.cos(a)], **_kw)

    def add_arrow(self, poslist: Positions | None = None,
                  anglelist: OverlayValues | None = None,
                  rlist: OverlayValues | None = None,
                  include_chan: ChannelIndices | None = None,
                  **kwargs: Any) -> None:
        """Use Axes.quiver of matplotlib.

        Angles follow the original data axes with the default angles='xy'.
        Overriding the quiver angle mode can change this interpretation.

        Default keyword values:
            Matplotlib: ``color='gray'``, ``width=0.012``,
            ``headwidth=5``, ``headlength=5``, ``zorder=10``,
            ``angles='xy'``, ``scale_units='xy'``, and ``scale=1``.
            User-supplied keyword arguments override these values.

        Args:
            poslist (str, pair, or sequence, optional): Sky coordinate or
                relative [x, y] pair, or a list/tuple/array of positions.
                None or an empty sequence draws nothing.
            anglelist (scalar or sequence, optional): Angle in degrees
                from original data +y toward +x. swapxy maps it to 90 - angle;
                axis flips reflect the direction with the data.
                Scalars broadcast; sequences must match the position count.
                Required when positions are supplied.
            rlist (scalar or sequence, optional): Length in data-coordinate units.
                Scalars broadcast; sequences must match the position count.
                Required when positions are supplied.
            include_chan (list, tuple, or ndarray, optional): Integer channel
                indices. None means all.
        """
        _kw = {'color': 'gray', 'width': 0.012,
               'headwidth': 5, 'headlength': 5, 'zorder': 10,
               'angles': 'xy', 'scale_units': 'xy', 'scale': 1}
        _kw.update(kwargs)
        x, y = self.pos2xy([] if poslist is None else poslist)
        angles = _overlay_values(anglelist, len(x), 'anglelist')
        radii = _overlay_values(rlist, len(x), 'rlist')
        for ch, axnow in enumerate(self.ax):
            if ch not in self._validchan(include_chan):
                continue
            for xp, yp, angle, r in zip(x, y, angles, radii):
                a = np.radians(90 - angle if self.swapxy else angle)
                axnow.quiver(xp, yp, r * np.sin(a), r * np.cos(a),
                             **_kw)

    def add_scalebar(self, length: RealScalar, label: str = '',
                     color: str | tuple[float, ...] = 'gray',
                     barpos: PositionPair = (0.8, 0.12),
                     fontsize: RealScalar | None = None,
                     linewidth: RealScalar = 3,
                     bbox: dict[str, Any] | None = None) -> None:
        """Draw a spatial scalebar using Matplotlib text and plot.

        Position-velocity (PV) diagrams are not supported.

        Args:
            length (float): Required finite positive length in horizontal
                data-axis units (arcsec, or au when distance scaling is used).
                Booleans, complex values, and arrays are rejected.
            label (str, optional): Text like '100 au'. Defaults to ''.
            color (str, optional): Same for bar and label. Defaults to 'gray'.
            barpos (list, tuple, or ndarray, optional): Relative position.
                Defaults to (0.8, 0.12).
            fontsize (float, optional): None means 20 points for one panel
                and 15 points for multiple panels. Defaults to None.
            linewidth (float, optional): Width of the bar. Defaults to 3.
            bbox (dict, optional): Keyword arguments for the text
                bounding box. None uses {'alpha': 0}.
        Raises:
            ValueError: If pv=True, length is not finite and positive, or
                barpos does not contain two finite coordinates.
        """
        if self.pv:
            raise ValueError('add_scalebar() does not support position-velocity'
                             + ' (PV) diagrams; use it for spatial maps only.')
        _require_positive_real_scalar(length, 'length')
        if np.shape(barpos) != (2,) or not all(np.isfinite(v) for v in barpos):
            raise ValueError('barpos must contain two finite'
                             + ' relative coordinates.')
        bbox = {'alpha': 0} if bbox is None else bbox
        if fontsize is None:
            fontsize = 20 if len(self.ax) == 1 else 15
        for ch, axnow in enumerate(self.ax):
            if ch not in self.bottomleft:
                continue
            x, y = self.pos2xy([barpos[0], barpos[1] - 0.012])
            axnow.text(x[0], y[0], label, color=color, size=fontsize,
                       ha='center', va='top', bbox=bbox, zorder=10)
            x, y = self.pos2xy([barpos[0], barpos[1] + 0.012])
            axnow.plot([x[0] - length/2., x[0] + length/2.], [y[0], y[0]],
                       '-', linewidth=linewidth, color=color)

    def _set_colorbar(self, mappable: list[QuadMesh | None], ch: int,
                      show_cbar: bool,
                      cblabel: str | None, cbformat: str,
                      cbticks: RealSequence | None,
                      cbticklabels: list[str] | tuple[str, ...]
                      | np.ndarray | None,
                      cblocation: Literal['left', 'right', 'top', 'bottom'],
                      cblabelfontsize: RealScalar, cbtickfontsize: RealScalar,
                      st: Stretcher) -> None:
        if not show_cbar:
            return

        if mappable[ch] is None:
            raise ValueError('Cannot create a colorbar'
                             + ' without a plotted mappable.')
        fig = self.figs[ch // self.rowcol]
        if len(self.ax) == 1:
            ax = self.ax[ch]
            cb = fig.colorbar(mappable[ch], ax=ax, label=cblabel,
                              format=cbformat, location=cblocation)
        else:
            cax = fig.add_axes([0.88, 0.105, 0.015, 0.77])
            cb = fig.colorbar(mappable[ch], cax=cax, label=cblabel,
                              format=cbformat)
        cb.ax.tick_params(labelsize=cbtickfontsize)
        font = mpl.font_manager.FontProperties(size=cblabelfontsize)
        horizontal = cb.orientation == 'horizontal'
        label_axis = cb.ax.xaxis if horizontal else cb.ax.yaxis
        label_axis.label.set_font_properties(font)
        cmin, cmax = mappable[ch].get_clim()
        if cbticks is None and st.stretch == 'log':
            cbticks, cbticklabels = logcbticks(10**cmin, 10**cmax)
        cbticks = cb.get_ticks() if cbticks is None else st.do(cbticks)
        cond = (cmin <= cbticks) * (cbticks <= cmax)
        cbticks = cbticks[cond]
        cb.set_ticks(cbticks)
        if cbticklabels is None:
            formatter = (mpl.ticker.StrMethodFormatter(cbformat)
                         if '{' in cbformat
                         else mpl.ticker.FormatStrFormatter(cbformat))
            cbticklabels = [formatter(t) for t in st.undo(cbticks)]
        else:
            cbticklabels = np.asarray(cbticklabels)[cond]
        cb.set_ticklabels(cbticklabels)

    def add_color(self,
                  show_cbar: bool = True,
                  cblabel: str | None = None,
                  cbformat: str = '%.1e',
                  cbticks: RealSequence | None = None,
                  cbticklabels: list[str] | tuple[str, ...]
                  | np.ndarray | None = None,
                  cblocation: Literal['left', 'right', 'top', 'bottom'] = 'right',
                  cblabelfontsize: RealScalar = 16,
                  cbtickfontsize: RealScalar = 14,
                  **kwargs: Any) -> None:
        """Use Axes.pcolormesh of matplotlib.

        Keyword groups accepted in ``**kwargs``:
            AstroData: Data input and metadata, such as ``fitsimage``,
            ``data``, ``x``, ``y``, ``v``, ``beam``, ``Tb``, ``sigma``,
            ``center``, ``restfreq``, ``cfactor``, ``pvpa``, ``pv``, and
            ``bunit``.
            Stretcher: Color scaling, such as ``stretch``,
            ``stretchscale``, ``stretchpower``, ``vmin``, and ``vmax``.
            Beam: Beam display, such as ``show_beam``, ``beamcolor``,
            ``beampos``, and ``beam_kwargs``.
            Sampling: ``xskip`` and ``yskip``.
            Matplotlib: Additional keyword arguments are passed to
            ``matplotlib.axes.Axes.pcolormesh``.

        Default keyword values:
            Matplotlib: ``cmap='cubehelix'``, ``alpha=1``,
            ``edgecolors='none'``, ``zorder=1``, ``vmin=None``, and
            ``vmax=None``. User-supplied keyword arguments override
            these values.

        Args:
            show_cbar (bool, optional): Show color bar. Defaults to
                True.
            cblabel (str, optional): Colorbar label. Defaults to None.
            cbformat (str, optional): Numeric printf-style (e.g. '%.1e')
                or brace-style (e.g. '{x:.1f}') format. Defaults to '%.1e'.
            cbticks (list, tuple, or ndarray, optional): Finite 1D tick values
                in original, unstretched units. Defaults to None.
            cbticklabels (list, tuple, or ndarray, optional): One label per
                explicit cbtick. Defaults to None (formatted automatically).
            cblocation (str, optional): 'left', 'right', 'top', or 'bottom'.
                Applies to single panels; channel maps use a fixed right-side
                colorbar. Defaults to 'right'.
            cblabelfontsize (float, optional): Fontsize for the colorbar
                label. This is independent of set_rcparams().
            cbtickfontsize (float, optional): Fontsize for the colorbar
                ticks. This is independent of set_rcparams().
        """
        if show_cbar:
            cbticks = _validate_colorbar(cbformat, cbticks, cbticklabels,
                                         cblocation)
        self._kw = {'cmap': 'cubehelix', 'alpha': 1,
                    'edgecolors': 'none', 'zorder': 1,
                    'vmin': None, 'vmax': None}
        c, x, y, v, sigma, bunit, _kw, singlepix = self._map_init(kwargs)
        if singlepix:
            raise ValueError(
                'add_color() requires at least two x and y pixels.')

        cblabel = bunit if cblabel is None else cblabel

        _kw['sigma'] = sigma
        st = kwargs2instance(Stretcher, _kw)
        c, cmin, cmax = st.set_minmax(c)
        _kw['vmin'] = cmin
        _kw['vmax'] = cmax
        c = self.vskipfill(c, v)
        p = [None] * len(self.ax)
        for ch, (axnow, cnow) in enumerate(zip(self.ax, c)):
            pnow = axnow.pcolormesh(x, y, cnow, **_kw)
            if ch in self.bottomleft:
                p[ch] = pnow
        for ch in self.bottomleft:
            self._set_colorbar(p, ch, show_cbar, cblabel, cbformat,
                               cbticks, cbticklabels, cblocation,
                               cblabelfontsize, cbtickfontsize, st)

    def add_contour(self,
                    levels: RealSequence = (-12, -6, -3, 3, 6, 12,
                                            24, 48, 96, 192, 384),
                    **kwargs: Any) -> None:
        """Use Axes.contour of matplotlib.

        Keyword groups accepted in ``**kwargs``:
            AstroData: Data input and metadata, such as ``fitsimage``,
            ``data``, ``x``, ``y``, ``v``, ``beam``, ``Tb``, ``sigma``,
            ``center``, ``restfreq``, ``cfactor``, ``pvpa``, ``pv``, and
            ``bunit``.
            Beam: Beam display, such as ``show_beam``, ``beamcolor``,
            ``beampos``, and ``beam_kwargs``.
            Sampling: ``xskip`` and ``yskip``.
            Matplotlib: Additional keyword arguments are passed to
            ``matplotlib.axes.Axes.contour``.

        Default keyword values:
            Matplotlib: ``colors='gray'``, ``linewidths=1.0``, and
            ``zorder=2``. User-supplied keyword arguments override these
            values.

        Args:
            levels (list, tuple, or ndarray, optional): Finite distinct contour
                levels in units of sigma. A finite positive scalar sigma is
                required. Defaults to [-12,-6,-3,3,6,12,24,48,96,192,384].
        """
        levels = np.asarray(levels)
        if (levels.ndim != 1
                or levels.size == 0
                or not np.all(np.isfinite(levels))
                or np.unique(levels).size != levels.size):
            raise ValueError('levels must be a nonempty 1D sequence'
                             + ' of finite distinct values.')
        self._kw = {'colors': 'gray', 'linewidths': 1.0, 'zorder': 2}
        c, x, y, v, sigma, _, _kw, singlepix = self._map_init(kwargs)
        if singlepix:
            raise ValueError('add_contour() requires at least'
                             + ' two x and y pixels.')

        _positive_noise(sigma, 'sigma')
        c = self.vskipfill(c, v)
        for axnow, cnow in zip(self.ax, c):
            axnow.contour(x, y, cnow, np.sort(levels) * sigma, **_kw)

    def _prepare_polarization(self, data: list[np.ndarray | None],
                              sigma: list[float | np.longdouble | None],
                              cutoff: float, angonly: bool
                              ) -> tuple[np.ndarray, np.ndarray] | None:
        """Derive original-frame angles and normalized amplitudes for segments.

        Stokes inputs update public sigma to a Python float, preserving
        extended precision. Invalid or zero-only fields produce no segments.
        """
        amp, ang, stU, stQ = data
        sigmaU, sigmaQ = sigma[2:]
        if stU is not None and stQ is not None:
            _positive_noise(sigmaU, 'Stokes U sigma')
            _positive_noise(sigmaQ, 'Stokes Q sigma')
            if stU.shape != stQ.shape:
                raise ValueError('Stokes U and Q must have matching shapes.')
            self.sigma = sigma = _normalize_float((sigmaU + sigmaQ) / 2.)
            ang = np.degrees(np.arctan2(stU, stQ) / 2.)
            amp = np.hypot(stU, stQ)
            amp[amp < cutoff * sigma] = np.nan
        if amp is None:
            amp = np.ones_like(ang)
        if amp.shape != ang.shape:
            raise ValueError('Amplitude and angle datasets must'
                             + ' have matching shapes.')
        valid = np.isfinite(amp) & np.isfinite(ang)
        if np.any(amp[valid] < 0):
            raise ValueError('Segment amplitudes must be nonnegative.')
        if not np.any(valid) or not np.any(amp[valid] > 0):
            print('No valid segments to draw.')
            return None  # No finite nonzero segments to draw.
        amp = np.where(valid, amp, np.nan)
        if angonly:
            amp = np.sign(amp)**2
        amp = amp / np.nanmax(amp)
        return amp, ang

    def add_segment(self,
                    ampfits: str | PathLike[str] | None = None,
                    angfits: str | PathLike[str] | None = None,
                    Ufits: str | PathLike[str] | None = None,
                    Qfits: str | PathLike[str] | None = None,
                    amp: np.ndarray | None = None,
                    ang: np.ndarray | None = None,
                    stU: np.ndarray | None = None,
                    stQ: np.ndarray | None = None,
                    ampfactor: float = 1., angonly: bool = False,
                    rotation: float = 0.,
                    cutoff: float = 3.,
                    **kwargs: Any) -> None:
        """Use Axes.quiver of matplotlib.

        ``fitsimage`` is built from
        ``[ampfits, angfits, Ufits, Qfits]``, and ``data`` is built from
        ``[amp, ang, stU, stQ]``.

        Angles are data-relative, including angles derived from Stokes U/Q.
        swapxy exchanges direction components (angle becomes 90 - angle).
        xflip and yflip change only the display; segments reflect with the
        data. This convention uses the default angles='xy' quiver setting.
        Supply angles or a complete Stokes U/Q pair; Stokes inputs require
        finite positive noise levels. Amplitudes must match angles in shape
        and be nonnegative. Zero or entirely invalid amplitudes draw nothing.
        scale is a finite positive, dimensionless inverse-length factor (default 1).
        It is divided by the displayed x pixel spacing before passing to quiver.
        With the default scale_units='xy', maximum segment length is
        ampfactor * abs(dx) / scale; scale=2 makes segments half as long.
        Here dx is the processed horizontal grid spacing, after sampling and
        swapxy. Overriding scale_units changes the length interpretation.

        Keyword groups accepted in ``**kwargs``:
            AstroData: Data input and metadata, such as ``x``, ``y``,
            ``v``, ``beam``, ``Tb``, ``sigma``, ``center``,
            ``restfreq``, ``cfactor``, ``pvpa``, ``pv``, and ``bunit``.
            The ``fitsimage`` and ``data`` arguments are assembled from
            the segment-specific arguments above.
            Beam: Beam display, such as ``show_beam``, ``beamcolor``,
            ``beampos``, and ``beam_kwargs``.
            Sampling: ``xskip`` and ``yskip``.
            Matplotlib: Additional keyword arguments are passed to
            ``matplotlib.axes.Axes.quiver``.

        Default keyword values:
            Matplotlib: ``angles='xy'``, ``scale_units='xy'``,
            ``color='gray'``, ``pivot='mid'``, ``headwidth=0``,
            ``headlength=0``, ``headaxislength=0``, ``width=0.007``, and
            ``zorder=3``, and dimensionless ``scale=1``. User-supplied keyword
            arguments override these values.

        Args:
            ampfits (str or os.PathLike, optional): Input FITS file name.
                Length of segment. Defaults to None.
            angfits (str or os.PathLike, optional): Input FITS file name.
                Angles in degrees from original data +y toward +x (north
                toward east for sky images). Defaults to None.
            Ufits (str or os.PathLike, optional): Input FITS file name.
                Stokes U. Defaults to None.
            Qfits (str or os.PathLike, optional): Input FITS file name.
                Stokes Q. Defaults to None.
            amp (np.ndarray, optional): Length of segment. Defaults to None.
            ang (np.ndarray, optional): Angles in degrees from original data +y
                toward +x. Defaults to None.
            stU (np.ndarray, optional): Stokes U. Defaults to None.
            stQ (np.ndarray, optional): Stokes Q. Defaults to None.
            ampfactor (float, optional): Multiplies amplitude after
                normalization by its finite maximum. With default quiver scale,
                maximum length is ampfactor times the x pixel spacing.
                Defaults to 1.
            angonly (bool, optional): True means amp=1 for all. Defaults
                to False.
            rotation (float, optional): Added to ang in the original data
                coordinates, before swapxy. Defaults to 0.
            cutoff (float, optional): Used when amp and ang are
                calculated from Stokes U and Q. In the unit of sigma.
                Defaults to 3..
        """
        has_u = stU is not None or Ufits is not None
        has_q = stQ is not None or Qfits is not None
        if has_u != has_q:
            raise ValueError('Provide both Stokes U and Q, or neither.')
        if not has_u and ang is None and angfits is None:
            raise ValueError('Provide an angle dataset or a complete'
                             + ' Stokes U/Q pair.')
        self._kw = {'angles': 'xy', 'scale_units': 'xy', 'color': 'gray',
                    'pivot': 'mid', 'headwidth': 0, 'headlength': 0,
                    'headaxislength': 0, 'width': 0.007, 'zorder': 3,
                    'fitsimage': [ampfits, angfits, Ufits, Qfits],
                    'data': [amp, ang, stU, stQ]}
        c, x, y, v, sigma, _, _kw, singlepix = self._map_init(kwargs)
        if singlepix:
            raise ValueError(
                'add_segment() requires at least two x and y pixels.')

        prepared = self._prepare_polarization(c, sigma, cutoff, angonly)
        if prepared is None:
            return
        amp, ang = prepared
        angle = ang + rotation
        if self.swapxy:
            angle = 90 - angle
        U = ampfactor * amp * np.sin(np.radians(angle))
        V = ampfactor * amp * np.cos(np.radians(angle))
        U = self.vskipfill(U, v)
        V = self.vskipfill(V, v)
        scale = _kw.get('scale', 1.)
        _require_positive_real_scalar(scale, 'Segment scale')
        _kw['scale'] = scale / np.abs(x[1] - x[0])
        for axnow, unow, vnow in zip(self.ax, U, V):
            axnow.quiver(x, y, unow, vnow, **_kw)

    def add_rgb(self, **kwargs: Any) -> None:
        """Render exactly three matching RGB datasets with Matplotlib imshow.

        Supply data or fitsimage as a list/tuple in red, green, blue order.
        Scalar stretch settings broadcast; sequences must contain three values.
        Each channel is normalized to [0, 1]; a zero normalization range gives
        zero intensity. Pixels with any nonfinite input or stretched component
        are transparent, including padded channel panels. Source arrays are
        not modified. xskip, yskip, and show_beam are single values.

        Default origin='lower' maps array rows directly to the supplied y grid;
        pixel-edge extent aligns RGB pixels with color and contour maps.
        Grids must be uniform (rtol=1e-5). Axis flips act through display limits,
        without reversing pixel arrays. Explicit imshow options (including
        alpha, aspect, origin, and extent) override defaults; overriding origin
        or extent can change alignment with other overlays.

        Keyword groups accepted in ``**kwargs``:
            AstroData: Data input and metadata, such as ``fitsimage``,
            ``data``, ``x``, ``y``, ``v``, ``beam``, ``Tb``, ``sigma``,
            ``center``, ``restfreq``, ``cfactor``, ``pvpa``, ``pv``, and
            ``bunit``.
            Stretcher: RGB scaling, such as ``stretch``,
            ``stretchscale``, ``stretchpower``, ``vmin``, and ``vmax``.
            Beam: Beam display, such as ``show_beam``, ``beamcolor``,
            ``beampos``, and ``beam_kwargs``.
            Sampling: ``xskip`` and ``yskip``.
            Matplotlib: Additional keyword arguments are passed to
            ``matplotlib.axes.Axes.imshow``.

        Default keyword values:
            Stretcher: ``vmin=None``, ``vmax=None``,
            ``stretch=['linear'] * 3``, ``stretchscale=None``, and
            ``stretchpower=0.5``. User-supplied keyword arguments
            override these values.
        """
        # Three channels are intrinsic to RGB; scalar stretch settings broadcast.
        source = (kwargs.get('data')
                  if kwargs.get('data') is not None
                  else kwargs.get('fitsimage'))
        if (not isinstance(source, (list, tuple))
                or len(source) != 3
                or any(c is None for c in source)):
            raise ValueError('add_rgb() requires exactly three red, green,'
                             + ' and blue datasets.')
        kwargs = kwargs.copy()
        if isinstance(source, tuple):
            key = 'data' if kwargs.get('data') is not None else 'fitsimage'
            kwargs[key] = list(source)
        self._kw = {'vmin': None, 'vmax': None, 'stretch': ['linear'] * 3,
                    'stretchscale': None, 'stretchpower': 0.5}
        c, x, y, v, sigma, _, _kw, singlepix = self._map_init(kwargs)
        if singlepix:
            raise ValueError('add_rgb() requires at least two x and y pixels.')
        if len(c) != 3 or not (c[0].shape == c[1].shape == c[2].shape):
            raise ValueError('add_rgb() requires three datasets with'
                             + ' matching shapes.')
        if isinstance(_kw['stretch'], str):
            _kw['stretch'] = [_kw['stretch']] * 3
        _kw['sigma'] = sigma
        st = kwargs2instance(Stretcher, _kw)
        _kw.pop('vmin', None)
        _kw.pop('vmax', None)
        if st.n != 3:
            raise ValueError('RGB stretch settings must contain three values.')
        channels = []
        for i, data in enumerate(c):
            normalized = _normalize_rgb_channel(data, st, i)
            channels.append(self.vskipfill(normalized, v))
        rgb = np.stack(channels, axis=-1)  # (channel, y, x, RGB)
        extent = [*_image_edges(x, 'x'), *_image_edges(y, 'y')]
        _kw.setdefault('origin', 'lower')
        _kw.setdefault('extent', extent)
        _kw.setdefault('aspect', np.abs((x[-1] - x[0]) / (y[-1] - y[0])))
        for axnow, image in zip(self.ax, rgb):
            valid = np.all(np.isfinite(image), axis=-1)
            # Float RGBA is a plotting boundary; never modify source arrays.
            rgba = np.zeros((*image.shape[:2], 4), dtype=float)
            rgba[..., :3] = np.clip(np.where(valid[..., None], image, 0), 0, 1)
            rgba[..., 3] = valid
            axnow.imshow(rgba, **_kw)

    def _set_axis_shared(self, pa2: PlotAxes2D,
                         title: dict | str | None) -> None:
        """Internal method used in set_axis() and set_axis_radec().

        Args:
            pa2 (PlotAxes2D): This is instantiated in set_axis() or
                set_axis_radec().
            title (dict | str | None): str means set_title(str) for
                a single panel or fig.suptitle(str) for multiple panels.
                Defaults to None.
        """
        for ch, axnow in enumerate(self.ax):
            pa2.set_xyaxes(axnow)
            self.Xlim = [_normalize_float(v) for v in axnow.get_xlim()]
            self.Ylim = [_normalize_float(v) for v in axnow.get_ylim()]
            if ch not in self.bottomleft:
                plt.setp(axnow.get_xticklabels(), visible=False)
                plt.setp(axnow.get_yticklabels(), visible=False)
                axnow.set_xlabel('')
                axnow.set_ylabel('')
            if len(self.ax) == 1:
                if self.fig is None:
                    self.figs[0].tight_layout()
        if title is not None:
            if len(self.ax) > 1:
                t = {'y': 0.9}
                t_in = {'t': title} if isinstance(title, str) else title
                t.update(t_in)
                for fig in self.figs:
                    fig.suptitle(**t)
            else:
                t = {'label': title} if isinstance(title, str) else title
                axnow.set_title(**t)

    def set_axis(self, title: dict | str | None = None,
                 **kwargs: Any) -> None:
        """Use Axes.set_* of matplotlib. kwargs can include the
        arguments of PlotAxes2D to adjust x and y axis.

        Default keyword values:
            PlotAxes2D: ``xlabel``, ``ylabel``, ``xlim``, and ``ylim``
            are filled from the current frame when they are omitted. In
            PV mode, ``samexy=False`` is also set internally.

        Args:
            title (dict | str | None): str means set_title(str) for
                a single panel or fig.suptitle(str) for multiple panels.
                Defaults to None.
        """
        _kw = {}
        _kw.update(kwargs)
        offunit = '(arcsec)' if self.dist == 1 else '(au)'
        if self.pv:
            offlabel = f'Offset {offunit}'
            vellabel = r'Velocity (km s$^{-1})$'
            if 'xlabel' not in _kw:
                _kw['xlabel'] = vellabel if self.swapxy else offlabel
            if 'ylabel' not in _kw:
                _kw['ylabel'] = offlabel if self.swapxy else vellabel
            _kw['samexy'] = False
        else:
            ralabel, declabel = f'R.A. {offunit}', f'Dec. {offunit}'
            if 'xlabel' not in _kw:
                _kw['xlabel'] = declabel if self.swapxy else ralabel
            if 'ylabel' not in _kw:
                _kw['ylabel'] = ralabel if self.swapxy else declabel
        if 'xlim' not in _kw:
            _kw['xlim'] = self.Xlim
        if 'ylim' not in _kw:
            _kw['ylim'] = self.Ylim
        pa2 = kwargs2instance(PlotAxes2D, _kw)
        self._set_axis_shared(pa2=pa2, title=title)

    def set_axis_radec(self, center: str | None = None,
                       xlabel: str | None = None,
                       ylabel: str | None = None,
                       nticksminor: int | np.integer = 2,
                       grid: dict | bool | None = None,
                       title: dict[str, Any] | str | None = None
                       ) -> None:
        """Label angular spatial axes with R.A. and Dec. coordinates.

        Requires dist=1 and pv=False. With swapxy=True, Dec. ticks and
        labels move to x, and R.A. ticks and labels move to y. Axis limits
        retain the configured display directions. Tick grids are rounded
        relative to the unchanged center. Large fields use whole-minute Dec.
        ticks; zero declination remains a valid reference.

        Args:
            center (str, optional): Defaults to the plot center. A known center
                is required. Its ICRS/FK4/FK5 frame is preserved.
            xlabel (str, optional): R.A. label, placed on y when swapxy=True.
                Defaults to R.A. with the center frame name.
            ylabel (str, optional): Dec. label, placed on x when swapxy=True.
                Defaults to Dec. with the center frame name.
            nticksminor (int or np.integer, optional): Positive integer ratio
                of major to minor tick spacing. Defaults to 2.
            grid (dict or bool, optional): A bool toggles the grid;
                a dict supplies options. Defaults to None.
            title (dict | str | None): str means set_title(str) for
                a single panel or fig.suptitle(str) for multiple panels.
                Defaults to None.
        Raises:
            ValueError: If dist != 1, the plot is a PV diagram, the center is
                missing, or nticksminor is not a positive integer.
        """
        _require_integer(nticksminor, 'nticksminor')
        nticksminor = int(nticksminor)
        if self.pv:
            raise ValueError('set_axis_radec() does not support'
                             + ' position-velocity (PV) diagrams.')
        if self.dist != 1:
            raise ValueError('set_axis_radec() requires dist=1;'
                             + ' distance-scaled axes are not supported.')
        if center is None:
            center = self.center
        if center is None:
            raise ValueError('set_axis_radec() requires a known center;'
                             + ' provide center or fitsimage.')
        center, frame, _ = _getframe(center, as_frame=True)
        frame_name = frame.name.upper()
        xlabel = f'R.A. ({frame_name})' if xlabel is None else xlabel
        ylabel = f'Dec. ({frame_name})' if ylabel is None else ylabel
        xtl, ytl = _radec_ticks(center, frame_name, frame.dec.degree,
                                self.rmax, nticksminor)
        xticks, xticksminor, xticklabels = xtl
        yticks, yticksminor, yticklabels = ytl
        if self.swapxy:
            xticks, yticks = yticks, xticks
            xticksminor, yticksminor = yticksminor, xticksminor
            xticklabels, yticklabels = yticklabels, xticklabels
            xlabel, ylabel = ylabel, xlabel
        pa2 = PlotAxes2D(True, None, 'linear', 'linear',
                         self.Xlim, self.Ylim, xlabel, ylabel,
                         xticks, yticks, xticklabels, yticklabels,
                         xticksminor, yticksminor, grid)
        self._set_axis_shared(pa2=pa2, title=title)

    def savefig(self, filename: str | PathLike[str] | None = None,
                show: bool = False, **kwargs: Any) -> None:
        """Use savefig of matplotlib.

        If ``filename`` is provided, existing files with the same name
        are overwritten by Matplotlib. After optional saving/showing,
        figures managed by this instance are closed. Multiple pages receive
        _0, _1, ... before the final filename suffix (or appended when no
        suffix is present). Matplotlib chooses the default format for filenames
        without an extension. transparent=False preserves an opaque background.

        Default keyword values:
            Figure.savefig: ``transparent=True`` and
            ``bbox_inches='tight'``. User-supplied keyword arguments
            override these values.

        Args:
            filename (str or os.PathLike, optional): Output image file name. Existing
                files may be overwritten, and figures managed by this
                instance are closed after saving/showing. Defaults to
                None.
            show (bool, optional): True means doing plt.show(). Defaults
                to False.
        """
        _kw = {'transparent': True, 'bbox_inches': 'tight'}
        _kw.update(kwargs)
        for axnow in self.ax:
            axnow.set_xlim(*self.Xlim)
            axnow.set_ylim(*self.Ylim)
        if filename is not None:
            path = Path(filename)
            for i, fig in enumerate(self.figs):
                fname = (path if self.npages == 1 else
                         path.with_name(f'{path.stem}_{i}{path.suffix}'))
                fig.savefig(fname, **_kw)
        if show:
            plt.show()
        for fig in self.figs:
            plt.close(fig)

    def get_figax(self) -> tuple[Figure, Axes]:
        """Return the figure and axes for a single-panel plot.

        Works for internally created and externally supplied figures.

        Returns:
            tuple[Figure, Axes]: (fig, ax).

        Raises:
            ValueError: If the plot has multiple panels.
        """
        if len(self.ax) > 1:
            raise ValueError(
                'PlotAstroData.get_figax() does not support channel maps.')

        return self.figs[0], self.ax[0]


ProfileLabels = str | list[str | None] | tuple[str | None, ...] | None
ProfileEllipses = RealSequence | list[RealSequence] | None


def _get_ylabel_profile(_kw: dict[str, Any], Tb: bool, flux: bool,
                        bunit: str | None) -> ProfileLabels:
    if 'ylabel' in _kw:
        return _kw.pop('ylabel')
    if Tb:
        return r'$T_b$ (K)'
    if flux:
        return 'Flux (Jy)'
    return bunit


def _prep_plotprofile(width: int | np.integer,
                      coords: str | list[str] | tuple[str, ...]
                      | np.ndarray | None,
                      xlist: RealSequence | None,
                      ylist: RealSequence | None,
                      ellipse: ProfileEllipses, ninterp: int | np.integer,
                      flux: bool, gaussfit: bool, _kw: dict[str, Any]
                      ) -> tuple[np.ndarray, np.ndarray,
                                 dict[str, list[np.ndarray]],
                                 PlotAxes2D, list[str | None]]:
    if coords is None:
        coords = []
    elif isinstance(coords, str):
        coords = [coords]
    xlist = [] if xlist is None else xlist
    ylist = [] if ylist is None else ylist
    _require_integer(width, 'width')
    _require_integer(ninterp, 'ninterp')
    Tb = _kw.get('Tb', False)
    f = kwargs2instance(AstroFrame, _kw)
    d = kwargs2instance(AstroData, _kw)
    f.read(d)
    d.binning([width, 1, 1])
    if len(coords):
        if d.center is None:
            raise ValueError('Sky-coordinate profiles require a known center.')
        xlist, ylist = coord2xy(coords, d.center) * 3600. * f.dist
        coords = []
    if f.swapxy:
        xlist, ylist = ylist, xlist
        if ellipse is not None:
            values = np.asarray(ellipse)
            ellipse = np.array(values,
                               dtype=np.result_type(values.dtype, np.float64),
                               copy=True)
            if ellipse.ndim not in (1, 2) or ellipse.shape[-1] != 3:
                raise ValueError('ellipse must contain [major, minor, pa] triples.')
            ellipse[..., 2] = 90 - ellipse[..., 2]
    v, prof, gfitres = d.profile(coords=coords, xlist=xlist, ylist=ylist,
                                 ellipse=ellipse, ninterp=ninterp,
                                 flux=flux, gaussfit=gaussfit)
    ylabel = _get_ylabel_profile(_kw, Tb, flux, d.bunit)
    if ylabel is None or isinstance(ylabel, str):
        ylabel = [ylabel] * len(prof)
    elif (not isinstance(ylabel, (list, tuple))
          or len(ylabel) != len(prof)
          or any(label is not None and not isinstance(label, str)
                 for label in ylabel)):
        raise ValueError('ylabel must be a string, None,'
                         + ' or one string/None per profile.')
    _kw.setdefault('xlim', [v.min(), v.max()])
    pa2 = kwargs2instance(PlotAxes2D, _kw)
    return v, prof, gfitres, pa2, list(ylabel)


def _validate_figax(fig: Figure | None, ax: Axes | None) -> None:
    if (fig is None) != (ax is None):
        raise ValueError('Provide fig and ax together, or omit both.')
    if fig is not None and (not isinstance(fig, Figure)
                            or not isinstance(ax, Axes)
                            or ax.figure is not fig):
        raise ValueError('fig and ax must be a Figure and its own Axes.')


def _set_figax_plotprofile(fig: Figure | None, ax: Axes | None,
                           nrows: int | np.integer, ncols: int | np.integer,
                           nprof: int) -> tuple[Figure, np.ndarray]:
    _validate_figax(fig, ax)
    _require_integer(ncols, 'ncols')
    _require_integer(nrows, 'nrows', minimum=0)
    nrows = (nprof + int(ncols) - 1) // int(ncols) if nrows == 0 else int(nrows)
    if nprof < 1 or nrows * ncols < nprof:
        raise ValueError('The subplot layout must have space'
                         + ' for every profile.')
    if ax is not None:
        if nprof != 1:
            raise ValueError('An external fig/ax pair supports'
                             + ' only one profile.')
        return fig, np.array([ax], dtype=object)
    fig = plt.figure(figsize=(6 * ncols, 3 * nrows))
    axes = np.empty(nprof, dtype=object)
    for i in range(nprof):
        axes[i] = fig.add_subplot(nrows, int(ncols), i + 1,
                                  sharex=None if i == 0 else axes[0])
    return fig, axes


def plotprofile(coords: str | list[str] | tuple[str, ...]
                | np.ndarray | None = None,
                xlist: RealSequence | None = None,
                ylist: RealSequence | None = None,
                ellipse: ProfileEllipses = None,
                ninterp: int | np.integer = 1,
                flux: bool = False, width: int | np.integer = 1,
                gaussfit: bool = False,
                gauss_kwargs: dict[str, Any] | None = None,
                title: list[str | dict[str, Any]]
                | tuple[str | dict[str, Any], ...] | None = None,
                text: list[dict[str, Any]] | tuple[dict[str, Any], ...]
                | None = None,
                nrows: int | np.integer = 0, ncols: int | np.integer = 1,
                fig: Figure | None = None, ax: Axes | None = None,
                getfigax: bool = False,
                savefig: dict[str, Any] | str | PathLike[str] | None = None,
                show: bool = False,
                **kwargs: Any) -> tuple[Figure, np.ndarray] | None:
    """Plot line profiles extracted from a spectral cube.

    Numeric offsets and aperture sizes use original data-axis units (arcsec,
    or au with distance scaling). Aperture position angles run from original
    +y toward +x. swapxy exchanges selection coordinates and maps aperture
    angles to 90 - pa; display flips do not change the selected physical region.
    Sky-coordinate offsets are converted to the distance-scaled spatial units.

    Keyword groups accepted in ``**kwargs``:
        AstroData: Data input and metadata for the cube, such as
        ``fitsimage``, ``data``, ``x``, ``y``, ``v``, ``beam``, ``Tb``,
        ``sigma``, ``center``, ``restfreq``, ``cfactor``, and ``bunit``.
        AstroFrame: Data trimming and coordinate-frame options, such as
        ``rmax``, ``center``, ``dist``, ``xoff``, ``yoff``, ``vsys``,
        ``vmin``, and ``vmax``.
        PlotAxes2D: Axis formatting, such as ``xlim``, ``ylim``,
        ``xlabel``, ``ylabel``, ``grid``, ``xscale``, ``yscale``, and
        tick options. ylabel accepts one string/None for all profiles or
        a list/tuple with one string/None per profile.
        Matplotlib: Additional keyword arguments are passed to
        ``matplotlib.axes.Axes.plot`` for the profile curve.

    Default keyword values:
        Profile curve: ``drawstyle='steps-mid'``, ``color='k'``,
        ``xlabel=r'Velocity (km s$^{-1}$)'``, and ``samexy=False``.
        User-supplied keyword arguments override these values.
        Gaussian overlay: ``drawstyle='default'`` and ``color='g'``.
        User-supplied values in ``gauss_kwargs`` override these values.

    Args:
        coords (str, list, tuple, or ndarray, optional): Sky coordinates.
            None means no sky-coordinate selection.
        xlist (list, tuple, or ndarray, optional): x offsets; None means empty.
        ylist (list, tuple, or ndarray, optional): y offsets; None means empty.
        ellipse (list, tuple, or ndarray, optional): One [major, minor, pa]
            triple or one triple per position, for averaging.
            Defaults to None.
        ninterp (int or np.integer, optional): Positive interpolation factor.
            Booleans, floats, and arrays are rejected. Defaults to 1.
        flux (bool, optional): y axis is flux density. Defaults to
            False.
        width (int or np.integer, optional): Positive rebinning step along v.
            Booleans, floats, and arrays are rejected. Defaults to 1.
        gaussfit (bool, optional): Fit the profiles. Defaults to False.
        gauss_kwargs (dict, optional): Gaussian overlay style for Axes.plot, not fitting options.
            None uses the default style.
        title (list or tuple, optional): One string or Axes.set_title keyword
            dictionary per profile. Inputs are not modified. Defaults to None.
        text (list or tuple, optional): One Axes.text keyword dictionary
            per profile (including x, y, s). Defaults to None.
        nrows (int or np.integer, optional): Subplot rows. 0 computes the
            required number from ncols and the profile count. Negative values,
            booleans, floats, and arrays are rejected. Defaults to 0.
        ncols (int or np.integer, optional): Positive subplot column count. Defaults to 1.
        fig (Figure, optional): External figure; supply together with ax.
            Only one profile/slice may use the pair. Defaults to None.
        ax (Axes, optional): Axes owned by fig, reused without adding a
            subplot. Supply together with fig. Defaults to None.
        getfigax (bool, optional): Defaults to False.
        savefig (dict, str, or os.PathLike, optional): Passed to ``close_figure``.
            Existing files may be overwritten, and the figure is closed
            after saving/showing. Defaults to None.
        show (bool, optional): True means doing plt.show(). Defaults to
            False.

    Returns:
        tuple[Figure, np.ndarray] or None: With getfigax=True, return the
        figure and 1D object array of Axes without saving, showing, or closing.
        Otherwise save/show as requested, close the figure, and return None.
    """
    _validate_figax(fig, ax)
    _kw = {'drawstyle': 'steps-mid', 'color': 'k',
           'xlabel': r'Velocity (km s$^{-1}$)', 'samexy': False}
    _kw.update(kwargs)
    _kwgauss = {'drawstyle': 'default', 'color': 'g'}
    _kwgauss.update(gauss_kwargs or {})
    v, prof, gfitres, pa2, ylabel \
        = _prep_plotprofile(width, coords, xlist, ylist, ellipse,
                            ninterp, flux, gaussfit, _kw)
    nprof = len(prof)
    for name, values, types in (('title', title, (str, dict)),
                                ('text', text, (dict,))):
        if values is not None and (not isinstance(values, (list, tuple))
                                   or len(values) != nprof
                                   or any(not isinstance(v, types)
                                          for v in values)):
            raise ValueError(f'{name} must contain one valid entry per profile.')
    set_rcparams(20, 'w')
    fig, ax = _set_figax_plotprofile(fig, ax, nrows, ncols, nprof)
    for i in range(nprof):
        if gaussfit:
            ax[i].plot(v, gaussian1d(v, *gfitres['best'][i]), **_kwgauss)
        ax[i].plot(v, prof[i], **_kw)
        ax[i].hlines([0], v.min(), v.max(), linestyle='dashed', color='k')
        ax[i].set_ylabel(ylabel[i])
        pa2.set_xyaxes(ax[i])
        if text is not None:
            ax[i].text(**text[i])
        if title is not None:
            title_kwargs = ({'label': title[i]} if isinstance(title[i], str)
                            else title[i])
            ax[i].set_title(**title_kwargs)
        if i <= nprof - ncols - 1:
            plt.setp(ax[i].get_xticklabels(), visible=False)
    if getfigax:
        return fig, ax

    close_figure(fig, savefig, show)


def plotslice(length: RealScalar, dx: RealScalar | None = None,
              pa: RealScalar = 0,
              txtfile: str | PathLike[str] | None = None,
              fig: Figure | None = None, ax: Axes | None = None,
              getfigax: bool = False,
              savefig: str | PathLike[str] | dict[str, Any] | None = None,
              show: bool = False,
              **kwargs: Any) -> tuple[Figure, Axes] | None:
    """Plot a one-dimensional spatial slice through a 2D map.

    Keyword groups accepted in ``**kwargs``:
        AstroData: Data input and metadata for the 2D map, such as
        ``fitsimage``, ``data``, ``x``, ``y``, ``beam``, ``Tb``,
        ``sigma``, ``center``, ``restfreq``, ``cfactor``, and ``bunit``.
        AstroFrame: Data trimming and coordinate-frame options, such as
        ``rmax``, ``center``, ``dist``, ``xoff``, ``yoff``, ``xflip``,
        ``yflip``, and ``swapxy``.
        PlotAxes2D: Axis formatting, such as ``xlim``, ``ylim``,
        ``xlabel``, ``ylabel``, ``grid``, ``xscale``, ``yscale``, and
        tick options.
        Matplotlib: Additional keyword arguments are passed to
        ``matplotlib.axes.Axes.plot`` for the slice curve.

    Default keyword values:
        Slice curve: ``linestyle='-'`` and ``marker='o'``. User-supplied
        keyword arguments override these values.
        Axis setup: ``rmax=length / 2`` and ``samexy=False`` are set
        internally. ``xlabel``, ``ylabel``, and ``xlim`` are filled from
        the slice geometry if they are omitted.

    Args:
        length (float): Required finite positive slice length.
        dx (float, optional): Grid increment. Defaults to None.
        pa (float, optional): Degrees from original data +y toward +x.
            swapxy exchanges direction components; display flips do not change
            the extracted physical direction. Defaults to 0.
        txtfile (str or os.PathLike, optional): File name for ``numpy.savetxt``.
            Existing files with the same name are overwritten. Defaults
            to None.
        fig (Figure, optional): External figure; supply together with ax.
            Only one profile/slice may use the pair. Defaults to None.
        ax (Axes, optional): Axes owned by fig, reused without adding a
            subplot. Supply together with fig. Defaults to None.
        getfigax (bool, optional): Defaults to False.
        savefig (dict, str, or os.PathLike, optional): Passed to ``close_figure``.
            Existing files may be overwritten, and the figure is closed
            after saving/showing. Defaults to None.
        show (bool, optional): True means doing plt.show(). Defaults to
            False.

    Returns:
        tuple[Figure, Axes] or None: With getfigax=True, return without
        figure saving/showing/closing; requested txtfile export still occurs.
        Otherwise save/show as requested, close the figure, and return None.
    """
    _validate_figax(fig, ax)
    _kw = {'linestyle': '-', 'marker': 'o'}
    _kw.update(kwargs)
    _kw['rmax'] = length / 2
    Tb = _kw.get('Tb', False)
    f = kwargs2instance(AstroFrame, _kw)
    d = kwargs2instance(AstroData, _kw)
    f.read(d)
    if np.ndim(d.data) > 2:
        raise ValueError('plotslice() supports only a 2D map.')

    r, z = d.slice(length=length, pa=90 - pa if f.swapxy else pa, dx=dx)
    xunit = 'arcsec' if f.dist == 1 else 'au'
    yunit = 'K' if Tb else d.bunit
    yquantity = r'$T_b$' if Tb else 'intensity'

    if txtfile is not None:
        np.savetxt(txtfile, np.c_[r, z],
                   header=f'x ({xunit}), {yquantity} ({yunit}); '
                   + f'positive x is pa={pa:.2f} deg.')
    if 'xlabel' not in _kw:
        _kw['xlabel'] = f'Offset ({xunit})'
    if 'ylabel' not in _kw:
        _kw['ylabel'] = f'{yquantity} ({yunit})'
    if 'xlim' not in _kw:
        _kw['xlim'] = [r.min(), r.max()]
    _kw['samexy'] = False
    set_rcparams()
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(1, 1, 1)
    pa2 = kwargs2instance(PlotAxes2D, _kw)
    ax.plot(r, z, **_kw)
    if d.sigma is not None:
        ax.plot(r, r * 0 + 3 * d.sigma, 'k--')
    pa2.set_xyaxes(ax)
    if getfigax:
        return fig, ax

    close_figure(fig, savefig, show)


class _Plot3DResult(TypedDict):
    data: list[dict[str, Any]]
    layout: 'Layout'


def _surface_levels(levels: RealSequence) -> np.ndarray:
    values = np.asarray(levels)
    if (values.ndim != 1 or values.size == 0
            or values.dtype.kind not in 'iuf'
            or not np.all(np.isfinite(values))
            or np.unique(values).size != values.size):
        raise ValueError('Isosurface levels must be a nonempty finite,'
                         + ' distinct real sequence.')
    return values


def _surface_opacity(alpha: RealScalar) -> None:
    if not _is_finite_real_scalar(alpha) or not 0 <= alpha <= 1:
        raise ValueError('Isosurface alpha must be finite and between 0 and 1.')


def _surface_grid(grid: np.ndarray | None, size: int, name: str) -> None:
    """Require a matching 1D grid with uniform spacing relative to its first interval."""
    if grid is None or np.ndim(grid) != 1 or len(grid) != size or size < 2:
        raise ValueError('plot3d requires at least two matching'
                         + f' {name} coordinates.')
    steps = np.diff(grid)
    if (not np.all(np.isfinite(grid)) or steps[0] == 0
            or not _is_uniform_spacing(steps, steps[0])):
        raise ValueError('plot3d requires a finite, uniformly spaced'
                         + f' {name} grid.')


def _surface_mesh(measure: ModuleType, volume: np.ndarray,
                  threshold: RealScalar
                  ) -> tuple[np.ndarray, np.ndarray] | None:
    # Boundary levels and constant volumes do not define an interior surface.
    if not np.min(volume) < threshold < np.max(volume):
        return None
    try:
        vertices, simplices, _, _ = measure.marching_cubes(volume, threshold)
    except RuntimeError as exc:
        if 'No surface found' not in str(exc):
            raise
        return None
    return vertices, simplices


def _plotly_values(value: Any, *, exporting: bool = False) -> Any:
    """Convert generated Plotly payloads only at return/export boundaries.

    Public results preserve arrays and extended precision. HTML export uses
    a separate payload with longdouble values narrowed to float64, which
    Plotly's JSON encoder supports. Never apply this to generic user data.
    """
    if isinstance(value, dict):
        return {k: _plotly_values(v, exporting=exporting)
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plotly_values(v, exporting=exporting) for v in value]
    if isinstance(value, np.ndarray):
        if exporting and value.dtype == np.dtype(np.longdouble):
            return value.astype(float)
        return value
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.floating):
        return float(value) if exporting else _normalize_float(value)
    return value


def _plot_on_wall(d: AstroData, x: np.ndarray, y: np.ndarray, v: np.ndarray,
                  measure: ModuleType, datalist: list[dict[str, Any]],
                  sign: Literal[-1, 1], axis: Literal[0, 1, 2],
                  **kwargs: Any) -> None:
    dx, dy, dv = x[1] - x[0], y[1] - y[0], v[1] - v[0]
    s, ds = [x, y, v], [dx, dy, dv]
    if kwargs == {}:
        return

    match axis:
        case 2:
            shape = np.shape(d.data[:, :, 0])
        case 1:
            shape = np.shape(d.data[:, 0, :])
        case 0:
            shape = np.shape(d.data[0, :, :])
    if 'data' not in kwargs:
        raise ValueError('Each wall map requires a data array.')
    if np.shape(kwargs['data']) != shape:
        raise ValueError(
            'The shape of the 2D wall data must match the corresponding '
            f'3D data plane; got {np.shape(kwargs["data"])} and {shape}.')

    _kw = {'levels': [3, 6, 12, 24, 48, 96, 192, 384],
           'sigma': 'hist', 'cmap': 'Jet', 'alpha': 0.3}
    _kw.update(kwargs)
    volume = np.array(_kw['data'], copy=True)
    if volume.dtype.kind not in 'iuf':
        raise ValueError('Wall data must be real numeric arrays.')
    levels = _surface_levels(_kw['levels'])
    cmap = _kw['cmap']
    alpha = _kw['alpha']
    _surface_opacity(alpha)
    sigma = estimate_rms(data=volume, sigma=_kw['sigma'])
    _positive_noise(sigma, 'Wall sigma')
    volume[~np.isfinite(volume)] = 0
    a = int(sign == -1)
    b = int(sign == 1)
    volume = np.moveaxis([volume * a, volume * b], 0, axis)
    if d.dx < 0:
        volume = volume[:, :, ::-1]
    if d.dy < 0:
        volume = volume[:, ::-1, :]
    if d.dv < 0:
        volume = volume[::-1, :, :]
    for lev in levels:
        surface = _surface_mesh(measure, volume, lev * sigma)
        if surface is None:
            continue
        vertices, simplices = surface
        Xg, Yg, Zg = [t[0] + i * dt for t, i, dt
                      in zip(s, vertices.T[::-1], ds)]
        match axis:
            case 2:
                Xg = Xg * 0 + (x[-1] if sign == 1 else x[0])
            case 1:
                Yg = Yg * 0 + (y[-1] if sign == 1 else y[0])
            case 0:
                Zg = Zg * 0 + (v[-1] if sign == 1 else v[0])
        i, j, k = simplices.T
        mesh2d = dict(type='mesh3d', x=Xg, y=Yg, z=Zg,
                      i=i, j=j, k=k,
                      intensity=Zg * 0 + lev,
                      colorscale=cmap, reversescale=False,
                      cmin=np.min(levels),
                      cmax=np.max(levels),
                      opacity=alpha, name='', showscale=False)
        datalist.append(mesh2d)


@dataclass(frozen=True)
class _SurfaceCube:
    """Private finite plotting volume with ascending coordinates and spacings."""
    volume: np.ndarray
    x: np.ndarray
    y: np.ndarray
    v: np.ndarray
    dx: RealScalar
    dy: RealScalar
    dv: RealScalar


def _prepare_surface_cube(d: AstroData) -> _SurfaceCube:
    """Validate and orient a private volume without changing processed data."""
    if not isinstance(d.data, np.ndarray) or d.data.ndim != 3 or d.data.dtype.kind not in 'iuf':
        raise ValueError('plot3d() requires one real 3D spectral cube'
                         + ' with at least two samples per axis.')
    for grid, size, name in zip((d.v, d.y, d.x), d.data.shape, ('v', 'y', 'x')):
        _surface_grid(grid, size, name)
    _positive_noise(d.sigma, 'sigma')
    volume = np.array(d.data, copy=True)
    x, y, v = d.x, d.y, d.v
    dx, dy, dv = d.dx, d.dy, d.dv
    volume[~np.isfinite(volume)] = 0
    if dx < 0:
        x, dx, volume = x[::-1], -dx, volume[:, :, ::-1]
    if dy < 0:
        y, dy, volume = y[::-1], -dy, volume[:, ::-1, :]
    if dv < 0:
        v, dv, volume = v[::-1], -dv, volume[::-1, :, :]
    return _SurfaceCube(volume, x, y, v, dx, dy, dv)


def _surface_traces(measure: ModuleType, cube: _SurfaceCube,
                    levels: np.ndarray, sigma: RealScalar,
                    cmap: str, alpha: RealScalar) -> list[dict[str, Any]]:
    """Build paired isosurface meshes and triangle-edge traces in data units."""
    volume = cube.volume
    x, y, v = cube.x, cube.y, cube.v
    dx, dy, dv = cube.dx, cube.dy, cube.dv
    s, ds = [x, y, v], [dx, dy, dv]
    data = []
    for lev in levels:
        surface = _surface_mesh(measure, volume, lev * sigma)
        if surface is None:
            continue
        vertices, simplices = surface
        Xg, Yg, Zg = [t[0] + i * dt for t, i, dt
                      in zip(s, vertices.T[::-1], ds)]
        i, j, k = simplices.T
        mesh = dict(type='mesh3d', x=Xg, y=Yg, z=Zg, i=i, j=j, k=k,
                    intensity=Zg * 0 + lev,
                    colorscale=cmap, reversescale=False,
                    cmin=np.min(levels), cmax=np.max(levels),
                    opacity=alpha, name='', showscale=False)
        data.append(mesh)
        Xe, Ye, Ze = [], [], []
        for t in vertices[simplices]:
            Xe += [x[0] + dx * t[k % 3][2] for k in range(4)] + [None]
            Ye += [y[0] + dy * t[k % 3][1] for k in range(4)] + [None]
            Ze += [v[0] + dv * t[k % 3][0] for k in range(4)] + [None]
        lines = dict(type='scatter3d', x=Xe, y=Ye, z=Ze,
                     mode='lines', opacity=0.04, visible=True,
                     name='', line=dict(color='rgb(0,0,0)', width=1))
        data.append(lines)

    return data


def _export_plot3d(go: ModuleType, data: list[dict[str, Any]],
                   layout: dict[str, Any], outname: str | PathLike[str],
                   show: bool) -> None:
    """Write HTML using a separate payload that narrows extended precision."""
    fig = go.Figure(data=_plotly_values(data, exporting=True),
                    layout=_plotly_values(layout, exporting=True))
    path = Path(outname)
    if path.suffix.lower() != '.html':
        path = path.with_name(path.name + '.html')
    fig.write_html(file=str(path), auto_open=show, auto_play=False)


def plot3d(levels: RealSequence = (3, 6, 12),
           cmap: str = 'jet', alpha: RealScalar = 0.08,
           xlabel: str = 'R.A. (arcsec)',
           ylabel: str = 'Dec. (arcsec)',
           vlabel: str = 'Velocity (km/s)',
           xskip: int | np.integer = 1, yskip: int | np.integer = 1,
           eye_p: RealScalar = 0, eye_i: RealScalar = 180,
           xplus: dict[str, Any] | None = None,
           xminus: dict[str, Any] | None = None,
           yplus: dict[str, Any] | None = None,
           yminus: dict[str, Any] | None = None,
           vplus: dict[str, Any] | None = None,
           vminus: dict[str, Any] | None = None,
           outname: str | PathLike[str] = 'plot3d', show: bool = False,
           return_data_layout: bool = False,
           **kwargs: Any) -> _Plot3DResult | None:
    """Create an interactive Plotly 3D isosurface visualization of a
    spectral cube.

    Requires dist=1, swapxy=False, and one real cube with at least two
    samples per uniformly spaced axis (rtol=1e-5). All coordinate axes are
    ascending to retain a right-handed system; xflip/yflip do not reverse
    the 3D display. A finite positive scalar sigma is required.
    Nonfinite voxels are replaced by zero in a private plotting copy.
    Levels at or outside the volume range, or with no surface, are skipped.
    Plotly and scikit-image are imported only when this function is called.
    Returned computed scalars are Python scalars, except extended-precision
    NumPy floats; mesh arrays retain their dtypes. HTML export converts
    extended-precision values to float64 in a separate serialization payload.

    Keyword groups accepted in ``**kwargs``:
        AstroData: Data input and metadata for the cube, such as
        ``fitsimage``, ``data``, ``x``, ``y``, ``v``, ``beam``, ``Tb``,
        ``sigma``, ``center``, ``restfreq``, ``cfactor``, and ``bunit``.
        AstroFrame: Data trimming and coordinate-frame options, such as
        ``rmax``, ``center``, ``dist``, ``xoff``, ``yoff``, ``vsys``,
        ``vmin``, ``vmax``, ``xflip``, ``yflip``, and ``swapxy``.

    Default keyword values:
        Wall maps: The dictionaries ``xplus``, ``xminus``, ``yplus``,
        ``yminus``, ``vplus``, and ``vminus`` use
        ``levels=[3, 6, 12, 24, 48, 96, 192, 384]``, ``sigma='hist'``,
        ``cmap='Jet'``, and ``alpha=0.3`` when those keys are omitted.

    Args:
        levels (list, tuple, or ndarray, optional): Finite distinct contour
            levels in units of sigma. Defaults to [3,6,12].
        cmap (str, optional): Color map name. Defaults to 'jet'.
        alpha (float, optional): Finite real scalar opacity in [0, 1].
            Booleans, complex values, and arrays are rejected. Defaults to 0.08.
        xlabel (str, optional): Defaults to 'R.A. (arcsec)'.
        ylabel (str, optional): Defaults to 'Dec. (arcsec)'.
        vlabel (str, optional): Defaults to 'Velocity (km/s)'.
        xskip (int, optional): Number of pixel to skip. Defaults to 1.
        yskip (int, optional): Number of pixel to skip. Defaults to 1.
        eye_p (float, optional): Azimuthal angle of camera. Defaults to
            0.
        eye_i (float, optional): Inclination angle of camera. Defaults
            to 180.
        xplus (dict, optional): 2D data to be plotted on the y-v plane
            at the positive edge of x. This dictionary must have a key
            of data and can have keys of levels, sigma, cmap, and alpha.
            Shape must match the processed cube plane: (nv, ny) for x walls,
            (nv, nx) for y walls, (ny, nx) for v walls, after trimming and
            sampling, in the corresponding processed coordinate order.
            Wall sigma must be finite and positive; alpha must be in [0, 1].
            Nonfinite wall pixels are replaced by zero in a private copy.
            None or {} omits the wall.
        xminus (dict, optional): See xplus. Defaults to None.
        yplus (dict, optional): See xplus. Defaults to None.
        yminus (dict, optional): See xplus. Defaults to None.
        vplus (dict, optional): See xplus. Defaults to None.
        vminus (dict, optional): See xplus. Defaults to None.
        outname (str or os.PathLike, optional): Output HTML file name,
            with or without '.html'. Existing files with the same name are
            overwritten by Plotly. Defaults to 'plot3d'.
        show (bool, optional): Open the saved HTML in the default browser.
            Defaults to False. Ignored when return_data_layout=True.
        return_data_layout (bool, optional): Whether to return data and
            layout for plotly.graph_objs.Figure. Defaults to False.

    Returns:
        _Plot3DResult: {'data': trace dictionaries, 'layout': Plotly Layout}, if
        return_data_layout=True (no file is written or opened). Otherwise, no return.
    """
    levels = _surface_levels(levels)
    _surface_opacity(alpha)
    f = kwargs2instance(AstroFrame, kwargs)
    if f.dist != 1:
        raise ValueError('plot3d() requires dist=1;'
                         + ' distance-scaled plots are not supported.')
    if f.swapxy:
        raise ValueError('plot3d() does not support swapxy=True.')
    if f.pv:
        raise ValueError('plot3d() requires a spectral cube, not a PV diagram.')
    try:
        import plotly.graph_objs as go
        from skimage import measure
    except ImportError as exc:
        raise ImportError('plot3d() requires plotly and scikit-image.') from exc
    d = kwargs2instance(AstroData, kwargs)
    f.read(d, xskip, yskip)
    cube = _prepare_surface_cube(d)
    x, y, v = cube.x, cube.y, cube.v
    deg = np.radians(1)
    xeye = -np.sin(eye_i * deg) * np.sin(eye_p * deg)
    yeye = -np.sin(eye_i * deg) * np.cos(eye_p * deg)
    zeye = np.cos(eye_i * deg)
    margin = dict(l=0, r=0, b=0, t=0)
    camera = dict(eye=dict(x=xeye, y=yeye, z=zeye), up=dict(x=0, y=1, z=0))
    xaxis = dict(range=[x[0], x[-1]], title=xlabel)
    yaxis = dict(range=[y[0], y[-1]], title=ylabel)
    zaxis = dict(range=[v[0], v[-1]], title=vlabel)
    scene = dict(aspectmode='cube', camera=camera,
                 xaxis=xaxis, yaxis=yaxis, zaxis=zaxis)
    layout = dict(margin=margin, scene=scene, showlegend=False)

    data = _surface_traces(measure, cube, levels, d.sigma, cmap, alpha)

    klist = [xplus, xminus, yplus, yminus, vplus, vminus]
    slist = [1, -1, 1, -1, 1, -1]
    alist = [2, 2, 1, 1, 0, 0]
    for kw, sign, axis in zip(klist, slist, alist):
        _plot_on_wall(d, x, y, v, measure, data, sign, axis, **(kw or {}))

    if return_data_layout:
        return {'data': _plotly_values(data),
                'layout': go.Layout(**_plotly_values(layout))}
    else:
        _export_plot3d(go, data, layout, outname, show)
