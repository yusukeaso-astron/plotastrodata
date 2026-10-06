import matplotlib.pyplot as plt
import numbers
import numpy as np
import warnings
from scipy.special import erf
from typing import Any, Callable

from plotastrodata._type_utils import _normalize_float
from plotastrodata.fitting_utils import EmceeCorner
from plotastrodata.other_utils import close_figure


def _uniform_bin_width(edges: np.ndarray) -> float | np.longdouble:
    """Validate equal-width edges with relative tolerance 1e-5."""
    if (edges.ndim != 1 or len(edges) < 2
            or not np.all(np.isfinite(edges))):
        raise ValueError('Histogram edges must be finite'
                         + ' and strictly increasing.')
    widths = np.diff(edges)
    if not np.all(np.isfinite(widths)) or np.any(widths <= 0):
        raise ValueError('Histogram edges must be finite'
                         + ' and strictly increasing.')
    width = np.mean(widths)
    if not np.allclose(widths, width, rtol=1e-5, atol=0):
        raise ValueError('Only equal-width bins are supported;'
                         + ' use an integer bin count, bins="auto",'
                         + ' or equally spaced edges'
                         + ' (for example np.linspace).')
    return _normalize_float(width)


def normalize(range: tuple[float, float] = (-3.5, 3.5),
              bins: int = 100, *,
              edges: np.ndarray | list[float] | None = None) -> Callable:
    """Decorator factory to normalize a function over the given range.

    Explicit edges override range and bins. Midpoint densities are
    normalized using a single shared bin width. Edges must be equally
    spaced within relative tolerance 1e-5 (zero absolute tolerance) to
    accommodate rounding of float32 histogram edges.
    If all midpoint values underflow to zero, mass is placed in the bin
    containing args[1] (the model mean); a mean outside the edges produces zeros.

    Ordinary scalar outputs are Python floats. Extended-precision scalars
    and arrays, including zero-dimensional arrays, are preserved.
    """
    if edges is None:
        if not isinstance(bins, numbers.Integral) or bins < 1:
            raise ValueError('bins must be a positive integer.')
        edges = np.linspace(*range, bins + 1)
    else:
        edges = np.array(edges, copy=True)
    width = _uniform_bin_width(edges)
    h = (edges[:-1] + edges[1:]) / 2

    def decorator(f: Callable) -> Callable:
        def wrapper(x: float | np.floating | np.ndarray, *args: Any
                    ) -> float | np.longdouble | np.ndarray:
            area = np.sum(f(h, *args)) * width
            if area == 0:
                # The model's second parameter is its mean. Match histogram
                # edge conventions, including the closed final bin.
                mean = args[1]
                i = min(int(np.searchsorted(edges, mean, side='right')) - 1,
                        len(edges) - 2)
                if i < 0 or mean > edges[-1]:
                    return np.zeros_like(x)
                xarr = np.asarray(x)
                inside = ((xarr >= edges[i])
                          & ((xarr <= edges[i + 1]) if i == len(edges) - 2
                             else (xarr < edges[i + 1])))
                p = np.where(inside, 1 / width, 0)
            else:
                p = f(x, *args) / area
            return _normalize_float(p)

        return wrapper
    return decorator


def gauss(x: float | np.floating | np.ndarray,
          s: float | np.floating, m: float | np.floating
          ) -> float | np.longdouble | np.ndarray:
    """Probability density of Gaussian noise.

    Args:
        x (float or np.floating or np.ndarray): Intensity. 
            The variable of the probability density.
        s (float): Standard deviation of the Gaussian noise.
        m (float): Mean of the Gaussian noise.

    Returns:
        float or np.longdouble or np.ndarray: Probability density.
        Ordinary scalar results are Python floats; extended-precision
        scalars and arrays retain their types.
    """
    x1 = (x - m) / np.sqrt(2) / s
    p = np.exp(-x1**2)
    p = p / (np.sqrt(2 * np.pi) * s)
    return _normalize_float(p)


def gauss_pbcor(x: float | np.floating | np.ndarray,
                s: float | np.floating, m: float | np.floating,
                R: float | np.floating
                ) -> float | np.longdouble | np.ndarray:
    """Probability density of Gaussian noise after primary-beam
    correction.

    Args:
        x (float or np.floating or np.ndarray): Intensity. 
            The variable of the probability density.
        s (float): Standard deviation of the Gaussian noise.
        m (float): Mean of the Gaussian noise.
        R (float): The maximum radius scaled by the FWHM of the primary beam.

    Returns:
        float or np.longdouble or np.ndarray: Probability density.
        Ordinary scalar results are Python floats; extended-precision
        scalars and arrays retain their types.
    """
    x1 = (x - m) / np.sqrt(2) / s
    x0 = (x * 2**(-R**2) - m) / np.sqrt(2) / s
    p = erf(x1) - erf(x0)
    # Odd bin counts may have a center at zero. Use the analytic limit.
    with np.errstate(divide='ignore', invalid='ignore'):
        p = p / (2 * np.log(2) * x * R**2)
    limit = (np.exp(-m**2 / (2 * s**2)) * (1 - 2**(-R**2))
             / (np.sqrt(2 * np.pi) * s * np.log(2) * R**2))
    if np.ndim(p) == 0:
        p = limit if x == 0 else p
    else:
        p = np.where(np.asarray(x) == 0, limit, p)
    return _normalize_float(p)


def select_noise(data: np.ndarray, sigma: str) -> np.ndarray:
    """Select data pixels to be used for noise estimation.

    Args:
        data (np.ndarray): Original data array.
        sigma (str): Selection methods. Multiple options are possible.
            'edge', 'out', 'neg', or 'iter'.

    Returns:
        np.ndarray: 1D array that includes only the selected pixels.
    """
    n = np.array(data).copy()
    if 'edge' in sigma:
        if np.ndim(n) <= 2:
            print('\'edge\' is ignored because ndim <= 2.')
        else:
            n = n[::len(n) - 1]
    if 'out' in sigma and 'pbcor' in sigma:
        print('\'out\' is ignored because of \'pbcor\'.')
    elif 'out' in sigma:
        nx = np.shape(n)[-1]
        ny = np.shape(n)[-2]
        ntmp = np.moveaxis(n, [-2, -1], [0, 1])
        ntmp[ny // 5: ny * 4 // 5, nx // 5: nx * 4 // 5] = np.nan
        if np.all(np.isnan(ntmp)):
            print('\'out\' is ignored because'
                  + ' the outer region is filled with nan.')
        else:
            n = ntmp
    n = n[~np.isnan(n)]
    if 'neg' in sigma:
        n = n[n < 0]
        n = np.r_[n, -n]
    if 'iter' in sigma:
        for _ in range(5):
            n = n[np.abs(n - np.mean(n)) < 3.5 * np.std(n)]
    return n.ravel()


class Noise:
    """This class holds the data selected as noise, histogram, and
    best-fit function.

    The following methods are acceptable for data selection. Multiple
    options are possible.
    'edge': use data[0] and data[-1].
    'out': exclude inner 60% about axes=-2 and -1.
    'neg': use only negative values.
    'iter': exclude outliers.
    The following methods are acceptable for noise estimation. Only
    single option is possible.
    'med': calculate rms from the median of data^2 assuming Gaussian.
    'hist': fit histogram with Gaussian.
    'hist-pbcor': fit histogram with PB-corrected Gaussian.
    '(no string)': calculate the mean and standard deviation.

    Args:
        data (np.ndarray): Original data array.
        sigma (str): Methods above, like 'edge,neg,hist-pbcor'.
    """
    def __init__(self, data: np.ndarray, sigma: str) -> None:
        self.data = select_noise(data, sigma)
        if self.data.size < 2:
            raise ValueError('Noise estimation requires at least'
                             + ' two selected finite pixels.')
        self.sigma = sigma
        self.m0 = _normalize_float(np.mean(self.data))
        self.s0 = _normalize_float(np.std(self.data))
        if not np.isfinite(self.s0):
            raise ValueError('Noise estimation requires selected pixels'
                             + ' with finite values.')

    def gen_histogram(self, **kwargs: Any) -> None:
        """Generate a pair of histogram and bins using numpy.histogram.

        The data values are shifted and scaled by the mean and standard
        deviation, respectively, to generate the histogram. The mean and
        standard deviation are stored as self.m0 and self.s0,
        respectively. Ordinary scalar statistics use Python floats;
        extended-precision scalar statistics are preserved.

        Default keyword values:
            numpy.histogram: ``bins=100``, ``range=(-3.5, 3.5)``, and
            ``density=True``. User-supplied keyword arguments override
            these values. Weighted histograms are unsupported. Actual
            edges, a shared bin width, and unweighted counts are retained for fitting.
            density=False plots counts instead of densities. Uneven edges
            raise ValueError; equal widths are checked with rtol=1e-5, atol=0.
        """
        if self.s0 == 0:
            raise ValueError('Histogram noise estimation requires'
                             + ' selected pixels with nonzero variance.')
        _kw = {'bins': 100, 'range': (-3.5, 3.5), 'density': True}
        _kw.update(kwargs)
        if _kw.get('weights') is not None:
            raise ValueError('Noise histograms currently support'
                             + ' unweighted data only.')
        n = (self.data - self.m0) / self.s0
        hist, edges = np.histogram(n, **_kw)
        width = _uniform_bin_width(edges)
        counts, _ = np.histogram(n, bins=edges)
        if np.sum(counts) == 0:
            raise ValueError('Histogram range contains no selected data.')
        self.bins = len(edges) - 1
        self.range = (edges[0], edges[-1])
        self.edges = edges
        self.bin_width = width
        self.counts = counts
        self.density = bool(_kw['density'])
        self.hist = hist
        self.hbin = (edges[:-1] + edges[1:]) / 2
        # A regenerated histogram invalidates its previous fit.
        for name in ('model', 'popt', 'mean', 'std'):
            if hasattr(self, name):
                delattr(self, name)

    def fit_histogram(self, **kwargs: Any) -> None:
        """Fit the noise histogram with
        plotastrodata.fitting_utils.EmceeCorner.

        Supports unweighted histograms with integer bin counts, automatic
        selectors, or explicit equally spaced edges. The likelihood uses raw counts
        and midpoint densities integrated with a shared bin width.
        mean and std are Python floats except for extended-precision results.

        Default keyword values:
            EmceeCorner.fit: ``nwalkersperdim=4``, ``nsteps=200``, and
            ``nburnin=0``. User-supplied keyword arguments override
            these values.
        """
        _kw = {'nwalkersperdim': 4, 'nsteps': 200, 'nburnin': 0}
        _kw.update(kwargs)
        if not hasattr(self, 'hist'):
            self.gen_histogram()
            print('Noise.gen_histogram() was done with default arguments.')
        f = gauss_pbcor if 'pbcor' in self.sigma else gauss
        model = normalize(edges=self.edges)(f)
        bounds = [[0.1, 2], [-2, 2]]
        if 'pbcor' in self.sigma:
            bounds.append([0.1, 2])
        # curve_fit does not work for this fitting.
        # For binned data, sigma^2 is the expected number of data in each bin.

        def logl(p: np.ndarray) -> float | np.longdouble:
            numobs = self.counts
            numexp = model(self.hbin, *p) * self.bin_width * np.sum(numobs)
            chi2 = np.sum((numobs - numexp)**2 / numexp.clip(1, None))
            return _normalize_float(-0.5 * chi2)

        fitter = EmceeCorner(bounds=bounds, logl=logl)
        fitter.fit(**_kw)
        self.popt = fitter.popt
        self.mean = _normalize_float(self.popt[1] * self.s0 + self.m0)
        self.std = _normalize_float(self.popt[0] * self.s0)
        self.model = model(self.hbin, *self.popt)
        if not self.density:
            self.model = self.model * self.bin_width * np.sum(self.counts)

    def plot_histogram(self, savefig: dict | str | None = None,
                       show: bool = False) -> None:
        """Make a simple figure of the histogram and model.

        Args:
            savefig (dict or str, optional): Passed to ``close_figure``.
                Existing files may be overwritten, and the figure is
                closed after saving/showing. Defaults to None.
            show (bool, optional): True means doing plt.show(). Defaults
                to False.
        """
        if not hasattr(self, 'model'):
            self.fit_histogram()
            print('Noise.fit_histogram() was done with default arguments.')
        fig, ax = plt.subplots()
        ax.plot(self.hbin, self.hist, drawstyle='steps-mid')
        ax.plot(self.hbin, self.model, '-')
        ax.set_xlabel('(noise - m0) / s0')
        ax.set_ylabel('Probability density' if self.density else 'Count')
        close_figure(fig, savefig, show)


def estimate_rms(data: np.ndarray,
                 sigma: float | numbers.Number | str | None = 'hist'
                 ) -> float | np.longdouble | numbers.Number | None:
    """Estimate a noise level of a data array.
    Numeric sigma values and None are returned unchanged.

    Args:
        data (np.ndarray): Data array whose noise is estimated.
        sigma (float or numbers.Number or str or None): A numeric value
            to return unchanged, None, or Noise selection/estimation methods
            such as 'edge,neg,hist-pbcor'. Defaults to 'hist'.

    Returns:
        float or np.longdouble or numbers.Number or None: Computed noise
        is a Python float, with extended-precision scalars preserved.
        Numeric inputs and None pass through unchanged.
    """
    if sigma is None or isinstance(sigma, numbers.Number):
        return sigma

    if np.ndim(np.squeeze(data)) == 0:
        raise ValueError('sigma cannot be estimated from only one pixel.')

    n = Noise(data, sigma)
    if 'hist' in sigma:
        n.gen_histogram()
        n.fit_histogram()
        ave = n.mean
        noise = n.std
    elif 'med' in sigma:
        ave = 0
        noise = np.sqrt(np.median(n.data**2) / 0.454936)
    else:
        ave = n.m0
        noise = n.s0
    if np.abs(ave) > 0.2 * noise:
        s = '|mean| > 0.2sigma.'
        warnings.warn(s, UserWarning)
    return _normalize_float(noise)
