import matplotlib.pyplot as plt
import numbers
import numpy as np
import warnings
from scipy.special import erf

from plotastrodata.fitting_utils import EmceeCorner
from plotastrodata.other_utils import close_figure


def normalize(range: tuple = (-3.5, 3.5), bins: int = 100):
    """Decorator factory to normalize a function over the given range."""
    def decorator(f):
        h = np.linspace(*range, bins + 1)
        h = (h[1:] + h[:-1]) / 2
        dh = h[1] - h[0]

        def wrapper(x, *args):
            area = np.sum(f(h, *args)) * dh
            if area == 0:
                p = np.where(np.abs(x - args[1]) < dh / 2, 1 / dh, 0)
            else:
                p = f(x, *args) / area
            return p

        return wrapper
    return decorator


def gauss(x: np.ndarray, s: float, m: float) -> np.ndarray:
    """Probability density of Gaussian noise.

    Args:
        x (np.ndarray): Intensity. The variable of the probability density.
        s (float): Standard deviation of the Gaussian noise.
        m (float): Mean of the Gaussian noise.

    Returns:
        np.ndarray: Probability density.
    """
    x1 = (x - m) / np.sqrt(2) / s
    p = np.exp(-x1**2)
    p = p / (np.sqrt(2 * np.pi) * s)
    return p


def gauss_pbcor(x: np.ndarray, s: float, m: float, R: float
                ) -> np.ndarray:
    """Probability density of Gaussian noise after primary-beam correction.

    Args:
        x (np.ndarray): Intensity. The variable of the probability density.
        s (float): Standard deviation of the Gaussian noise.
        m (float): Mean of the Gaussian noise.
        R (float): The maximum radius scaled by the FWHM of the primary beam.

    Returns:
        np.ndarray: Probability density.
    """
    x1 = (x - m) / np.sqrt(2) / s
    x0 = (x * 2**(-R**2) - m) / np.sqrt(2) / s
    p = erf(x1) - erf(x0)
    p = p / (2 * np.log(2) * x * R**2)
    return p


def select_noise(data: np.ndarray, sigma: str) -> np.ndarray:
    """Select data pixels to be used for noise estimation.

    Args:
        data (np.ndarray): Original data array.
        sigma (str): Selection methods. Multiple options are possible. 'edge', 'out', 'neg', or 'iter'.

    Returns:
        np.ndarray: 1D array that includes only the selected pixels.
    """
    n = np.array(data) * 1
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
    """This class holds the data selected as noise, histogram, and best-fit function.
       The following methods are acceptable for data selection. Multiple options are possible.
       'edge': use data[0] and data[-1].
       'out': exclude inner 60% about axes=-2 and -1.
       'neg': use only negative values.
       'iter': exclude outliers.
       The following methods are acceptable for noise estimation. Only single option is possible.
       'med': calculate rms from the median of data^2 assuming Gaussian.
       'hist': fit histgram with Gaussian.
       'hist-pbcor': fit histgram with PB-corrected Gaussian.
       '(no string)': calculate the mean and standard deviation.

    Args:
        data (np.ndarray): Original data array.
        sigma (str): Methods above, like 'edge,neg,hist-pbcor'.
    """
    def __init__(self, data: np.ndarray, sigma: str):
        self.data = select_noise(data, sigma)
        self.sigma = sigma
        self.m0 = np.mean(self.data)
        self.s0 = np.std(self.data)

    def gen_histogram(self, **kwargs):
        """Generage a pair of histogram and bins using numpy.histogram. The data values are shifted and scaled by the mean and standard deviation, respectively, to generate the histogram. The mean and standard deviation are stored as self.m0 and self.s0, respectively.
        """
        _kw = {'bins': 100, 'range': (-3.5, 3.5), 'density': True}
        _kw.update(kwargs)
        self.bins = _kw['bins']
        self.range = _kw['range']
        n = (self.data - self.m0) / self.s0
        hist, hbin = np.histogram(n, **_kw)
        hbin = (hbin[:-1] + hbin[1:]) / 2
        self.hist = hist
        self.hbin = hbin

    def fit_histogram(self, **kwargs):
        """kwargs is for plotastrodata.fitting_utils.EmceeCorner.
        """
        _kw = {'nwalkersperdim': 4, 'nsteps': 200, 'nburnin': 0}
        _kw.update(kwargs)
        if not hasattr(self, 'hist'):
            self.gen_histogram()
            print('Noise.gen_histogram() was done with default arguments.')
        f = gauss_pbcor if 'pbcor' in self.sigma else gauss
        model = normalize(range=self.range, bins=self.bins)(f)
        bounds = [[0.1, 2], [-2, 2]]
        if 'pbcor' in self.sigma:
            bounds.append([0.1, 2])
        # curve_fit does not work for this fitting.
        # 0.01 in sigma is set only to search the best-fit parameters.
        # Thus, this sigma does not justify the parameter errors.
        fitter = EmceeCorner(bounds=bounds, model=model,
                             xdata=self.hbin, ydata=self.hist,
                             sigma=np.max(self.hist) * 0.01)
        fitter.fit(**_kw)
        self.popt = fitter.popt
        self.mean = float(self.popt[1] * self.s0 + self.m0)
        self.std = float(self.popt[0] * self.s0)
        self.model = model(self.hbin, *self.popt)

    def plot_histogram(self, savefig: dict | str | None = None,
                       show: bool = False):
        """Make a simple figure of the histogram and model.

        Args:
            savefig (dict or str, optional): For plt.figure().savefig(). Defaults to None.
            show (bool, optional): True means doing plt.show(). Defaults to False.
        """
        if not hasattr(self, 'model'):
            self.fit_histogram()
            print('Noise.fit_histogram() was done with default arguments.')
        fig, ax = plt.subplots()
        ax.plot(self.hbin, self.hist, drawstyle='steps-mid')
        ax.plot(self.hbin, self.model, '-')
        ax.set_xlabel('(noise - m0) / s0')
        ax.set_ylabel('Probability density')
        close_figure(fig, savefig, show)


def estimate_rms(data: np.ndarray,
                 sigma: float | str | None = 'hist'
                 ) -> float:
    """Estimate a noise level of a data array.
       When a float number or None is given as sigma, this function just outputs it.

    Args:
        data (np.ndarray): Data array whose noise is estimated.
        sigma (float or str): Methods for the Noise class, like 'edge,neg,hist-pbcor'. Defaults to 'hist'.

    Returns:
        float: The estimated standard deviation of noise.
    """
    if sigma is None or isinstance(sigma, numbers.Number):
        return sigma

    if np.ndim(np.squeeze(data)) == 0:
        print('sigma cannot be estimated from only one pixel.')
        return 0.0

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
        s = 'Mean > 0.2 x standard deviation.'
        warnings.warn(s, UserWarning)
    return noise
