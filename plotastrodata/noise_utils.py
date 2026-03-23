import warnings
import numpy as np
from scipy.special import erf

from plotastrodata.fitting_utils import EmceeCorner


def normalize(h_range: tuple = (-3.5, 3.5), bins: int = 100):
    """Decorator factory to normalize a function over h_range."""
    def decorator(f):
        h = np.linspace(*h_range, bins + 1)
        h = (h[1:] + h[:-1]) / 2
        dh = (h_range[1] - h_range[0]) / 100

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


def estimate_rms_hist(data: np.ndarray, pbcor: bool = False,
                      h_range: tuple = (-3.5, 3.5),
                      bins: int = 100) -> tuple:
    """Function to obtain the mean and standard deviation using the histogram of a given data array.

    Args:
        data (np.ndarray): Data array whose noise is estimated.
        pbcor (bool): Whether it considers the primary beam correction.
        h_range (tuple, optional): Range for numpy.histogram(). Defaults to (-3.5, 3.5).
        bins (int, optional): Bins for numpy.histogram(). Defaults to 100.

    Returns:
        tuple: (mean, standard deviation)
    """
    m0 = np.mean(data)
    s0 = np.std(data)
    hist, hbin = np.histogram((data - m0) / s0, bins=bins,
                              density=True, range=h_range)
    hbin = (hbin[:-1] + hbin[1:]) / 2
    f = gauss_pbcor if pbcor else gauss
    model = normalize(h_range, bins)(f)
    bounds = [[0.1, 2], [-2, 2]]
    if pbcor:
        bounds.append([0.1, 2])
    # curve_fit does not work for this fitting.
    fitter = EmceeCorner(bounds=bounds, sigma=np.max(hist) * 0.01,
                         model=model, xdata=hbin, ydata=hist)
    fitter.fit(nwalkersperdim=4, nsteps=200, nburnin=0)
    popt = fitter.popt
    ave = popt[1] * s0 + m0
    noise = popt[0] * s0
    return ave, noise


def estimate_rms(data: np.ndarray, sigma: float | str | None = 'hist'
                 ) -> float:
    """Estimate a noise level of a N-D array.
       When a float number or None is given, this function just outputs it.
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
        data (np.ndarray): Data array whose noise is estimated.
        sigma (float or str): Methods above, like 'edge,neg,hist-pbcor'. Defaults to 'hist'.

    Returns:
        float: The estimated standard deviation of noise.
    """
    nums = [float, int, np.float64, np.int64, np.float32, np.int32]
    if sigma is None or type(sigma) in nums:
        return sigma

    if np.ndim(np.squeeze(data)) == 0:
        print('sigma cannot be estimated from only one pixel.')
        return 0.0

    # Selection
    n = data * 1
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
    # Estimation
    if 'hist' in sigma:
        ave, noise = estimate_rms_hist(n, sigma, 'pbcor' in sigma)
    elif 'med' in sigma:
        ave = 0
        noise = np.sqrt(np.median(n**2) / 0.454936)
    else:
        ave = np.mean(n)
        noise = np.std(n)
    if np.abs(ave) > 0.2 * noise:
        s = 'The intensity offset is larger than 0.2 sigma.'
        warnings.warn(s, UserWarning)
    return noise

