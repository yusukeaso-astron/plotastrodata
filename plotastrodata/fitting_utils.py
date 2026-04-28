import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
import ptemcee
import warnings
from dynesty import DynamicNestedSampler as DNS
from multiprocessing import Pool
from tqdm import tqdm

from plotastrodata.matrix_utils import Mrot, dot2d
from plotastrodata.other_utils import close_figure


global_bounds = None
bar = None
global_progressbar = True


def logp(x: np.ndarray) -> float:
    """Log prior function made from the boundary (global_bounds) of fitting parameters.

    Args:
        x (np.ndarray): The fitting parameters.

    Returns:
        float: The log prior function. 0 if all the parameters are in the boundary else -np.inf.
    """
    if global_progressbar:
        bar.update(1)
    if np.all((global_bounds[:, 0] < x) & (x < global_bounds[:, 1])):
        return 0
    else:
        return -np.inf


def _get_GR(samples: np.ndarray, nwalkers: int, ndata: int, dim: int
            ) -> np.ndarray:
    """Calculate the Gelman-Rubin statistics."""
    B = np.std(np.mean(samples, axis=1), axis=0)
    W = np.mean(np.std(samples, axis=1), axis=0)
    V = (len(samples[0]) - 1) / len(samples[0]) * W \
        + (nwalkers + 1) / (nwalkers - 1) * B
    d = ndata - dim - 1
    GR = np.sqrt((d + 3) / (d + 1) * V / W)
    return GR


def _check_GR(samples: np.ndarray, nwalkers: int, ndata: int, dim: int,
              i: int, ntry: int = 1, grcheck: bool = False) -> int:
    if not grcheck:
        return ntry

    GR = _get_GR(samples=samples, nwalkers=nwalkers, ndata=ndata, dim=dim)
    if np.max(GR) <= 1.25:
        return ntry

    if i == ntry:
        print(f'!!! Max GR >1.25 during {ntry:d} trials.!!!')
    return i


class EmceeCorner():
    warnings.simplefilter('ignore', RuntimeWarning)

    def __init__(self, bounds: np.ndarray, logl: object | None = None,
                 model: object | None = None,
                 xdata: np.ndarray | None = None,
                 ydata: np.ndarray | None = None,
                 sigma: np.ndarray = 1, progressbar: bool = False,
                 percent: list = [16, 84]):
        """Make bounds, logl, and logp for ptemcee.

        Args:
            bounds (np.ndarray): Bounds for ptemcee in the shape of (dim, 2).
            logl (function, optional): Log likelihood for ptemcee. Defaults to None.
            model (function, optional): Model function to make a log likelihood function. Defaults to None.
            xdata (np.ndarray, optional): Input for the model function. Defaults to None.
            ydata (np.ndarray, optional): Values to be compared with the model. Defaults to None.
            sigma (np.ndarray, optional): Uncertainty to make a log likelihood function from the model. Defaults to 1.
            progressbar (bool, optional): Whether to show a progress bar. Defaults to True.
            percent (list, optional): The lower and upper percnetiles to be calculated. Defaults to [16, 84].
        """
        global global_bounds, global_progressbar
        if len(bounds[0]) > 3:
            global_bounds = np.transpose(bounds)
            print('bounds has been transposed because its shape is (2, dim).')
        else:
            global_bounds = np.array(bounds)
        global_progressbar = progressbar
        if logl is None and (model is not None
                             and xdata is not None
                             and ydata is not None):
            def logl(x: np.ndarray) -> float:
                chi2 = np.sum((ydata - model(xdata, *x))**2 / sigma**2)
                return chi2 / (-2)
        self.bounds = global_bounds
        self.dim = len(self.bounds)
        self.logl = logl
        self.logp = logp
        self.percent = percent
        self.ndata = 10000 if xdata is None else len(xdata)

    def _get_pos0(self, ntemps: int, nwalkers: int, pt: bool) -> np.ndarray:
        """Create initial walker positions within parameter bounds."""
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        width = upper - lower
        pos0 = np.random.rand(ntemps, nwalkers, self.dim) * width + lower
        return pos0 if pt else pos0[0]

    def _run_sampler(self, pos0: np.ndarray, pt: bool,
                     ncores: int, ntemps: int,
                     nsteps: int, nwalkers: int) -> object:
        """Create and run the sampler, then return it."""
        if pt:
            sampler_cls = ptemcee.Sampler
            sampler_kwargs = {'ntemps': ntemps,
                              'nwalkers': nwalkers, 'dim': self.dim,
                              'logl': self.logl, 'logp': self.logp}
        else:
            if ncores > 1:
                print('Use logl as log_prob_fn to avoid function-in-function.')
                log_prob_fn = self.logl
            else:
                def log_prob_fn(x):
                    return self.logp(x) + self.logl(x)

            sampler_cls = emcee.EnsembleSampler
            sampler_kwargs = {'nwalkers': nwalkers, 'ndim': self.dim,
                              'log_prob_fn': log_prob_fn}
        if ncores > 1:
            with Pool(ncores) as pool:
                sampler = sampler_cls(**sampler_kwargs, pool=pool)
        else:
            sampler = sampler_cls(**sampler_kwargs, pool=None)
        sampler.run_mcmc(pos0, nsteps)
        return sampler

    def _get_samples(self, sampler, nburnin: int, pt: bool) -> np.ndarray:
        """Extract post-burn-in samples from sampler chain."""
        if pt:
            return sampler.chain[0, :, nburnin:, :]  # temperatures, walkers, steps, dim
        else:
            return sampler.chain[:, nburnin:, :]  # walkers, steps, dim

    def _get_lnp_popt(self, sampler, pt: bool, nburnin: int,
                      ) -> tuple[np.ndarray, np.ndarray]:
        """Get log probabilities and best-fit parameters from sampler."""
        if pt:
            lnp = sampler.logprobability[0]  # 0th temperature chain
            chain = sampler.chain[0]
        else:
            lnp = sampler.lnprobability
            chain = sampler.chain
        idx_best = np.unravel_index(np.argmax(lnp), lnp.shape)
        popt = chain[idx_best]
        lnp = lnp[:, nburnin:]
        return lnp, popt

    def _get_percentiles(self, samples: np.ndarray
                         ) -> tuple[float, float, float]:
        """Compute summary statistics (percentiles) from MCMC samples."""
        s = samples.reshape(-1, self.dim)
        plow = np.percentile(s, self.percent[0], axis=0)
        pmid = np.percentile(s, 50, axis=0)
        phigh = np.percentile(s, self.percent[1], axis=0)
        return plow, pmid, phigh

    def fit(self, nwalkersperdim: int = 2,
            ntemps: int = 1, nsteps: int = 1000,
            nburnin: int = 500, ntry: int = 1,
            pos0: np.ndarray | None = None,
            savechain: str | None = None, ncores: int = 1,
            grcheck: bool = False, pt: bool = False) -> None:
        """Perform a Markov Chain Monte Carlo (MCMC) fitting process using the ptemcee library, which is a parallel tempering version of the emcee package, and make a corner plot of the samples using the corner package.

        Args:
            nwalkersperdim (int, optional): Number of walkers per dimension. Defaults to 2.
            ntemps (int, optional): Number of temperatures. Defaults to 2.
            nsteps (int, optional): Number of steps, including the steps for burn-in. Defaults to 1000.
            nburnin (int, optional): Number of burn-in steps. Defaults to 500.
            ntry (int, optional): Number of trials for the Gelman-Rubin check. Defaults to 1.
            pos0 (np.nparray, optional): Initial parameter set in the shape of (ntemps, nwalkers, dim). Defaults to None.
            savechain (str, optional): File name of the chain in format of .npy. Defaults to None.
            ncores (int, optional): Number of cores for multiprocessing.Pool. ncores=1 does not use multiprocessing. Defaults to 1.
            grcheck (bool, optional): Whether to check Gelman-Rubin statistics. Defaults to False.
            pt (bool, optional): Whether to use ptemcee; otherwise, emcee is used. Defaults to False.
        """
        global bar
        if nwalkersperdim < 2:
            print('nwalkersperdim < 2 is not allowed.'
                  + f' Use 2 instead of {nwalkersperdim:d}.')
        nwalkers = max(nwalkersperdim, 2) * self.dim  # must be even and >= 2 * dim
        if ntemps > 1:
            pt = True
        if global_progressbar:
            total = ntry * ntemps * nwalkers * (nsteps + 1) // ncores
            bar = tqdm(total=total)
            bar.set_description('Within the ranges')
        samples = None
        sampler = None
        for i in range(1, ntry + 1):
            if pos0 is None:
                pos0 = self._get_pos0(ntemps=ntemps, nwalkers=nwalkers, pt=pt)
            sampler = self._run_sampler(pos0=pos0, pt=pt, ncores=ncores,
                                        ntemps=ntemps, nsteps=nsteps,
                                        nwalkers=nwalkers)
            samples = self._get_samples(sampler=sampler,
                                        nburnin=nburnin, pt=pt)
            i = _check_GR(samples=samples, nwalkers=nwalkers,
                          ndata=self.ndata, dim=self.dim,
                          i=i, ntry=ntry, grcheck=grcheck)
        if savechain is not None:
            np.save(savechain.removesuffix('.npy') + '.npy', samples)
        self.lnp, self.popt = self._get_lnp_popt(sampler=sampler, pt=pt,
                                                 nburnin=nburnin)
        self.plow, self.pmid, self.phigh = self._get_percentiles(samples)
        self.samples = samples
        if global_progressbar:
            print()

    def plotcorner(self, labels: list[str] | None = None,
                   cornerrange: list[float] | None = None,
                   savefig: dict | str | None = None,
                   show: bool = False) -> None:
        """Make the corner plot from self.samples.

        Args:
            labels (list, optional): Labels for the corner plot. Defaults to None.
            cornerrange (list, optional): Range for the corner plot. Defaults to None.
            savefig (dict or str, optional): For plt.figure().savefig(). Defaults to None.
            show (bool, optional): True means doing plt.show(). Defaults to False.
        """
        if labels is None:
            labels = [f'Par {i:d}' for i in range(self.dim)]
        if cornerrange is None:
            cornerrange = self.bounds
        corner.corner(np.reshape(self.samples, (-1, self.dim)),
                      truths=self.popt,
                      quantiles=[self.percent[0] / 100,
                                 0.5,
                                 self.percent[1] / 100],
                      show_titles=True, labels=labels, range=cornerrange)
        close_figure(plt, savefig, show, tight=False)

    def plotchain(self, labels: list = None, ylim: list = None,
                  savefig: dict | str = None, show: bool = False):
        """Plot parameters as a function of steps using self.samples. This method plots nine lines: percent[0], 50%, percent[1] percentiles (over the steps by 1% binning) of percent[0], 50%, percent[1] percentiles (over the walkers).

        Args:
            labels (list, optional): Labels for the chain plot. Defaults to None.
            ylim (list, optional): Y-range for the chain plot. Defaults to None.
            savefig (dict or str, optional): For plt.figure().savefig(). Defaults to None.
            show (bool, optional): True means doing plt.show(). Defaults to False.
        """
        if labels is None:
            labels = [f'Par {i:d}' for i in range(self.dim)]
        if ylim is None:
            ylim = self.bounds
        fig = plt.figure(figsize=(4, 2 * self.dim))
        x = np.arange(np.shape(self.samples)[1])
        naverage = max(1, len(x) // 100)
        nend = len(x) - len(x) % 100 if naverage > 1 else len(x)
        x = x[:nend:naverage]
        for i in range(self.dim):
            y = self.samples[:, :, i]  # walkers, steps, dim
            plist = [self.percent[0], 50, self.percent[1]]
            y = [np.percentile(y, p, axis=0) for p in plist]  # percent over the walkers, steps
            y = [[np.percentile(np.reshape(yy[:nend], (naverage, -1)), p, axis=0)
                  for p in plist] for yy in y]  # percent over the walkers, percent over the steps
            ax = fig.add_subplot(self.dim, 1, i + 1)
            for yy, scale, c in zip(y, [1, 2, 1], ['c', 'b', 'c']):
                for yyy, lw in zip(yy, [0.25, 1, 0.25]):
                    ax.plot(x, yyy, '-', color=c, linewidth=lw * scale)
            ax.set_ylim(ylim[i])
            ax.set_ylabel(labels[i])
            if i < self.dim - 1:
                ax.set_xticks([])
            else:
                ax.set_xlabel('Step')
        close_figure(fig, savefig, show)

    def posteriorongrid(self, ngrid: list[int] | int = 100,
                        log: list[bool] | bool = False, pcut: float = 0):
        """Calculate the posterior on a grid of ngrid x ngrid x ... x ngrid.

        Args:
            ngrid (list, optional): Number of grid on each parameter. Defaults to 100.
            log (list, optional): Whether to search in the logarithmic space. The percentile is counted in the linear space regardless of this option. Defaults to False.
            pcut (float, optional): Posterior is reset to be zero if it is below this cut off.
        """
        if isinstance(ngrid, int):
            ngrid = [ngrid] * self.dim
        if isinstance(log, bool):
            log = [log] * self.dim
        pargrid = []
        inzip = [self.bounds[:, 0], self.bounds[:, 1], ngrid, log]
        for start, stop, num, uselog in zip(*inzip):
            args = [start, stop, num]
            pargrid.append(np.geomspace(*args) if uselog else np.linspace(*args))
        p = np.exp(self.logl(np.meshgrid(*pargrid[::-1], indexing='ij')[::-1]))
        p[p < pcut] = 0
        dpar = []
        for pg, uselog in zip(pargrid, log):
            if uselog:
                r = np.sqrt(pg[1] / pg[0])
                dpar.append(pg * (r - r**(-1)))
            else:
                dpar.append(pg * 0 + pg[1] - pg[0])
        vol = np.prod(np.meshgrid(*dpar[::-1], indexing='ij')[::-1], axis=0)
        adim = np.arange(self.dim)
        axlist = [tuple(np.delete(adim, i)) for i in adim[::-1]]  # adim[::-1] is becuase the 0th parameter is the innermost axis.
        p1d = [np.sum(p * vol, axis=a) / np.sum(vol, axis=a) for a in axlist]
        evidence = np.sum(p * vol) / np.sum(vol)
        if np.all(p == 0):
            print('All posterior is below pcut.')
            self.popt = np.full(self.dim, np.nan)
            self.plow = np.full(self.dim, np.nan)
            self.pmid = np.full(self.dim, np.nan)
            self.phigh = np.full(self.dim, np.nan)
        else:
            i_max = np.unravel_index(np.argmax(p), np.shape(p))[::-1]
            self.popt = np.array([p[i] for p, i in zip(pargrid, i_max)])

            def getpercentile(percent: float):
                a = [np.percentile(g, percent, method='inverted_cdf', weights=p)
                     for g, p in zip(pargrid, p1d)]
                return np.array(a)

            self.plow = getpercentile(self.percent[0])
            self.pmid = getpercentile(50)
            self.phigh = getpercentile(self.percent[1])
        self.p = p
        self.p1d = p1d
        self.pargrid = pargrid
        self.vol = vol
        self.evidence = evidence

    def plotongrid(self, show: bool = False, savefig: str | None = None,
                   labels: list[str] = None,
                   cornerrange: list[float] = None, cmap: str = 'binary',
                   levels: list[float] = [0.011109, 0.135335, 0.606531]
                   ) -> None:
        """Make the corner plot from the posterior calculated on a grid.

        Args:
            show (bool, optional): Whether to show the corner plot. Defaults to False.
            savefig (str, optional): File name of the corner plot. Defaults to None.
            labels (list, optional): Labels for the corner plot. Defaults to None.
            cornerrange (list, optional): Range for the corner plot. Defaults to None.
            cmap: (str, optional): cmap for matplotlib.pyplot.plt.pcolormesh(). Defaults to 'binary'.
            levels: (list, optional): levels for matplotlib.pyplot.plt.contour() relative to the peak. Defaults to [exp(-0.5*3^2), exp(-0.5*2^2), exp(-0.5*1^2)].
        """
        adim = np.arange(self.dim)
        if labels is None:
            labels = [f'Par {i:d}' for i in adim]
        if cornerrange is None:
            cornerrange = self.bounds
        x = self.pargrid
        y = self.p1d
        fig = plt.figure(figsize=(2 * self.dim * 1.2, 2 * self.dim * 1.2))
        fig.subplots_adjust(hspace=0.05, wspace=0.05, top=0.87, right=0.87)
        ax = np.empty(self.dim * self.dim, dtype='object')
        for i in adim:
            for j in adim:
                if i < j:
                    continue
                k = self.dim * i + j
                if i == j:
                    s0 = self.pmid[i]
                    s1 = self.phigh[i] - self.pmid[i]
                    s2 = self.pmid[i] - self.plow[i]
                    s0 = f'{s0:.2f}'
                    s1 = '^{+' + f'{s1:.2f}' + '}'
                    s2 = '_{-' + f'{s2:.2f}' + '}'
                    s0 = s0 + s1 + s2
                    ax[k] = fig.add_subplot(self.dim, self.dim, k + 1)
                    ax[k].plot(x[i], y[i], 'k-', drawstyle='steps-mid')
                    ax[k].axvline(self.popt[i])
                    ax[k].axvline(self.plow[i], linestyle='--', color='k')
                    ax[k].axvline(self.pmid[i], linestyle='--', color='k')
                    ax[k].axvline(self.phigh[i], linestyle='--', color='k')
                    ax[k].set_title(rf'{labels[i]}=${s0}$')
                    ax[k].set_xlim(cornerrange[i])
                    ax[k].set_ylim([0, np.max(y[i]) * 1.2])
                    ax[k].set_yticks([])
                    if i == self.dim - 1:
                        plt.setp(ax[k].get_xticklabels(), rotation=45)
                        ax[k].set_xlabel(labels[i])
                    else:
                        plt.setp(ax[k].get_xticklabels(), visible=False)
                else:
                    sharex = ax[self.dim * (i - 1) + j]
                    sharey = ax[self.dim * i + (j - 1)] if j > 1 else None
                    ax[k] = fig.add_subplot(self.dim, self.dim, k + 1,
                                            sharex=sharex, sharey=sharey)
                    axis = tuple(np.delete(adim[::-1], [i, j]))
                    yy = np.sum(self.p * self.vol, axis=axis) \
                        / np.sum(self.vol, axis=axis)
                    ax[k].pcolormesh(x[j], x[i], yy, cmap=cmap)
                    ax[k].contour(x[j], x[i], yy, colors='k',
                                  levels=np.array(levels) * np.nanmax(yy))
                    ax[k].plot(self.popt[j], self.popt[i], 'o')
                    ax[k].axvline(self.popt[j])
                    ax[k].axhline(self.popt[i])
                    ax[k].set_xlim(cornerrange[j])
                    ax[k].set_ylim(cornerrange[i])
                    if j == 0:
                        ax[k].set_ylabel(labels[i])
                        plt.setp(ax[k].get_yticklabels(), rotation=45)
                    else:
                        plt.setp(ax[k].get_yticklabels(), visible=False)
                    if i == self.dim - 1:
                        ax[k].set_xlabel(labels[j])
                        plt.setp(ax[k].get_xticklabels(), rotation=45)
                    else:
                        plt.setp(ax[k].get_xticklabels(), visible=False)
        close_figure(fig, savefig, show, tight=False)

    def getDNSevidence(self, **kwargs):
        """Calculate the Bayesian evidence for a model using dynamic nested sampling through dynesty.
        """
        def prior_transform(u):
            b0 = self.bounds[:, 0]
            b1 = self.bounds[:, 1]
            return b0 + (b1 - b0) * u

        dsampler = DNS(loglikelihood=self.logl,
                       prior_transform=prior_transform,
                       ndim=self.dim, **kwargs)
        dsampler.run_nested(print_progress=False)
        results = dsampler.results
        evidence = np.exp(results.logz[-1])
        error = evidence * results.logzerr[-1]
        self.evidence = evidence
        return {'evidence': evidence, 'error': error}


def gaussian1d(x: np.ndarray | float,
               amplitude: float, xo: float, fwhm: float,
               ) -> np.ndarray:
    """One dimensional Gaussian function.

    Args:
        x (np.ndarray): Variable of the Gaussian function.
        amplitude (float): Peak value.
        xo (float): Offset in the x direction.
        fwhm (float): Full width at half maximum.

    Returns:
        g (np.ndarray): 1D numpy array.
    """
    g = amplitude * np.exp2(-4 * (((x - xo) / fwhm)**2))
    return g


def gaussian2d(xy: np.ndarray,
               amplitude: float, xo: float, yo: float,
               fwhm_major: float, fwhm_minor: float, pa: float
               ) -> np.ndarray:
    """Two dimensional Gaussian function.

    Args:
        xy (np.ndarray): A pair of (x, y).
        amplitude (float): Peak value.
        xo (float): Offset in the x direction.
        yo (float): Offset in the y direction.
        fwhm_major (float): Full width at half maximum in the major axis (but can be shorter than the minor axis).
        fwhm_minor (float): Full width at half maximum in the minor axis (but can be longer then the major axis).
        pa (float): Position angle of the major axis from the +y axis to the +x axis in the unit of degree.

    Returns:
        g (np.ndarray): Output array in the same shape as xy.
    """
    s, t = dot2d(Mrot(-pa), [xy[1] - yo, xy[0] - xo])
    g = amplitude * np.exp2(-4 * ((s / fwhm_major)**2 + (t / fwhm_minor)**2))
    return g


def gaussfit1d(xdata: np.ndarray, ydata: np.ndarray,
               sigma: float | np.ndarray | None,
               show: bool = False, **kwargs) -> dict:
    """Gaussian fitting to a pair of 1D arrays.

    Args:
        xdata (np.ndarray): ydata is compared with Gauss(xdata).
        ydata (np.ndarray): ydata is compared with Gauss(xdata).
        sigma (float | np.ndarray | None): Noise level of ydata. If None is given, sigma is estimated by a temporary fitting. Defaults to None.
        show (bool, optional): True means to show the best-fit parameters and uncertainties. Defaults to False.

    Returns:
        dict: The keys are popt, perr, and sigma.
    """
    xmin, xmax = np.min(xdata), np.max(xdata)
    ymin, ymax = np.min(ydata), np.max(ydata)
    xw = xmax - xmin
    yw = ymax - ymin
    dx = np.abs(xdata[1] - xdata[0])
    bounds = [[ymin - yw * 10, ymax + yw * 10], [xmin, xmax], [dx, xw]]
    sigtmp = sigma or max(np.abs(ymin), np.abs(ymax)) * 0.01
    for i in range(2 if sigma is None else 1):
        fitter = EmceeCorner(bounds=bounds, model=gaussian1d,
                             sigma=sigtmp, xdata=xdata, ydata=ydata)
        fitter.fit(**kwargs)
        if i == 0:
            sigtmp = np.std(ydata - gaussian1d(xdata, *fitter.popt))
    popt = fitter.popt
    plow = fitter.plow
    phigh = fitter.phigh
    perr = (phigh - plow) / 2
    if show:
        print('Gauss (peak, center, FWHM):', popt)
        print('Gauss uncertainties:', perr)
        if sigma is None:
            print('Estimated sigma: ', sigtmp)
    return {'popt': popt, 'perr': perr, 'sigma': sigtmp}


def gaussfit2d(xdata: np.ndarray, ydata: np.ndarray, zdata: np.ndarray,
               sigma: float | np.ndarray | None,
               show: bool = False, **kwargs) -> dict:
    """Gaussian fitting to a pair of 1D arrays.

    Args:
        xdata (np.ndarray): zdata is compared with Gauss(xdata, ydata).
        ydata (np.ndarray): zdata is compared with Gauss(xdata, ydata).
        zdata (np.ndarray): zdata is compared with Gauss(xdata, ydata).
        sigma (float | np.ndarray | None): Noise level of ydata. If None is given, sigma is estimated by a temporary fitting. Defaults to None.
        show (bool, optional): True means to show the best-fit parameters and uncertainties. Defaults to False.

    Returns:
        dict: The keys are popt, perr, and sigma.
    """
    xmin, xmax = np.min(xdata), np.max(xdata)
    ymin, ymax = np.min(ydata), np.max(ydata)
    zmin, zmax = np.min(zdata), np.max(zdata)
    xw = xmax - xmin
    yw = ymax - ymin
    zw = zmax - zmin
    dx = min(np.abs(xdata[1] - xdata[0]), np.abs(ydata[1] - ydata[0]))
    xw = max(xw, yw)
    xy = np.meshgrid(xdata, ydata)
    sigtmp = sigma or max(np.abs(zmin), np.abs(zmax)) * 0.01

    def model(xy, a, cx, cy, wmaj, wmin, pa):
        if wmaj < wmin:
            return np.inf
        else:
            return gaussian2d(xy, a, cx, cy, wmaj, wmin, pa)

    bounds = [[zmin - zw * 10, zmax + zw * 10],
              [xmin, xmax], [ymin, ymax],
              [dx, xw], [dx, xw], [-90, 90]]
    for i in range(2):
        fitter = EmceeCorner(bounds=bounds, model=model,
                             sigma=sigtmp, xdata=xy, ydata=zdata)
        fitter.fit(**kwargs)
        a, cx, cy, wmaj, wmin, pa = fitter.popt
        wmax = max(wmaj, wmin)
        bounds = [sorted([a / 2, a * 2]),
                  [cx - wmax, cx + wmax], [cy - wmax, cy + wmax],
                  [wmaj / 2, wmaj * 2], [wmin / 2, wmin * 2],
                  [pa - 45, pa + 45]]
        if i == 0 and sigma is None:
            sigtmp = np.std(ydata - gaussian2d(xy, *fitter.popt))
    popt = fitter.popt
    popt[-1] = (popt[-1] + 90) % 180 - 90
    perr = (fitter.phigh - fitter.plow) / 2
    if show:
        print('Gauss (peak, center, FWHM):', popt)
        print('Gauss uncertainties:', perr)
        if sigma is None:
            print('Estimated sigma: ', sigtmp)
    return {'popt': popt, 'perr': perr, 'sigma': sigtmp}
