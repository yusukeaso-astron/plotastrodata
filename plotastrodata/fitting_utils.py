import numpy as np
import matplotlib.pyplot as plt
import emcee
import ptemcee
import corner
from dynesty import DynamicNestedSampler as DNS
import warnings
from tqdm import tqdm
from multiprocessing import Pool


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


class EmceeCorner():
    warnings.simplefilter('ignore', RuntimeWarning)

    def __init__(self, bounds: np.ndarray, logl: object | None = None,
                 model: object | None = None,
                 xdata: np.ndarray | None = None,
                 ydata: np.ndarray | None = None,
                 sigma: np.ndarray = 1, progressbar: bool = True,
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
        if logl is None and None not in [model, xdata, ydata]:
            def logl(x: np.ndarray) -> float:
                return np.sum((ydata - model(xdata, *x))**2 / sigma**2) / (-2)
        self.bounds = global_bounds
        self.dim = len(self.bounds)
        self.logl = logl
        self.logp = logp
        self.percent = percent
        self.ndata = 10000 if xdata is None else len(xdata)

    def fit(self, nwalkersperdim: int = 2, ntemps: int = 1, nsteps: int = 1000,
            nburnin: int = 500, ntry: int = 1, pos0: np.ndarray | None = None,
            savechain: str | None = None, ncores: int = 1, grcheck: bool = False,
            pt: bool = False) -> None:
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
        if ntemps > 1 and not pt:
            print('ntemps>1 is supported only with pt=True. Set pt=True.')
            pt = True
        if global_progressbar:
            bar = tqdm(total=ntry * ntemps * nwalkers * (nsteps + 1) // ncores)
            bar.set_description('Within the ranges')

        GR = [2] * self.dim
        i = 0
        while np.max(GR) > 1.25 and i < ntry:
            i += 1
            if pos0 is None:
                pos0 = np.random.rand(ntemps, nwalkers, self.dim) \
                       * (self.bounds[:, 1] - self.bounds[:, 0]) \
                       + self.bounds[:, 0]
                if not pt:
                    pos0 = pos0[0]
            if pt:
                pars = {'ntemps': ntemps, 'nwalkers': nwalkers, 'dim': self.dim,
                        'logl': self.logl, 'logp': self.logp}
                if ncores > 1:
                    with Pool(ncores) as pool:
                        sampler = ptemcee.Sampler(**pars, pool=pool)
                        sampler.run_mcmc(pos0, nsteps)
                else:
                    sampler = ptemcee.Sampler(**pars)
                    sampler.run_mcmc(pos0, nsteps)
                samples = sampler.chain[0, :, nburnin:, :]  # temperatures, walkers, steps, dim
            else:
                if ncores > 1:
                    print('Use logl as log_prob_fn to avoid function-in-function.')
                    log_prob_fn = self.logl
                else:
                    def log_prob_fn(x):
                        return self.logp(x) + self.logl(x)

                pars = {'nwalkers': nwalkers, 'ndim': self.dim,
                        'log_prob_fn': log_prob_fn}
                if ncores > 1:
                    with Pool(ncores) as pool:
                        sampler = emcee.EnsembleSampler(**pars, pool=pool)
                        sampler.run_mcmc(pos0, nsteps)
                else:
                    sampler = emcee.EnsembleSampler(**pars)
                    sampler.run_mcmc(pos0, nsteps)
                samples = sampler.chain[:, nburnin:, :]  # walkers, steps, dim
            if grcheck:
                # Gelman-Rubin statistics #
                B = np.std(np.mean(samples, axis=1), axis=0)
                W = np.mean(np.std(samples, axis=1), axis=0)
                V = (len(samples[0]) - 1) / len(samples[0]) * W \
                    + (nwalkers + 1) / (nwalkers - 1) * B
                d = self.ndata - self.dim - 1
                GR = np.sqrt((d + 3) / (d + 1) * V / W)
                ###########################
            else:
                GR = np.zeros(self.dim)
            if i == ntry - 1 and np.max(GR) > 1.25:
                print(f'!!! Max GR >1.25 during {ntry:d} trials.!!!')

        self.samples = samples
        if savechain is not None:
            np.save(savechain.replace('.npy', '') + '.npy', samples)
        if pt:
            lnps = sampler.logprobability[0]  # [0] is in the temperature axis.
            idx_best = np.unravel_index(np.argmax(lnps), lnps.shape)
            self.popt = sampler.chain[0][idx_best]  # [0] is in the temperature axis.
        else:
            lnps = sampler.lnprobability
            idx_best = np.unravel_index(np.argmax(lnps), lnps.shape)
            self.popt = sampler.chain[idx_best]
        self.lnps = lnps[:, nburnin:]
        s = samples.reshape((-1, self.dim))
        self.plow = np.percentile(s, self.percent[0], axis=0)
        self.pmid = np.percentile(s, 50, axis=0)
        self.phigh = np.percentile(s, self.percent[1], axis=0)
        if global_progressbar:
            print('')

    def plotcorner(self, show: bool = False,
                   savefig: str | None = None,
                   labels: list[str] | None = None,
                   cornerrange: list[float] | None = None) -> None:
        """Make the corner plot from self.samples.

        Args:
            show (bool, optional): Whether to show the corner plot. Defaults to False.
            savefig (str, optional): File name of the corner plot. Defaults to None.
            labels (list, optional): Labels for the corner plot. Defaults to None.
            cornerrange (list, optional): Range for the corner plot. Defaults to None.
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
        if savefig is not None:
            plt.savefig(savefig)
        if show:
            plt.show()
        plt.close()

    def plotchain(self, show: bool = False, savefig: str = None,
                  labels: list = None, ylim: list = None):
        """Plot parameters as a function of steps using self.samples. This method plots nine lines: percent[0], 50%, percent[1] percentiles (over the steps by 1% binning) of percent[0], 50%, percent[1] percentiles (over the walkers).

        Args:
            show (bool, optional): Whether to show the chain plot. Defaults to False.
            savefig (str, optional): File name of the chain plot. Defaults to None.
            labels (list, optional): Labels for the chain plot. Defaults to None.
            ylim (list, optional): Y-range for the chain plot. Defaults to None.
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
        fig.tight_layout()
        if savefig is not None:
            plt.savefig(savefig)
        if show:
            plt.show()
        plt.close()

    def posteriorongrid(self, ngrid: list[int] | int = 100,
                        log: list[bool] | bool = False, pcut: float = 0):
        """Calculate the posterior on a grid of ngrid x ngrid x ... x ngrid.

        Args:
            ngrid (list, optional): Number of grid on each parameter. Defaults to 100.
            log (list, optional): Whether to search in the logarithmic space. The percentile is counted in the linear space regardless of this option. Defaults to False.
            pcut (float, optional): Posterior is reset to be zero if it is below this cut off.
        """
        if type(ngrid) is int:
            ngrid = [ngrid] * self.dim
        if type(log) is bool:
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
        # fig.tight_layout()
        if savefig is not None:
            plt.savefig(savefig)
        if show:
            plt.show()
        else:
            plt.close()

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
