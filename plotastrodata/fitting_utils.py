import numpy as np
import matplotlib.pyplot as plt
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
    if np.all((global_bounds[0] < x) & (x < global_bounds[1])):
        return 0
    else:
        return -np.inf


class PTEmceeCorner():
    warnings.simplefilter('ignore', RuntimeWarning)

    def __init__(self, bounds: np.ndarray, logl: object | None = None,
                 model: object | None = None,
                 xdata: np.ndarray | None = None, ydata: np.ndarray | None = None,
                 sigma: np.ndarray = 1, progressbar: bool = True,
                 percent: list = [16, 84]):
        """Make bounds, logl, and logp for ptemcee.

        Args:
            bounds (np.ndarray): Bounds for ptemcee in the shape of (2, dim).
            logl (function, optional): Log likelihood for ptemcee. Defaults to None.
            model (function, optional): Model function to make a log likelihood function. Defaults to None.
            xdata (np.ndarray, optional): Input for the model function. Defaults to None.
            ydata (np.ndarray, optional): Values to be compared with the model. Defaults to None.
            sigma (np.ndarray, optional): Uncertainty to make a log likelihood function from the model. Defaults to 1.
            progressbar (bool, optional): Whether to show a progress bar. Defaults to True.
            percent (list, optional): The lower and upper percnetiles to be calculated. Defaults to [16, 84].
        """
        global global_bounds, global_progressbar
        global_bounds = np.array(bounds) if len(bounds) < 3 else np.transpose(bounds)
        global_progressbar = progressbar
        if logl is None and not (None in [model, xdata, ydata]):
            def logl(x: np.ndarray) -> float:
                return np.sum((ydata - model(xdata, *x))**2 / sigma**2) / (-2)
        self.bounds = global_bounds
        self.dim = len(self.bounds[0])
        self.logl = logl
        self.logp = logp
        self.percent = percent
        self.ndata = 10000 if xdata is None else len(xdata)

    def fit(self, nwalkersperdim: int = 2, ntemps: int = 1, nsteps: int = 1000,
            nburnin: int = 500, ntry: int = 1, pos0: np.ndarray | None = None,
            savechain: str | None = None, ncores: int = 1, grcheck: bool = False
            ) -> None:
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
        """
        global bar
        if nwalkersperdim < 2:
            print(f'nwalkersperdim < 2 is not allowed. Use 2 instead of {nwalkersperdim:d}.')
        nwalkers = max(nwalkersperdim, 2) * self.dim  # must be even and >= 2 * dim
        if global_progressbar:
            bar = tqdm(total=ntry * ntemps * nwalkers * (nsteps + 1) // ncores)
            bar.set_description('Within the ranges')

        GR = [2] * self.dim
        i = 0
        while np.max(GR) > 1.25 and i < ntry:
            i += 1
            if pos0 is None:
                pos0 = np.random.rand(ntemps, nwalkers, self.dim) \
                       * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
            pars = {'ntemps': ntemps, 'nwalkers': nwalkers, 'dim': self.dim,
                    'logl': self.logl, 'logp': self.logp}
            if ncores > 1:
                with Pool(ncores) as pool:
                    sampler = ptemcee.Sampler(**pars, pool=pool)
                    sampler.run_mcmc(pos0, nsteps)
            else:
                sampler = ptemcee.Sampler(**pars)
                sampler.run_mcmc(pos0, nsteps)
            samples = sampler.chain[0, :, nburnin:, :]  # temperature, walker, step, dim
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
        lnps = sampler.logprobability[0]  # [0] is in the temperature axis.
        idx_best = np.unravel_index(np.argmax(lnps), lnps.shape)
        self.popt = sampler.chain[0][idx_best]
        self.lnps = lnps[:, nburnin:]
        s = samples.reshape((-1, self.dim))
        self.plow = np.percentile(s, self.percent[0], axis=0)
        self.pmid = np.percentile(s, 50, axis=0)
        self.phigh = np.percentile(s, self.percent[1], axis=0)
        if global_progressbar:
            print('')

    def plotcorner(self, show: bool = False,
                   savefig: str | None = None, labels: list[float] | None = None,
                   cornerrange: list[float] = None) -> None:
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
            cornerrange = np.transpose(self.bounds)
        corner.corner(np.reshape(self.samples, (-1, self.dim)), truths=self.popt,
                      quantiles=[self.percent[0] / 100, 0.5, self.percent[1] / 100],
                      show_titles=True, labels=labels, range=cornerrange)
        if savefig is not None:
            plt.savefig(savefig)
        if show:
            plt.show()
        plt.close()

    def plotchain(self, show: bool = False, savefig: str = None,
                  labels: list = None, ylim: list = None):
        """Plot parameters as a function of steps using self.samples.

        Args:
            show (bool, optional): Whether to show the chain plot. Defaults to False.
            savefig (str, optional): File name of the chain plot. Defaults to None.
            labels (list, optional): Labels for the chain plot. Defaults to None.
            ylim (list, optional): Y-range for the chain plot. Defaults to None.
        """
        if labels is None:
            labels = [f'Par {i:d}' for i in range(self.dim)]
        if ylim is None:
            ylim = np.transpose(self.bounds)
        fig = plt.figure(figsize=(4, 2 * self.dim))
        x = np.arange(np.shape(self.samples)[1])
        naverage = max(1, len(x) // 100)
        nend = len(x) - len(x) % 100 if naverage > 1 else len(x)
        x = x[:nend:naverage]
        for i in range(self.dim):
            y = self.samples[:, :, i]
            plist = [self.percent[0], 50, self.percent[1]]
            y = [np.percentile(y, p, axis=0) for p in plist]
            y = [[np.percentile(np.reshape(yy[:nend], (naverage, -1)), p, axis=0)
                  for p in plist] for yy in y]
            ax = fig.add_subplot(self.dim, 1, i + 1)
            for yy, l, c in zip(y, [1, 2, 1], ['c', 'b', 'c']):
                for yyy, w in zip(yy, [0.25, 1, 0.25]):
                    ax.plot(x, yyy, '-', color=c, linewidth=l * w)
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

    def posteriorongrid(self, ngrid: list = 100, log: list[bool] = False, pcut: float = 0):
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
        for a, b, c, d in zip(*global_bounds, ngrid, log):
            pargrid.append(np.geomspace(a, b, c) if d else np.linspace(a, b, c))
        p = np.exp(self.logl(np.meshgrid(*pargrid[::-1], indexing='ij')[::-1]))
        p[p < pcut] = 0
        dpar = []
        for pg, l in zip(pargrid, log):
            if l:
                r = np.sqrt(pg[1] / pg[0])
                dpar.append(pg * (r - r**(-1)))
            else:
                dpar.append(pg * 0 + pg[1] - pg[0])
        vol = np.prod(np.meshgrid(*dpar[::-1], indexing='ij')[::-1], axis=0)
        adim = np.arange(self.dim)
        axlist = [tuple(np.delete(adim, i)) for i in adim[::-1]]  # adim[::-1] is becuase the 0th parameter is the innermost axis.
        p1d = [np.sum(p * vol, axis=a) / np.sum(vol, axis=a) for a in axlist]
        evidence = np.sum(p * vol) / np.sum(vol)
        p1dcum = [np.cumsum(q * w) / np.transpose([np.sum(q * w)])
                  for q, w in zip(p1d, dpar)]
        if np.all(p == 0):
            print('All posterior is below pcut.')
            self.popt = np.full(self.dim, np.nan)
            self.plow = np.full(self.dim, np.nan)
            self.pmid = np.full(self.dim, np.nan)
            self.phigh = np.full(self.dim, np.nan)
        else:
            iopt = np.unravel_index(np.argmax(p), np.shape(p))[::-1]
            self.popt = [t[i] for t, i in zip(pargrid, iopt)]

            def getpercentile(percent: float):
                idxmin = [np.argmin(np.abs(q - percent)) for q in p1dcum]
                return np.array([t[i] for t, i in zip(pargrid, idxmin)])

            self.plow = getpercentile(self.percent[0] / 100)
            self.pmid = getpercentile(0.5)
            self.phigh = getpercentile(self.percent[1] / 100)
        self.p = p
        self.p1d = p1d
        self.pargrid = pargrid
        self.vol = vol
        self.evidence = evidence

    def plotongrid(self, show: bool = False, savefig: str | None = None,
                   labels: list[str] = None, cornerrange: list[float] = None,
                   cmap: str = 'binary', levels: list[float] = [0.001, 0.01, 0.1]) -> None:
        """Make the corner plot from the posterior calculated on a grid.

        Args:
            show (bool, optional): Whether to show the corner plot. Defaults to False.
            savefig (str, optional): File name of the corner plot. Defaults to None.
            labels (list, optional): Labels for the corner plot. Defaults to None.
            cornerrange (list, optional): Range for the corner plot. Defaults to None.
            cmap: (str, optional): cmap for matplotlib.pyplot.plt.pcolormesh(). Defaults to 'binary'.
            levels: (list, optional): levels for matplotlib.pyplot.plt.contour() relative to the peak. Defaults to [0.001, 0.01, 0.1].
        """
        adim = np.arange(self.dim)
        if labels is None:
            labels = [f'Par {i:d}' for i in adim]
        if cornerrange is None:
            cornerrange = np.transpose(self.bounds)
        x = self.pargrid
        y = self.p1d
        fig = plt.figure(figsize=(2 * self.dim * 1.2, 2 * self.dim))
        fig.subplots_adjust(hspace=0, wspace=0, top=0.87, right=0.87)
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
                    ax[k].plot(x[i], y[i], 'k-')
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
        fig.tight_layout()
        if savefig is not None:
            plt.savefig(savefig)
        if show:
            plt.show()

    def getDNSevidence(self, **kwargs):
        """Calculate the Bayesian evidence for a model using dynamic nested sampling through dynesty.
        """
        def prior_transform(u):
            return self.bounds[0] + (self.bounds[1] - self.bounds[0]) * u
        dsampler = DNS(loglikelihood=self.logl,
                       prior_transform=prior_transform,
                       ndim=self.dim, **kwargs)
        dsampler.run_nested(print_progress=False)
        results = dsampler.results
        evidence = np.exp(results.logz[-1])
        error = evidence * results.logzerr[-1]
        return {'evidence': evidence, 'error': error}
