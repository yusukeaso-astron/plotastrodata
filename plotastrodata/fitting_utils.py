import numpy as np
import matplotlib.pyplot as plt
import ptemcee
import corner
import warnings
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


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
    if  np.all((global_bounds[0] < x) & (x < global_bounds[1])):
        return 0
    else:
        return -np.inf

class PTEmceeCorner():
    warnings.simplefilter('ignore', RuntimeWarning)
    def __init__(self, bounds: np.ndarray, logl=None, model=None,
                 xdata: np.ndarray = None, ydata: np.ndarray = None,
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
            nburnin: int = 500, ntry: int = 1, pos0: np.ndarray = None,
            savechain: str = None, ncore: int = 1):
        """Perform a Markov Chain Monte Carlo (MCMC) fitting process using the ptemcee library, which is a parallel tempering version of the emcee package, and make a corner plot of the samples using the corner package.

        Args:
            nwalkersperdim (int, optional): Number of walkers per dimension. Defaults to 2.
            ntemps (int, optional): Number of temperatures. Defaults to 2.
            nsteps (int, optional): Number of steps, including the steps for burn-in. Defaults to 1000.
            nburnin (int, optional): Number of burn-in steps. Defaults to 500.
            ntry (int, optional): Number of trials for the Gelman-Rubin check. Defaults to 1.
            pos0 (np.nparray, optional): Initial parameter set in the shape of (ntemps, nwalkers, dim). Defaults to None.
            savechain (str, optional): File name of the chain in format of .npy. Defaults to None.
            ncore (int, optional): Number of cores for multiprocessing.Pool. ncore=1 does not use multiprocessing. Defaults to 1.
        """
        global bar
        nwalkers = max(nwalkersperdim, 2) * self.dim  # must be even and >= 2 * dim
        if global_progressbar:
            bar = tqdm(total=ntry * ntemps * nwalkers * (nsteps + 1) / ncore)
            bar.set_description('Within the ranges')

        GR = [2] * self.dim
        i =  0
        while np.min(GR) > 1.25 and i < ntry:
            i += 1
            if pos0 is None:
                pos0 = np.random.rand(ntemps, nwalkers, self.dim) \
                       * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
            pars = {'ntemps':ntemps, 'nwalkers':nwalkers, 'dim':self.dim,
                    'logl':self.logl, 'logp':self.logp}
            if ncore > 1:
                print(f'Use {ncore:d} / {cpu_count():d} CPUs')
                with Pool(ncore) as pool:
                    sampler = ptemcee.Sampler(**pars, pool=pool)
                    sampler.run_mcmc(pos0, nsteps)
            else:
                sampler = ptemcee.Sampler(**pars)
                sampler.run_mcmc(pos0, nsteps)
            samples = sampler.chain[0, :, nburnin:, :]
            ##### Gelman-Rubin statistics #####
            B = np.std(np.mean(samples, axis=1), axis=0)
            W = np.mean(np.std(samples, axis=1), axis=0)
            V = (len(samples[0]) - 1) / len(samples[0]) * W \
                + (nwalkers + 1) / (nwalkers - 1) * B
            d = self.ndata - self.dim - 1
            GR = np.sqrt((d + 3) / (d + 1) * V / W)
            ###################################
            if i == ntry - 1 and np.max(GR) > 1.25:
                print(f'!!! Max GR >1.25 during {ntry:d} trials.!!!')

        self.samples = samples
        if savechain is not None:
            np.save(savechain.replace('.npy', '') + '.npy', samples)
        lnps = sampler.logprobability[0]  # [0] is in the temperature axis.
        idx_best = np.unravel_index(np.argmax(lnps), lnps.shape)
        self.popt = sampler.chain[0][idx_best]
        lnps = lnps[:, nburnin:]
        self.lnps = lnps
        samples = samples.reshape((-1, self.dim))
        lnps = lnps.reshape((-1, 1))
        self.low = np.percentile(samples, self.percent[0], axis=0)
        self.mid = np.percentile(samples, 50, axis=0)
        self.high = np.percentile(samples, self.percent[1], axis=0)
    
    def plotcorner(self, show: bool = False,
                   savefig: str = None, labels: list = None,
                   cornerrange: list = None):
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
        corner.corner(self.samples, truths=self.popt,
                      quantiles=[self.percent[0] / 100, 0.5, self.percent[1] / 100],
                      show_titles=True, labels=labels, range=cornerrange)
        if savefig is not None:
            plt.savefig(savefig)
        if show:
            plt.show()
        plt.close()
        
    def posteriorongrid(self, ngrid: int = 100):
        """Calculate the posterior on a grid of ngrid x ngrid x ... x ngrid.

        Args:
            ngrid (int, optional): Number of grid on each parameter. Defaults to 100.
        """
        pargrid = [np.linspace(a, b, ngrid) for a, b in zip(*global_bounds)]
        p = np.exp(self.logl(np.meshgrid(*pargrid[::-1], indexing='ij')[::-1]))
        iopt = np.unravel_index(np.argmax(p), np.shape(p))[::-1]
        self.popt = [t[i] for t, i in zip(pargrid, iopt)]
        adim = np.arange(self.dim)
        p1d = [np.sum(p, axis=tuple(np.delete(adim, i))) for i in adim[::-1]]
        p1dcum = np.cumsum(p1d, axis=1) / np.transpose([np.sum(p1d, axis=1)])
        def getpercentile(percent: float):
            idxmin = np.argmin(np.abs(p1dcum - percent), axis=1)
            return np.array([t[i] for t, i in zip(pargrid, idxmin)])
        self.plow = getpercentile(self.percent[0] / 100)
        self.pmid = getpercentile(0.5)
        self.phigh = getpercentile(self.percent[1] / 100)
        self.p = p
        self.p1d = p1d
        self.pargrid = pargrid

    def plotongrid(self, show: bool = False, savefig: str = None,
                   labels: list = None, cornerrange: list = None,
                   cmap: str = 'binary'):
        """Make the corner plot from the posterior calculated on a grid.

        Args:
            show (bool, optional): Whether to show the corner plot. Defaults to False.
            savefig (str, optional): File name of the corner plot. Defaults to None.
            labels (list, optional): Labels for the corner plot. Defaults to None.
            cornerrange (list, optional): Range for the corner plot. Defaults to None.
            cmap: (str, optional): cmap for matplotlib.pyplot.plt.pcolormesh(). Defaults to 'binary'.
        """
        adim = np.arange(self.dim)
        if labels is None:
            labels = [f'Par {i:d}' for i in adim]
        if cornerrange is None:
            cornerrange = np.transpose(self.bounds)
        x = self.pargrid
        y = self.p1d
        fig = plt.figure(figsize=(3 * self.ndim * 1.2, 3 * self.ndim))
        for i in adim:
            for j in adim:
                if i < j:
                    continue
                ax = fig.add_subplot(self.dim, self.dim, self.dim * i + j + 1)
                if i == j:
                    ax.plot(x[i], y[i], 'k-')
                    ax.axvline(self.popt[i])
                    ax.axvline(self.plow[i], linestyle='--', color='k')
                    ax.axvline(self.pmid[i], linestyle='--', color='k')
                    ax.axvline(self.phigh[i], linestyle='--', color='k')
                    ax.set_title(f'{labels[i]}={self.pmid[i]:.2f}')
                    ax.set_xlim(cornerrange[i])
                    ax.set_yticks([])
                    if i < self.dim - 1:
                        ax.set_xticks([])
                else:
                    yy = np.sum(self.p, axis=tuple(np.delete(adim, [i, j])))
                    ax.pcolormesh(x[j], x[i], yy, cmap=cmap)
                    ax.plot(self.popt[j], self.popt[i], 'o')
                    ax.axvline(self.popt[j])
                    ax.axhline(self.popt[i])
                    ax.set_xlim(cornerrange[j])
                    ax.set_ylim(cornerrange[i])
                    if j == 0:
                        ax.set_ylabel(labels[i])
                    else:
                        ax.set_yticks([])
                    if i == self.dim - 1:
                        ax.set_xlabel(labels[j])
                    else:
                        ax.set_xticks([])
        if savefig is not None:
            plt.savefig(savefig)
        if show:
            plt.show()


