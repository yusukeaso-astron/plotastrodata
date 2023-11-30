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
                 sigma: np.ndarray = 1, progressbar: bool = True):
        """Make bounds, logl, and logp for ptemcee.

        Args:
            bounds (np.ndarray): Bounds for ptemcee in the shape of (2, dim).
            logl (function, optional): Log likelihood for ptemcee. Defaults to None.
            model (function, optional): Model function to make a log likelihood function. Defaults to None.
            xdata (np.ndarray, optional): Input for the model function. Defaults to None.
            ydata (np.ndarray, optional): Values to be compared with the model. Defaults to None.
            sigma (np.ndarray, optional): Uncertainty to make a log likelihood function from the model. Defaults to 1.
            progressbar (bool, optional): Whether to show a progress bar. Defaults to True.
        """
        global global_bounds, global_progressbar
        global_bounds = np.array(bounds) if len(bounds) < 3 else np.transpose(bounds)
        global_progressbar = progressbar
        if logl is None and not (None in [model, xdata, ydata]):
            def logl(x: np.ndarray) -> float:
                return np.sum((ydata - model(xdata, *x))**2 / sigma**2) / (-2)
        self.bounds = global_bounds
        self.logl = logl
        self.logp = logp
        self.ndata = 10000 if xdata is None else len(xdata)
    
    def fit(self, nwalkersperdim: int = 2, ntemps: int = 1, nsteps: int = 1000,
            nburnin: int = 500, ntry: int = 1, pos0: np.ndarray = None,
            percent: list = [16, 84], savechain: str = None, ncore: int = 1):
        """Perform a Markov Chain Monte Carlo (MCMC) fitting process using the ptemcee library, which is a parallel tempering version of the emcee package, and make a corner plot of the samples using the corner package.

        Args:
            nwalkersperdim (int, optional): Number of walkers per dimension. Defaults to 2.
            ntemps (int, optional): Number of temperatures. Defaults to 2.
            nsteps (int, optional): Number of steps, including the steps for burn-in. Defaults to 1000.
            nburnin (int, optional): Number of burn-in steps. Defaults to 500.
            ntry (int, optional): Number of trials for the Gelman-Rubin check. Defaults to 1.
            pos0 (np.nparray, optional): Initial parameter set in the shape of (ntemps, nwalkers, dim). Defaults to None.
            percent (list, optional): The lower and upper percnetiles to be calculated. Defaults to [16, 84].
            show (bool, optional): Whether to show the corner plot. Defaults to False.
            savefig (str, optional): File name of the corner plot. Defaults to None.
            savechain (str, optional): File name of the chain in format of .npy. Defaults to None.
            ncore (int, optional): Number of cores for multiprocessing.Pool. ncore=1 does not use multiprocessing. Defaults to 1.
        """
        global bar
        dim = len(self.bounds[0])
        nwalkers = max(nwalkersperdim, 2) * dim  # must be even and >= 2 * dim
        if ncore > 1:
            print(f'Use {ncore:d} / {cpu_count():d} CPUs')
        if global_progressbar:
            bar = tqdm(total=ntry * ntemps * nwalkers * (nsteps + 1) / ncore)
            bar.set_description('Within the ranges')

        GR = [2] * dim
        i =  0
        while np.min(GR) > 1.25 and i < ntry:
            i += 1
            if pos0 is None:
                pos0 = np.random.rand(ntemps, nwalkers, dim) \
                       * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
            pars = {'ntemps':ntemps, 'nwalkers':nwalkers, 'dim':dim,
                    'logl':self.logl, 'logp':self.logp}
            if ncore > 1:
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
            d = self.ndata - dim - 1
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
        samples = samples.reshape((-1, dim))
        lnps = lnps.reshape((-1, 1))
        self.low = np.percentile(samples, percent[0], axis=0)
        self.mid = np.percentile(samples, 50, axis=0)
        self.high = np.percentile(samples, percent[1], axis=0)
    
    def plotcorner(self, percent: list = [16, 84], show: bool = False,
                   savefig: str = None, labels: list = None,
                   cornerrange: list = None):
        """Make the corner plot from self.samples.

        Args:
            percent (list, optional): The lower and upper percnetiles to be calculated. Defaults to [16, 84].
            show (bool, optional): Whether to show the corner plot. Defaults to False.
            savefig (str, optional): File name of the corner plot. Defaults to None.
            labels (list, optional): Labels for the corner plot. Defaults to None.
            cornerrange (list, optional): Range for the corner plot. Defaults to None.
        """
        dim = np.shape(self.samples)[-1]
        if labels is None:
            labels = [f'Par {i:d}' for i in range(dim)]
        if cornerrange is None:
            cornerrange = np.transpose(self.bounds)
        corner.corner(self.samples, truths=self.popt,
                      quantiles=[percent[0] / 100, 0.5, percent[1] / 100],
                      show_titles=True, labels=labels, range=cornerrange)
        if savefig is not None:
            plt.savefig(savefig)
        if show:
            plt.show()
        plt.close()
