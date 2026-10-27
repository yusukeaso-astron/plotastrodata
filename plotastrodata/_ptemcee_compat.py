"""Parallel-tempering compatibility, loaded only when selected."""

import numpy as np
import ptemcee


class _PTSampler(ptemcee.Sampler):
    """Support ptemcee 1.0's sampler on NumPy without the np.float alias.

    Only reset and likelihood evaluation need the obsolete alias. Keep
    their behavior while using float for the sampler's float64 buffers.
    """

    def reset(self, random=None, betas=None, time=None) -> None:
        """Clear stored samples, counters, and current sampling state."""
        for name in ('_chain', '_logposterior', '_loglikelihood',
                     '_beta_history', '_p0', '_logposterior0',
                     '_loglikelihood0'):
            setattr(self, name, None)
        self._time = 0 if time is None else time
        if betas is not None:
            self._betas = betas
        if random is not None:
            self._random = random
        self.nswap = np.zeros(self.ntemps, dtype=float)
        self.nswap_accepted = np.zeros(self.ntemps, dtype=float)
        self.nprop = np.zeros((self.ntemps, self.nwalkers), dtype=float)
        self.nprop_accepted = np.zeros_like(self.nprop)

    def _evaluate(self, ps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate likelihood and prior for every temperature and walker."""
        map_function = map if self.pool is None else self.pool.map
        results = list(map_function(self._likeprior,
                                    ps.reshape((-1, self.dim))))
        values = np.asarray(results, dtype=float)
        shape = (self.ntemps, -1)
        return values[:, 0].reshape(shape), values[:, 1].reshape(shape)
