"""
Moments Calculation Module for Kruskal-Wallis Statistic (Improved Version)

Computes raw moments, central moments, cumulants, and related statistics
required for higher-order asymptotic approximations.

Key improvement: For small samples, uses exact distribution to compute
accurate moments. For larger samples, estimates moments by simulation
under H0 to obtain higher-order cumulants used in paper-style methods.

References:
- Murakami & Ha: Higher Order Asymptotic Approximations of Kruskal-Wallis Statistics
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from functools import lru_cache


class KWMoments:
    """
    Computes moments and cumulants of the Kruskal-Wallis H statistic.

    Under the null hypothesis, the distribution of H depends only on
    the sample sizes. This class computes exact finite-sample moments using:
    1. Exact distribution enumeration (for small N)
    2. Simulation-based estimation (for larger N)

    Note: Asymptotic chi-square-based cumulants are NOT used.
    Per paper Section 4, all approximation methods require exact
    finite-sample cumulants κ₁,...,κ₄.

    Parameters
    ----------
    sample_sizes : List[int]
        Sample sizes for each group [n1, n2, ..., nk]
    max_moment : int
        Maximum moment order to compute (default: 6)
    use_exact : bool or None
        If True, use exact distribution for moments.
        If False, use simulation-based moments.
        If None (default), auto-select based on sample size.
    n_simulations : int
        Number of simulations for moment estimation (default: 50000).
    simulation_seed : int or None
        RNG seed for simulation-based moments. If None, uses a deterministic
        seed based on sample sizes.

    Attributes
    ----------
    k : int
        Number of groups
    N : int
        Total sample size
    raw_moments : Dict[int, float]
        Dictionary of raw moments mu_H(h) for h = 1, ..., max_moment
    cumulants : Dict[int, float]
        Dictionary of cumulants kappa_h for h = 1, ..., max_moment
    """
    
    def __init__(self, sample_sizes: List[int], max_moment: int = 6,
                 use_exact: Optional[bool] = None,
                 n_simulations: int = 50000,
                 simulation_seed: Optional[int] = None):
        self.sample_sizes = np.array(sample_sizes, dtype=int)
        self.k = len(sample_sizes)
        self.N = np.sum(self.sample_sizes)
        self.max_moment = max_moment
        self.n_simulations = int(n_simulations)
        self.simulation_seed = simulation_seed
        
        # Auto-select exact vs asymptotic based on k and N
        # For k>=4, exact computation explodes faster, so use stricter limit
        if use_exact is None:
            if self.k <= 3:
                self.use_exact = self.N <= 15
            else:
                self.use_exact = self.N <= 13
        else:
            self.use_exact = use_exact
        
        # Compute and cache moments
        self._raw_moments = {}
        self._central_moments = {}
        self._cumulants = {}
        
        self._compute_moments()
    
    def _compute_moments(self):
        """Compute all moments up to max_moment order.

        Uses exact distribution for small samples and simulation otherwise.
        Asymptotic chi-square-based cumulants are NOT used (see paper Section 4:
        all methods require exact finite-sample cumulants κ₁,...,κ₄).
        """
        if self.use_exact:
            self._compute_moments_from_exact()
        else:
            self._compute_moments_from_simulation()

        # Compute central moments from raw moments
        self._compute_central_moments()

        # Compute cumulants from central moments
        self._compute_cumulants()
    
    def _compute_moments_from_exact(self):
        """Compute moments from exact distribution (accurate for small N)."""
        # Import here to avoid circular dependency
        from .exact import ExactDistribution
        
        exact = ExactDistribution(list(self.sample_sizes))
        dist = exact.distribution
        
        # Compute raw moments directly from exact distribution
        for h in range(self.max_moment + 1):
            if h == 0:
                self._raw_moments[0] = 1.0
            else:
                self._raw_moments[h] = sum(
                    H_val**h * prob for H_val, prob in dist.items()
                )

    @staticmethod
    def _deterministic_seed(sample_sizes: Tuple[int, ...],
                            max_moment: int,
                            n_simulations: int) -> int:
        """Build a deterministic seed from configuration."""
        seed = 1729 + 97 * max_moment + 3 * n_simulations
        for i, n_i in enumerate(sample_sizes):
            seed += (i + 1) * (n_i + 17) * 10007
        return int(seed % (2**32 - 1))

    @staticmethod
    @lru_cache(maxsize=256)
    def _simulate_raw_moments_cached(sample_sizes: Tuple[int, ...],
                                     max_moment: int,
                                     n_simulations: int,
                                     seed: int) -> Tuple[float, ...]:
        """
        Estimate raw moments E[H^h] by Monte Carlo under H0.
        """
        n = np.array(sample_sizes, dtype=int)
        N = int(np.sum(n))
        k = len(sample_sizes)

        if N <= 0 or k <= 1:
            return tuple([1.0] + [0.0] * max_moment)

        rng = np.random.default_rng(seed)
        factor = 12.0 / (N * (N + 1.0))
        constant = 3.0 * (N + 1.0)
        starts = np.concatenate(([0], np.cumsum(n)[:-1]))
        ends = np.cumsum(n)

        sums = np.zeros(max_moment + 1, dtype=float)
        sums[0] = float(n_simulations)

        done = 0
        target_elements = 2_000_000
        batch = max(256, min(5000, target_elements // max(N, 1)))

        while done < n_simulations:
            b = min(batch, n_simulations - done)

            # i.i.d. continuous draws induce random rank permutations
            x = rng.random((b, N))
            order = np.argsort(x, axis=1)
            ranks = np.argsort(order, axis=1) + 1

            term = np.zeros(b, dtype=float)
            for gi in range(k):
                rsum = np.sum(ranks[:, starts[gi]:ends[gi]], axis=1)
                term += (rsum**2) / n[gi]

            H = factor * term - constant
            for h in range(1, max_moment + 1):
                sums[h] += np.sum(H**h)

            done += b

        raw = [1.0]
        for h in range(1, max_moment + 1):
            raw.append(sums[h] / n_simulations)
        return tuple(raw)

    def _compute_moments_from_simulation(self):
        """Compute moments by Monte Carlo under H0."""
        sample_sizes_tuple = tuple(int(x) for x in self.sample_sizes.tolist())
        if self.simulation_seed is None:
            seed = self._deterministic_seed(sample_sizes_tuple, self.max_moment, self.n_simulations)
        else:
            seed = int(self.simulation_seed)

        raw_tuple = self._simulate_raw_moments_cached(
            sample_sizes_tuple,
            self.max_moment,
            self.n_simulations,
            seed
        )

        for h, val in enumerate(raw_tuple):
            self._raw_moments[h] = float(val)

        # enforce known exact mean under H0
        self._raw_moments[1] = float(self.k - 1)
    
    def _compute_central_moments(self):
        """Convert raw moments to central moments."""
        mu = self._raw_moments[1]
        
        self._central_moments[1] = 0.0
        self._central_moments[2] = self._raw_moments[2] - mu**2
        
        if self.max_moment >= 3 and 3 in self._raw_moments:
            # m3 = E[(X-mu)^3] = E[X^3] - 3*mu*E[X^2] + 2*mu^3
            self._central_moments[3] = (self._raw_moments[3] - 
                                        3 * mu * self._raw_moments[2] + 
                                        2 * mu**3)
        
        if self.max_moment >= 4 and 4 in self._raw_moments:
            # m4 = E[X^4] - 4*mu*E[X^3] + 6*mu^2*E[X^2] - 3*mu^4
            self._central_moments[4] = (self._raw_moments[4] - 
                                        4 * mu * self._raw_moments[3] + 
                                        6 * mu**2 * self._raw_moments[2] - 
                                        3 * mu**4)
        
        if self.max_moment >= 5 and 5 in self._raw_moments:
            self._central_moments[5] = (self._raw_moments[5] - 
                                        5 * mu * self._raw_moments[4] + 
                                        10 * mu**2 * self._raw_moments[3] - 
                                        10 * mu**3 * self._raw_moments[2] + 
                                        4 * mu**5)
        
        if self.max_moment >= 6 and 6 in self._raw_moments:
            self._central_moments[6] = (self._raw_moments[6] - 
                                        6 * mu * self._raw_moments[5] + 
                                        15 * mu**2 * self._raw_moments[4] - 
                                        20 * mu**3 * self._raw_moments[3] + 
                                        15 * mu**4 * self._raw_moments[2] - 
                                        5 * mu**6)
    
    def _compute_cumulants(self):
        """
        Convert moments to cumulants.
        
        Cumulants are related to central moments by:
        - kappa_1 = mu (mean)
        - kappa_2 = sigma^2 (variance)
        - kappa_3 = m3 (third central moment)
        - kappa_4 = m4 - 3*sigma^4
        - etc.
        """
        self._cumulants[1] = self._raw_moments[1]
        self._cumulants[2] = self._central_moments[2]
        
        if 3 in self._central_moments:
            self._cumulants[3] = self._central_moments[3]
        
        if 4 in self._central_moments:
            self._cumulants[4] = (self._central_moments[4] - 
                                  3 * self._central_moments[2]**2)
        
        if 5 in self._central_moments:
            self._cumulants[5] = (self._central_moments[5] - 
                                  10 * self._central_moments[3] * 
                                  self._central_moments[2])
        
        if 6 in self._central_moments:
            self._cumulants[6] = (self._central_moments[6] - 
                                  15 * self._central_moments[4] * self._central_moments[2] -
                                  10 * self._central_moments[3]**2 + 
                                  30 * self._central_moments[2]**3)
    
    @property
    def raw_moments(self) -> Dict[int, float]:
        """Get dictionary of raw moments."""
        return self._raw_moments.copy()
    
    @property
    def cumulants(self) -> Dict[int, float]:
        """Get dictionary of cumulants."""
        return self._cumulants.copy()
    
    def get_mean(self) -> float:
        """Get mean of H."""
        return self._raw_moments[1]
    
    def get_variance(self) -> float:
        """Get variance of H."""
        return self._central_moments[2]
    
    def get_std(self) -> float:
        """Get standard deviation of H."""
        return np.sqrt(self._central_moments[2])
    
    def get_skewness(self) -> float:
        """Get skewness of H."""
        if 3 not in self._central_moments:
            raise ValueError("Need max_moment >= 3 for skewness")
        var = self._central_moments[2]
        if var <= 0:
            return 0.0
        return self._central_moments[3] / var**(3/2)
    
    def get_kurtosis(self) -> float:
        """Get excess kurtosis of H."""
        if 4 not in self._central_moments:
            raise ValueError("Need max_moment >= 4 for kurtosis")
        var = self._central_moments[2]
        if var <= 0:
            return 0.0
        return self._central_moments[4] / var**2 - 3
    
    def get_gamma_params(self) -> Tuple[float, float]:
        """
        Get gamma distribution parameters matched to first two moments.
        
        If X ~ Gamma(alpha, beta), then:
        - E[X] = alpha * beta
        - Var[X] = alpha * beta^2
        
        Returns
        -------
        Tuple[float, float]
            (alpha, beta) parameters
        """
        mu = self._raw_moments[1]
        var = self._central_moments[2]
        
        if var <= 0 or mu <= 0:
            # Fallback to chi-square parameters
            df = self.k - 1
            return df / 2, 2.0
        
        beta = var / mu
        alpha = mu**2 / var
        
        return alpha, beta
    
    def __repr__(self) -> str:
        mode = "exact" if self.use_exact else "simulation"
        return (f"KWMoments(k={self.k}, N={self.N}, mode={mode}, "
                f"mean={self.get_mean():.4f}, var={self.get_variance():.4f})")
