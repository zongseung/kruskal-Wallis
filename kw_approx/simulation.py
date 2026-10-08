"""
Monte Carlo Simulation for Kruskal-Wallis Statistics

Provides simulation-based estimation of tail probabilities and critical values
for the Kruskal-Wallis H statistic. This is useful for larger sample sizes
where exact computation is infeasible.

References:
- Murakami & Ha: Higher Order Asymptotic Approximations of Kruskal-Wallis Statistics
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy import stats


class MonteCarloSimulation:
    """
    Monte Carlo simulation for Kruskal-Wallis H statistic.
    
    Generates random samples under the null hypothesis (all groups from same
    distribution) and computes H statistics to estimate tail probabilities.
    
    Parameters
    ----------
    sample_sizes : List[int]
        Sample sizes for each group [n1, n2, ..., nk]
    n_simulations : int
        Number of Monte Carlo simulations (default: 10000)
    seed : Optional[int]
        Random seed for reproducibility
        
    Attributes
    ----------
    H_values : np.ndarray
        Array of simulated H statistics
        
    Examples
    --------
    >>> sim = MonteCarloSimulation([5, 5, 5], n_simulations=10000, seed=42)
    >>> sim.tail_probability(5.0)
    0.0823  # approximate
    """
    
    def __init__(self, sample_sizes: List[int], n_simulations: int = 10000,
                 seed: Optional[int] = None):
        self.sample_sizes = list(sample_sizes)
        self.k = len(sample_sizes)
        self.N = sum(sample_sizes)
        self.n_simulations = n_simulations
        self.seed = seed
        
        # Run simulation
        self._H_values = None
        self._run_simulation()
    
    def _run_simulation(self):
        """Run Monte Carlo simulation."""
        if self.seed is not None:
            np.random.seed(self.seed)
        
        H_values = np.zeros(self.n_simulations)
        
        for i in range(self.n_simulations):
            # Generate random data under null hypothesis
            # All groups from same distribution (standard normal)
            data = np.random.randn(self.N)
            
            # Compute ranks
            ranks = stats.rankdata(data)
            
            # Split into groups
            idx = 0
            rank_sums = []
            for n in self.sample_sizes:
                rank_sums.append(np.sum(ranks[idx:idx + n]))
                idx += n
            
            # Compute H statistic
            H = self._compute_H(rank_sums)
            H_values[i] = H
        
        self._H_values = np.sort(H_values)
    
    def _compute_H(self, rank_sums: List[float]) -> float:
        """Compute Kruskal-Wallis H statistic from rank sums."""
        N = self.N
        
        # H = (12 / (N(N+1))) * sum(R_i^2 / n_i) - 3(N+1)
        term = 0.0
        for R, n in zip(rank_sums, self.sample_sizes):
            term += R**2 / n
        
        H = (12.0 / (N * (N + 1))) * term - 3 * (N + 1)
        return H
    
    @property
    def H_values(self) -> np.ndarray:
        """Get sorted array of simulated H values."""
        return self._H_values
    
    def tail_probability(self, h: float) -> float:
        """
        Estimate tail probability P(H >= h) using simulation.
        
        Parameters
        ----------
        h : float
            H statistic value
            
        Returns
        -------
        float
            Estimated P(H >= h)
        """
        # Count proportion of simulated H values >= h
        return np.mean(self._H_values >= h)
    
    def cdf(self, h: float) -> float:
        """
        Estimate CDF P(H <= h) using simulation.
        
        Parameters
        ----------
        h : float
            H statistic value
            
        Returns
        -------
        float
            Estimated P(H <= h)
        """
        return 1.0 - self.tail_probability(h)
    
    def critical_value(self, alpha: float) -> Tuple[float, float]:
        """
        Estimate critical value for given significance level.
        
        Parameters
        ----------
        alpha : float
            Significance level (e.g., 0.05, 0.10)
            
        Returns
        -------
        Tuple[float, float]
            (critical_value, actual_alpha)
            actual_alpha is the proportion of H values >= critical_value
        """
        # Find the (1-alpha) quantile
        idx = int(np.ceil((1 - alpha) * self.n_simulations)) - 1
        idx = max(0, min(idx, self.n_simulations - 1))
        
        cv = self._H_values[idx]
        actual_alpha = self.tail_probability(cv)
        
        return cv, actual_alpha
    
    def percentile(self, p: float) -> float:
        """
        Get the p-th percentile of simulated H values.
        
        Parameters
        ----------
        p : float
            Percentile (0-100)
            
        Returns
        -------
        float
            p-th percentile value
        """
        return np.percentile(self._H_values, p)
    
    def summary(self) -> dict:
        """
        Get summary statistics of simulated H values.
        
        Returns
        -------
        dict
            Summary statistics
        """
        return {
            'n_simulations': self.n_simulations,
            'mean': np.mean(self._H_values),
            'std': np.std(self._H_values),
            'min': np.min(self._H_values),
            'max': np.max(self._H_values),
            'median': np.median(self._H_values),
            'q90': self.percentile(90),
            'q95': self.percentile(95),
            'q99': self.percentile(99),
        }
    
    def __repr__(self) -> str:
        return (f"MonteCarloSimulation(k={self.k}, N={self.N}, "
                f"n_simulations={self.n_simulations})")
