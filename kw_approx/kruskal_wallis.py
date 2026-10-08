"""
Kruskal-Wallis Statistic Computation Module

Implements the Kruskal-Wallis H statistic and related computations.
"""

import numpy as np
from typing import List
from scipy import stats


class KruskalWallisStatistic:
    """
    Computes the Kruskal-Wallis H statistic and provides utilities.
    
    The Kruskal-Wallis test statistic H is given by:
    
        H = (12 / (N(N+1))) * sum(R_i^2 / n_i) - 3(N+1)
    
    where:
        - N = total number of observations
        - n_i = sample size of group i
        - R_i = sum of ranks for group i
    
    Parameters
    ----------
    sample_sizes : List[int]
        List of sample sizes for each group [n1, n2, ..., nk]
    
    Attributes
    ----------
    k : int
        Number of groups
    N : int
        Total sample size
    sample_sizes : np.ndarray
        Array of sample sizes
    df : int
        Degrees of freedom (k - 1)
    """
    
    def __init__(self, sample_sizes: List[int]):
        """
        Initialize the Kruskal-Wallis statistic calculator.
        
        Parameters
        ----------
        sample_sizes : List[int]
            Sample sizes for each group
        """
        self.sample_sizes = np.array(sample_sizes, dtype=int)
        self.k = len(sample_sizes)
        self.N = np.sum(self.sample_sizes)
        self.df = self.k - 1
        
        # Validate inputs
        if self.k < 2:
            raise ValueError("At least 2 groups are required")
        if np.any(self.sample_sizes < 1):
            raise ValueError("All sample sizes must be at least 1")
    
    def compute_statistic(self, *groups: np.ndarray) -> float:
        """
        Compute the Kruskal-Wallis H statistic from data.
        
        Parameters
        ----------
        *groups : np.ndarray
            Variable number of arrays, one for each group
            
        Returns
        -------
        float
            The Kruskal-Wallis H statistic
        """
        if len(groups) != self.k:
            raise ValueError(f"Expected {self.k} groups, got {len(groups)}")
        
        # Combine all data and compute ranks
        all_data = np.concatenate(groups)
        ranks = stats.rankdata(all_data)
        
        # Split ranks back into groups
        idx = 0
        rank_sums = []
        for n in self.sample_sizes:
            rank_sums.append(np.sum(ranks[idx:idx + n]))
            idx += n
        
        return self._compute_h_from_rank_sums(rank_sums)
    
    def _compute_h_from_rank_sums(self, rank_sums: List[float]) -> float:
        """
        Compute H statistic from rank sums.
        
        Parameters
        ----------
        rank_sums : List[float]
            Sum of ranks for each group
            
        Returns
        -------
        float
            The H statistic
        """
        rank_sums = np.array(rank_sums)
        N = self.N
        
        H = (12.0 / (N * (N + 1))) * np.sum(rank_sums**2 / self.sample_sizes) - 3 * (N + 1)
        
        return H
    
    def compute_from_rank_sums(self, rank_sums: List[float]) -> float:
        """
        Public method to compute H from rank sums directly.
        
        Parameters
        ----------
        rank_sums : List[float]
            Sum of ranks for each group
            
        Returns
        -------
        float
            The H statistic
        """
        return self._compute_h_from_rank_sums(rank_sums)
    
    def mean(self) -> float:
        """
        Compute the expected value (mean) of H under H0.
        
        E[H] = k - 1 (degrees of freedom)
        
        Returns
        -------
        float
            Expected value of H
        """
        return float(self.k - 1)
    
    def chi_square_approximation(self, h_value: float) -> float:
        """
        Compute p-value using chi-square approximation.
        
        This is the traditional large-sample approximation where
        H ~ chi^2(k-1) under H0.
        
        Parameters
        ----------
        h_value : float
            Observed H statistic
            
        Returns
        -------
        float
            P-value P(H >= h_value)
        """
        return 1 - stats.chi2.cdf(h_value, self.df)
    
    def __repr__(self) -> str:
        return (f"KruskalWallisStatistic(k={self.k}, N={self.N}, "
                f"sample_sizes={list(self.sample_sizes)})")
