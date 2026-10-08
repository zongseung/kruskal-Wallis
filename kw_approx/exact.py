"""
Exact Distribution Computation for Kruskal-Wallis Statistic

Implements exact computation of the null distribution using:
1. Recursive algorithm from Iman et al. (1975)
2. Dynamic programming approach from van de Wiel (2002)

Note: Exact computation is only feasible for small sample sizes due to
computational complexity.

References:
- Iman, R.L., Quade, D., and Alexander, D.A. (1975). Exact probability levels 
  for the Kruskal-Wallis test.
- van de Wiel, M.A. (2002). Exact distributions of multiple comparisons rank 
  statistics.
- Murakami & Ha: Higher Order Asymptotic Approximations of Kruskal-Wallis Statistics
"""

import numpy as np
from typing import List, Tuple, Dict
from functools import lru_cache
from collections import defaultdict
import warnings
import gc


class ExactDistribution:
    """
    Exact distribution of the Kruskal-Wallis H statistic.
    
    Uses recursive algorithms to enumerate all possible rank configurations
    and compute exact probabilities under H0.
    
    WARNING: Computational complexity grows rapidly with sample size.
    For N > 20, exact computation may be infeasible.
    
    Parameters
    ----------
    sample_sizes : List[int]
        Sample sizes for each group [n1, n2, ..., nk]
    
    Attributes
    ----------
    k : int
        Number of groups
    N : int
        Total sample size
    distribution : Dict[float, float]
        Dictionary mapping H values to their exact probabilities
    """
    
    def __init__(self, sample_sizes: List[int]):
        self.sample_sizes = tuple(sample_sizes)  # Tuple for hashability
        self.k = len(sample_sizes)
        self.N = sum(self.sample_sizes)
        
        # Check feasibility
        self._check_feasibility()
        
        # Compute distribution
        self._distribution = None

    def _check_feasibility(self):
        """Check if exact computation is feasible."""
        if self.N > 15:
            if self.N <= 100:
                recommended = "pam (PAG, degree=4) or saddlepoint (SD1)"
            else:
                recommended = "saddlepoint (SD1)"

            warnings.warn(
                f"N={self.N} > 15: Consider using {recommended} instead of exact. "
                f"Exact computation may be slow or use excessive memory.",
                RuntimeWarning
            )
    
    def _compute_distribution(self) -> Dict[float, float]:
        """
        Compute the exact distribution of H using recursive algorithm.
        
        Based on Iman et al. (1975) recursive formula for rank sum counts.
        
        Returns
        -------
        Dict[float, float]
            Mapping from H values to probabilities
        """
        # Use dynamic programming to count rank sum configurations
        rank_sum_counts = self._enumerate_rank_sums()
        
        # Compute total number of configurations
        total_configs = self._multinomial_coefficient()
        
        # Convert to H values and probabilities
        distribution = defaultdict(float)
        
        for rank_sums, count in rank_sum_counts.items():
            H = self._compute_h_from_rank_sums(rank_sums)
            prob = count / total_configs
            # Round H to avoid floating point issues
            H_rounded = round(H, 10)
            distribution[H_rounded] += prob
        
        return dict(distribution)
    
    def _multinomial_coefficient(self) -> int:
        """
        Compute N! / (n1! * n2! * ... * nk!)
        
        Returns
        -------
        int
            Multinomial coefficient (total number of rank assignments)
        """
        from math import comb

        result = 1
        remaining = self.N
        for n_i in self.sample_sizes:
            result *= comb(remaining, n_i)
            remaining -= n_i
        
        return result
    
    def _compute_h_from_rank_sums(self, rank_sums: Tuple[int, ...]) -> float:
        """
        Compute H statistic from rank sums.
        
        H = (12 / (N(N+1))) * sum(R_i^2 / n_i) - 3(N+1)
        
        Parameters
        ----------
        rank_sums : Tuple[int, ...]
            Rank sums for each group
            
        Returns
        -------
        float
            H statistic value
        """
        N = self.N
        H = 0.0
        for i, R_i in enumerate(rank_sums):
            H += R_i**2 / self.sample_sizes[i]
        
        H = 12.0 / (N * (N + 1)) * H - 3 * (N + 1)
        return H
    
    def _enumerate_rank_sums(self) -> Dict[Tuple[int, ...], int]:
        """
        Enumerate all valid rank sum configurations and their counts.
        
        Uses dynamic programming based on Iman et al. (1975).
        
        Returns
        -------
        Dict[Tuple[int, ...], int]
            Mapping from rank sum tuples to their occurrence counts
        """
        n = self.sample_sizes
        k = self.k
        N = self.N
        
        # Recursive function with memoization (limited cache to prevent memory explosion)
        @lru_cache(maxsize=100000)
        def count_ways(r_tuple: Tuple[int, ...], n_tuple: Tuple[int, ...]) -> int:
            """
            Count ways to achieve rank sums r_tuple given remaining sample sizes n_tuple.
            
            Based on the recursive formula:
            W(r1,...,rk; n1,...,nk) = sum over i where n_i > 0:
                W(r1,...,r_i - N,...,rk; n1,...,n_i - 1,...,nk)
            
            where N = sum(n_tuple).
            """
            N_curr = sum(n_tuple)
            
            # Base case: all observations assigned
            if N_curr == 0:
                return 1 if all(r == 0 for r in r_tuple) else 0
            
            # Check validity
            for i, (r_i, n_i) in enumerate(zip(r_tuple, n_tuple)):
                if n_i < 0 or r_i < 0:
                    return 0
                # Minimum possible rank sum for n_i observations from ranks 1 to N_curr
                min_sum = n_i * (n_i + 1) // 2
                # Maximum possible rank sum
                max_sum = n_i * (2 * N_curr - n_i + 1) // 2
                if r_i < min_sum or r_i > max_sum:
                    return 0
            
            # Recursive case: assign the largest rank (N_curr) to one group
            total = 0
            for i in range(k):
                if n_tuple[i] > 0:
                    # Create new tuples with updated values
                    new_r = tuple(r_tuple[j] - N_curr if j == i else r_tuple[j] 
                                  for j in range(k))
                    new_n = tuple(n_tuple[j] - 1 if j == i else n_tuple[j] 
                                  for j in range(k))
                    total += count_ways(new_r, new_n)
            
            return total
        
        # Generate all valid rank sum combinations
        # R_i must be in range [n_i*(n_i+1)/2, n_i*(2N-n_i+1)/2]
        
        def generate_rank_sums(group_idx: int, current_sums: List[int], 
                               remaining_sum: int) -> List[Tuple[int, ...]]:
            """Generate all valid rank sum tuples."""
            if group_idx == k - 1:
                # Last group: rank sum is determined
                current_sums.append(remaining_sum)
                result = [tuple(current_sums)]
                current_sums.pop()
                return result
            
            n_i = n[group_idx]
            min_sum = n_i * (n_i + 1) // 2
            max_sum = n_i * (2 * N - n_i + 1) // 2
            
            # Also constrain by remaining sum needed for other groups
            remaining_n = sum(n[group_idx + 1:])
            min_remaining = remaining_n * (remaining_n + 1) // 2
            max_remaining = remaining_n * (2 * N - remaining_n + 1) // 2
            
            min_sum = max(min_sum, remaining_sum - max_remaining)
            max_sum = min(max_sum, remaining_sum - min_remaining)
            
            results = []
            for r_i in range(min_sum, max_sum + 1):
                current_sums.append(r_i)
                results.extend(generate_rank_sums(group_idx + 1, current_sums, 
                                                  remaining_sum - r_i))
                current_sums.pop()
            
            return results
        
        # Total sum of all ranks is N*(N+1)/2
        total_rank_sum = N * (N + 1) // 2
        
        # Generate all rank sum tuples
        all_rank_sums = generate_rank_sums(0, [], total_rank_sum)
        
        # Count ways for each configuration
        result = {}
        for r_tuple in all_rank_sums:
            ways = count_ways(r_tuple, tuple(n))
            if ways > 0:
                result[r_tuple] = ways

        # Clear cache to free memory
        count_ways.cache_clear()
        gc.collect()

        return result
    
    @property
    def distribution(self) -> Dict[float, float]:
        """Get the exact probability distribution."""
        if self._distribution is None:
            self._distribution = self._compute_distribution()
        return self._distribution
    
    def pmf(self, h: float) -> float:
        """
        Probability mass at H = h.
        
        Parameters
        ----------
        h : float
            H statistic value
            
        Returns
        -------
        float
            P(H = h)
        """
        h_rounded = round(h, 10)
        return self.distribution.get(h_rounded, 0.0)
    
    def cdf(self, h: float) -> float:
        """
        Cumulative distribution function P(H <= h).
        
        Parameters
        ----------
        h : float
            H statistic value
            
        Returns
        -------
        float
            P(H <= h)
        """
        return sum(prob for h_val, prob in self.distribution.items() if h_val <= h)
    
    def sf(self, h: float) -> float:
        """
        Survival function P(H >= h).
        
        Parameters
        ----------
        h : float
            H statistic value
            
        Returns
        -------
        float
            P(H >= h)
        """
        return sum(prob for h_val, prob in self.distribution.items() if h_val >= h)
    
    def tail_probability(self, h: float) -> float:
        """
        Alias for sf(). Compute P(H >= h).
        """
        return self.sf(h)
    
    def critical_value(self, alpha: float) -> Tuple[float, float]:
        """
        Find critical value for significance level alpha.
        
        Returns the smallest h such that P(H >= h) <= alpha.
        Also returns the exact significance level achieved.
        
        Parameters
        ----------
        alpha : float
            Desired significance level
            
        Returns
        -------
        Tuple[float, float]
            (critical_value, exact_alpha)
            The exact_alpha may be smaller than requested alpha
            due to discreteness of the distribution.
        """
        # Sort H values in descending order
        sorted_h = sorted(self.distribution.keys(), reverse=True)
        
        cumulative_prob = 0.0
        for h in sorted_h:
            cumulative_prob += self.distribution[h]
            if cumulative_prob >= alpha:
                return h, cumulative_prob
        
        # Return smallest H value if alpha > 1 (shouldn't happen)
        return min(sorted_h), 1.0
    
    def summary(self) -> Dict:
        """
        Compute summary statistics of the exact distribution.
        
        Returns
        -------
        Dict
            Dictionary containing mean, variance, skewness, etc.
        """
        dist = self.distribution
        
        # Mean
        mean = sum(h * p for h, p in dist.items())
        
        # Variance
        var = sum(h**2 * p for h, p in dist.items()) - mean**2
        
        # Third central moment (for skewness)
        m3 = sum((h - mean)**3 * p for h, p in dist.items())
        
        # Fourth central moment (for kurtosis)
        m4 = sum((h - mean)**4 * p for h, p in dist.items())
        
        std = np.sqrt(var)
        skewness = m3 / std**3 if std > 0 else 0
        kurtosis = m4 / var**2 - 3 if var > 0 else 0
        
        return {
            'mean': mean,
            'variance': var,
            'std': std,
            'skewness': skewness,
            'excess_kurtosis': kurtosis,
            'n_distinct_values': len(dist),
            'min_H': min(dist.keys()),
            'max_H': max(dist.keys())
        }
    
    def __repr__(self) -> str:
        n_vals = len(self.distribution) if self._distribution else "not computed"
        return (f"ExactDistribution(k={self.k}, N={self.N}, "
                f"sample_sizes={list(self.sample_sizes)}, "
                f"n_distinct_values={n_vals})")
