"""
Unified Approximator Interface for Kruskal-Wallis Statistics

Provides a single interface to access CGF-based saddlepoint approximation
methods (ER1, ER2) with Lugannani-Rice tail probability.

References:
- Murakami & Ha: Higher Order Asymptotic Approximations of Kruskal-Wallis Statistics
"""

from typing import List, Dict, Optional
from scipy import stats
import warnings

from .kruskal_wallis import KruskalWallisStatistic
from .moments import KWMoments
from .saddlepoint import SaddlepointApproximation
from .exact import ExactDistribution
from .simulation import MonteCarloSimulation
from .edgeworth import EdgeworthApproximation, EdgeworthHallKolassa
from .gram_charlier import GramCharlierApproximation
from .pam import PolynomialAdjustedGamma


class KWApproximator:
    """
    Unified interface for Kruskal-Wallis saddlepoint approximation methods.

    Supports CGF approximations (ER1, ER2) with
    Lugannani-Rice tail probability, plus chi-square baseline and
    exact/simulation references.

    All CGF methods use exact finite-sample cumulants (not asymptotic
    chi-square cumulants), as specified in Section 4 of the paper.

    Parameters
    ----------
    sample_sizes : List[int]
        Sample sizes for each group [n1, n2, ..., nk]

    Examples
    --------
    >>> approx = KWApproximator([3, 3, 3])
    >>> approx.tail_probability(4.62, method='ER1')
    0.0933829
    >>> approx.compare_methods(4.62)
    {'chi_square': 0.0993, 'ER1': 0.0789, ...}
    """

    # CGF methods with L-R tail probability
    CGF_METHODS = [
        'ER1',
        'ER2',
    ]

    # Aliases for convenience.
    # Paper Section 4.2: SD1 = K_ER1 + Lugannani-Rice;
    #                    SD2 = gamma-based saddlepoint (Wood, Booth & Butler 1993).
    # SDC1 = SD1 + continuity correction; SDC2 = SD2 + continuity correction.
    METHOD_ALIASES = {
        'saddlepoint': 'ER1',
        'saddlepoint_sd2': 'gamma',
        'saddlepoint_cc': 'ER1_cc',
        'saddlepoint_cc2': 'gamma_cc',
        # Paper table notation
        'SD1': 'ER1',
        'SD2': 'gamma',
        'SDC1': 'ER1_cc',
        'SDC2': 'gamma_cc',
    }

    AVAILABLE_METHODS = [
        'chi_square',   # Traditional chi-square approximation (baseline)
        'ER1',          # Easton-Ronchetti 1st CGF + Lugannani-Rice
        'ER2',          # Easton-Ronchetti 2nd CGF + Lugannani-Rice
        'ER1_cc',       # ER1 with continuity correction
        'ER2_cc',       # ER2 with continuity correction
        'gamma',        # Gamma-based saddlepoint (Wood-Booth-Butler 1993) = SD2
        'gamma_cc',     # Gamma-based saddlepoint + continuity correction = SDC2
        'edgeworth',    # Edgeworth expansion approximation
        'edgeworth_hk', # Hall/Kolassa Edgeworth with asymptotic cumulants (ED-HK)
        'gram_charlier', # Gram-Charlier Type A series
        'pam',          # Polynomially adjusted gamma (degree=4); PAG(d) as 'pam<d>' (e.g. 'pam8')
        'pam6',         # Polynomially adjusted gamma (degree=6)
        'exact',        # Exact combinatorial (small samples only)
        'simulation',   # Monte Carlo simulation
    ]

    def __init__(self, sample_sizes: List[int]):
        self.sample_sizes = list(sample_sizes)
        self.k = len(sample_sizes)
        self.N = sum(sample_sizes)

        # Initialize basic calculators
        self.kw = KruskalWallisStatistic(sample_sizes)
        self.moments = KWMoments(sample_sizes, max_moment=6)

        # Lazy initialization of approximation methods
        self._methods = {}
        self._exact = None
        self._simulation = None
        self._n_simulations = 10000

    def _resolve_method(self, method: str) -> str:
        """Resolve method aliases to canonical names."""
        return self.METHOD_ALIASES.get(method, method)

    def _get_cgf_method_name(self, method: str) -> str:
        """Extract CGF method name (e.g., 'ER1' from 'ER1_cc')."""
        return method.replace('_cc', '')

    @staticmethod
    def _pam_degree(method: str):
        """Return the PAG polynomial degree for a 'pam'/'pam<d>' method string.

        'pam' -> 4 (default), 'pam6' -> 6, 'pam8' -> 8, ...; None otherwise.
        Lets any degree be requested as method='pam<d>' (e.g. 'pam10').
        """
        if method == 'pam':
            return 4
        if method.startswith('pam') and method[3:].isdigit():
            return int(method[3:])
        return None

    def _get_method(self, method: str):
        """Get or initialize an approximation method."""
        method = self._resolve_method(method)

        if method in self._methods:
            return self._methods[method]

        cgf_name = self._get_cgf_method_name(method)

        if cgf_name in self.CGF_METHODS:
            self._methods[method] = SaddlepointApproximation(
                self.sample_sizes, cgf_method=cgf_name
            )
        elif cgf_name == 'gamma':
            # SD2/SDC2: gamma-based (WBB 1993) saddlepoint. The H-scale CGF
            # is the Easton-Ronchetti ER1 polynomial; the gamma is the
            # non-normal reference base for the Lugannani-Rice mapping.
            self._methods[method] = SaddlepointApproximation(
                self.sample_sizes, cgf_method='ER1'
            )
        elif method == 'edgeworth_hk':
            self._methods[method] = EdgeworthHallKolassa(self.sample_sizes)
        elif method == 'exact':
            if self._exact is None:
                self._exact = ExactDistribution(self.sample_sizes)
            self._methods[method] = self._exact
        elif method == 'chi_square':
            self._methods[method] = None
        elif method == 'simulation':
            if self._simulation is None:
                self._simulation = MonteCarloSimulation(
                    self.sample_sizes, n_simulations=self._n_simulations
                )
            self._methods[method] = self._simulation
        elif method == 'edgeworth':
            self._methods[method] = EdgeworthApproximation(self.sample_sizes)
        elif method == 'gram_charlier':
            self._methods[method] = GramCharlierApproximation(self.sample_sizes)
        elif self._pam_degree(method) is not None:
            self._methods[method] = PolynomialAdjustedGamma(
                self.sample_sizes, degree=self._pam_degree(method)
            )
        else:
            raise ValueError(f"Unknown method: {method}. "
                           f"Available: {self.AVAILABLE_METHODS}")

        return self._methods[method]

    def tail_probability(self, h: float, method: str = 'ER1') -> float:
        """
        Compute tail probability P(H >= h) using specified method.

        Parameters
        ----------
        h : float
            H statistic value
        method : str
            Approximation method to use (default: 'ER1')

        Returns
        -------
        float
            Approximate P(H >= h)
        """
        method = self._resolve_method(method)

        if method == 'chi_square':
            return self.kw.chi_square_approximation(h)

        cgf_name = self._get_cgf_method_name(method)
        cc = method.endswith('_cc')

        if cgf_name in self.CGF_METHODS:
            sp = self._get_method(method)
            return sp.tail_probability_lr(h, continuity_correction=cc)

        elif cgf_name == 'gamma':
            sp = self._get_method(method)
            return sp.tail_probability_gamma_based(h, continuity_correction=cc)

        elif method == 'edgeworth_hk':
            ed_hk = self._get_method(method)
            return ed_hk.tail_probability(h)

        elif method == 'exact':
            exact = self._get_method(method)
            return exact.tail_probability(h)

        elif method == 'simulation':
            sim = self._get_method(method)
            return sim.tail_probability(h)

        elif method in ('edgeworth', 'gram_charlier') or self._pam_degree(method) is not None:
            obj = self._get_method(method)
            return obj.tail_probability(h)

        else:
            raise ValueError(f"Unknown method: {method}")

    def cdf(self, h: float, method: str = 'ER1') -> float:
        """
        Compute CDF P(H <= h) using specified method.

        Parameters
        ----------
        h : float
            H statistic value
        method : str
            Approximation method to use

        Returns
        -------
        float
            Approximate P(H <= h)
        """
        return 1.0 - self.tail_probability(h, method)

    def pdf(self, h: float, method: str = 'ER1') -> float:
        """
        Compute approximate PDF at h using specified method.

        Parameters
        ----------
        h : float
            H statistic value
        method : str
            Approximation method to use

        Returns
        -------
        float
            Approximate density f_H(h)
        """
        method = self._resolve_method(method)

        if method == 'chi_square':
            return stats.chi2.pdf(h, self.k - 1)

        cgf_name = self._get_cgf_method_name(method)
        cc = method.endswith('_cc')

        if cgf_name in self.CGF_METHODS:
            sp = self._get_method(method)
            return sp.density_approximation(h, continuity_correction=cc)

        elif method == 'exact':
            exact = self._get_method(method)
            return exact.pmf(h)

        elif method in ('edgeworth', 'gram_charlier') or self._pam_degree(method) is not None:
            obj = self._get_method(method)
            return obj.pdf(h)

        else:
            raise ValueError(f"Unknown method: {method}")

    def critical_value(self, alpha: float, method: str = 'ER1') -> float:
        """
        Compute critical value for significance level alpha.

        Finds c such that P(H >= c) = alpha.

        Parameters
        ----------
        alpha : float
            Significance level (e.g., 0.05, 0.10)
        method : str
            Approximation method to use

        Returns
        -------
        float
            Critical value c
        """
        method = self._resolve_method(method)

        if method == 'chi_square':
            return stats.chi2.ppf(1 - alpha, self.k - 1)

        cgf_name = self._get_cgf_method_name(method)
        cc = method.endswith('_cc')

        if cgf_name in self.CGF_METHODS:
            sp = self._get_method(method)
            return sp.critical_value(alpha, continuity_correction=cc)

        elif cgf_name == 'gamma':
            sp = self._get_method(method)
            return sp.critical_value(alpha, method='gamma', continuity_correction=cc)

        elif method == 'exact':
            exact = self._get_method(method)
            cv, exact_alpha = exact.critical_value(alpha)
            return cv

        elif method in ('edgeworth', 'gram_charlier') or self._pam_degree(method) is not None:
            obj = self._get_method(method)
            return obj.critical_value(alpha)

        else:
            raise ValueError(f"Unknown method: {method}")

    def compare_methods(self, h: float,
                       methods: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compare tail probabilities across multiple methods.

        Parameters
        ----------
        h : float
            H statistic value
        methods : List[str], optional
            Methods to compare. If None, uses chi_square + active CGF methods.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping method names to tail probabilities
        """
        if methods is None:
            methods = ['chi_square'] + self.CGF_METHODS
            if self.N <= 20:
                methods.append('exact')

        results = {}
        for method in methods:
            try:
                results[method] = self.tail_probability(h, method)
            except Exception as e:
                results[method] = None
                warnings.warn(f"Method {method} failed: {e}")

        return results

    def compare_critical_values(self, alpha: float,
                               methods: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compare critical values across multiple methods.

        Parameters
        ----------
        alpha : float
            Significance level
        methods : List[str], optional
            Methods to compare

        Returns
        -------
        Dict[str, float]
            Dictionary mapping method names to critical values
        """
        if methods is None:
            methods = ['chi_square'] + self.CGF_METHODS
            if self.N <= 20:
                methods.append('exact')

        results = {}
        for method in methods:
            try:
                results[method] = self.critical_value(alpha, method)
            except Exception as e:
                results[method] = None
                warnings.warn(f"Method {method} failed: {e}")

        return results

    def recommend_method(self) -> str:
        """
        Recommend the best approximation method based on sample sizes.

        Returns
        -------
        str
            Recommended method name
        """
        if self.N <= 15:
            return 'exact'
        else:
            return 'ER1'

    def summary(self) -> Dict:
        """
        Get summary information about the approximation setup.

        Returns
        -------
        Dict
            Summary statistics and parameters
        """
        return {
            'sample_sizes': self.sample_sizes,
            'k': self.k,
            'N': self.N,
            'df': self.k - 1,
            'mean_H': self.moments.get_mean(),
            'var_H': self.moments.get_variance(),
            'std_H': self.moments.get_std(),
            'skewness': self.moments.get_skewness(),
            'kurtosis': self.moments.get_kurtosis(),
            'cumulants': {i: self.moments.cumulants[i] for i in range(1, 5)},
            'recommended_method': self.recommend_method()
        }

    def __repr__(self) -> str:
        return (f"KWApproximator(k={self.k}, N={self.N}, "
                f"sample_sizes={self.sample_sizes})")


def quick_test(sample_sizes: List[int], h: float, alpha: float = 0.10) -> None:
    """
    Quick test function to demonstrate all CGF approximation methods.

    Parameters
    ----------
    sample_sizes : List[int]
        Sample sizes for each group
    h : float
        H statistic value to evaluate
    alpha : float
        Significance level for critical values
    """
    N = sum(sample_sizes)
    k = len(sample_sizes)

    print(f"=" * 60)
    print(f"Kruskal-Wallis Saddlepoint Approximation Comparison")
    print(f"=" * 60)
    print(f"Sample sizes: {sample_sizes}")
    print(f"k = {k}, N = {N}")
    print(f"H value: {h}")
    print(f"Significance level: {alpha}")
    print(f"-" * 60)

    approx = KWApproximator(sample_sizes)

    # Summary
    summary = approx.summary()
    print(f"\nDistribution Parameters (exact finite-sample cumulants):")
    print(f"  Mean(H):     {summary['mean_H']:.6f}")
    print(f"  Var(H):      {summary['var_H']:.6f}")
    print(f"  Skewness:    {summary['skewness']:.6f}")
    print(f"  Kurtosis:    {summary['kurtosis']:.6f}")
    for i in range(1, 5):
        print(f"  kappa_{i}:    {summary['cumulants'][i]:.6f}")

    # Tail probabilities
    print(f"\nTail Probabilities P(H >= {h}):")
    print(f"  {'Method':<20} {'P-value':>12}")
    print(f"  {'-'*20} {'-'*12}")

    probs = approx.compare_methods(h)
    for method, prob in probs.items():
        if prob is not None:
            print(f"  {method:<20} {prob:>12.6f}")
        else:
            print(f"  {method:<20} {'N/A':>12}")

    # Critical values
    print(f"\nCritical Values (alpha = {alpha}):")
    print(f"  {'Method':<20} {'Critical Value':>15}")
    print(f"  {'-'*20} {'-'*15}")

    cvs = approx.compare_critical_values(alpha)
    for method, cv in cvs.items():
        if cv is not None:
            print(f"  {method:<20} {cv:>15.6f}")
        else:
            print(f"  {method:<20} {'N/A':>15}")

    print(f"=" * 60)
