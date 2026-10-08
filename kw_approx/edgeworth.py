"""
Edgeworth Expansion Approximation for Kruskal-Wallis Statistic

Implements the chi-square based Edgeworth series expansion for approximating
the null distribution of the Kruskal-Wallis H statistic, as described in
Section 3.3 of:

"Higher Order Asymptotic Approximations of Kruskal-Wallis Statistics"
by Murakami, Lee, and Ha

The Edgeworth expansion uses the chi-square(k-1) distribution as the base
and expands corrections using generalized Laguerre polynomials, which is
the natural expansion for chi-square type statistics.

References:
- Murakami, Lee & Ha: Higher Order Asymptotic Approximations of Kruskal-Wallis Statistics
- Kotz, Johnson & Boyd: Series Representations of Distributions of Quadratic Forms
"""

import numpy as np
from typing import List
from scipy import stats
from scipy.special import gamma as gamma_func, factorial
from .moments import KWMoments


def generalized_laguerre(n: int, alpha: float, x: float) -> float:
    """
    Compute generalized Laguerre polynomial L_n^{(alpha)}(x) via recurrence.

    L_0^{(alpha)}(x) = 1
    L_1^{(alpha)}(x) = 1 + alpha - x
    (n+1) L_{n+1} = (2n + 1 + alpha - x) L_n - (n + alpha) L_{n-1}

    Parameters
    ----------
    n : int
        Degree of polynomial (>= 0)
    alpha : float
        Generalization parameter
    x : float
        Evaluation point

    Returns
    -------
    float
        L_n^{(alpha)}(x)
    """
    if n == 0:
        return 1.0
    if n == 1:
        return 1.0 + alpha - x

    L_prev = 1.0                  # L_0
    L_curr = 1.0 + alpha - x     # L_1
    for k in range(1, n):
        L_next = ((2 * k + 1 + alpha - x) * L_curr - (k + alpha) * L_prev) / (k + 1)
        L_prev = L_curr
        L_curr = L_next
    return L_curr


def laguerre_coefficients(n: int, alpha: float) -> np.ndarray:
    """
    Return the polynomial coefficients of L_n^{(alpha)}(x).

    L_n^{(alpha)}(x) = sum_{j=0}^{n} a_j * x^j

    where a_j = (-1)^j * C(n+alpha, n-j) / j!

    Parameters
    ----------
    n : int
        Degree
    alpha : float
        Generalization parameter

    Returns
    -------
    np.ndarray
        Coefficients [a_0, a_1, ..., a_n] where a_j is the coeff of x^j
    """
    coeffs = np.zeros(n + 1)
    for j in range(n + 1):
        # C(n+alpha, n-j) = Gamma(n+alpha+1) / (Gamma(n-j+1) * Gamma(alpha+j+1))
        binom = gamma_func(n + alpha + 1) / (gamma_func(n - j + 1) * gamma_func(alpha + j + 1))
        coeffs[j] = (-1)**j * binom / factorial(j, exact=True)
    return coeffs


def hall_kolassa_cumulants(sample_sizes) -> tuple:
    """
    Compute asymptotic cumulants κ3, κ4 per Hall (1992, 1993) and Kolassa (1995).

    These represent the asymptotic skewness and excess kurtosis of the
    standardized Kruskal-Wallis H statistic under H0. Paper B, Equations (13)-(14).

    Parameters
    ----------
    sample_sizes : array-like
        Sample sizes for each group [n1, n2, ..., nk]

    Returns
    -------
    tuple (kappa3, kappa4)
        kappa3 : asymptotic skewness
        kappa4 : asymptotic excess kurtosis
    """
    n = np.asarray(sample_sizes, dtype=float)
    N = np.sum(n)

    S1 = np.sum(1.0 / n) - 1.0 / N          # Σ(1/ni) - 1/N
    S2 = np.sum(1.0 / n**2) - 1.0 / N**2    # Σ(1/ni²) - 1/N²
    S3 = np.sum(1.0 / n**3) - 1.0 / N**3    # Σ(1/ni³) - 1/N³

    if S1 <= 0:
        return 0.0, 0.0

    kappa3 = S2 / S1**(3.0 / 2.0)
    kappa4 = S3 / S1**2

    return float(kappa3), float(kappa4)


class EdgeworthHallKolassa:
    """
    Conventional Edgeworth expansion using Hall/Kolassa asymptotic cumulants.

    This serves as the **comparison baseline** referenced in Paper B.
    Hall (1992, 1993) and Kolassa (1995) developed Edgeworth expansions for
    rank-based and nonparametric statistics using *asymptotic* cumulants,
    whereas the paper's main ED method uses *exact finite-sample* cumulants.

    The expansion uses a normal base distribution with Hermite polynomial
    corrections determined by κ3 (skewness) and κ4 (excess kurtosis):

        f(h) ≈ (1/σ)·φ(z)·[1 + (γ1/6)·H3(z) + (γ2/24)·H4(z) + (γ1²/72)·H6(z)]

    where z = (h - μ)/σ, γ1 = κ3, γ2 = κ4.

    Parameters
    ----------
    sample_sizes : List[int]
        Sample sizes for each group [n1, n2, ..., nk]

    References
    ----------
    - Hall, P. (1992). The Bootstrap and Edgeworth Expansion. Springer.
    - Hall, P. (1993). Edgeworth expansions for studentized statistics
      under weak assumptions. Annals of Statistics, 21, 927-951.
    - Kolassa, J.E. (1995). Edgeworth approximations for rank-sum test
      statistics. Statistics & Probability Letters, 24(2), 169-171.
    """

    def __init__(self, sample_sizes):
        self.sample_sizes = np.array(sample_sizes, dtype=int)
        self.k = len(sample_sizes)
        self.N = int(np.sum(self.sample_sizes))
        self.df = self.k - 1

        # Mean and variance from KWMoments (exact when possible)
        self._moments = KWMoments(sample_sizes, max_moment=4)
        self.mu = self._moments.get_mean()
        self.sigma = self._moments.get_std()

        # Asymptotic cumulants from Hall/Kolassa (Equations 13-14)
        self.kappa3, self.kappa4 = hall_kolassa_cumulants(sample_sizes)

        # γ1 = skewness, γ2 = excess kurtosis (of the standardized variable)
        self.gamma1 = self.kappa3
        self.gamma2 = self.kappa4

    def pdf(self, h: float) -> float:
        """
        Hall/Kolassa Edgeworth density approximation.

        f(h) ≈ (1/σ)·φ(z)·[1 + (γ1/6)·H3(z) + (γ2/24)·H4(z) + (γ1²/72)·H6(z)]
        """
        if h <= 0 or self.sigma <= 0:
            return 0.0

        z = (h - self.mu) / self.sigma
        phi_z = stats.norm.pdf(z)

        # Hermite polynomials
        H3 = z**3 - 3 * z
        H4 = z**4 - 6 * z**2 + 3
        H6 = z**6 - 15 * z**4 + 45 * z**2 - 15

        correction = (1.0
                      + self.gamma1 / 6.0 * H3
                      + self.gamma2 / 24.0 * H4
                      + self.gamma1**2 / 72.0 * H6)

        density = phi_z / self.sigma * correction
        return max(density, 0.0)

    def cdf(self, h: float) -> float:
        """
        Hall/Kolassa Edgeworth CDF approximation.

        F(h) ≈ Φ(z) - φ(z)·[(γ1/6)·He2(z) + (γ2/24)·He3(z) + (γ1²/72)·He5(z)]

        where He_n are probabilists' Hermite polynomials used in the
        integrated Edgeworth series.
        """
        if h <= 0:
            return 0.0
        if self.sigma <= 0:
            return 1.0 if h >= self.mu else 0.0

        z = (h - self.mu) / self.sigma
        Phi_z = stats.norm.cdf(z)
        phi_z = stats.norm.pdf(z)

        # Integrated Hermite polynomial terms
        He2 = z**2 - 1
        He3 = z**3 - 3 * z
        He5 = z**5 - 10 * z**3 + 15 * z

        correction = phi_z * (
            self.gamma1 / 6.0 * He2
            + self.gamma2 / 24.0 * He3
            + self.gamma1**2 / 72.0 * He5
        )

        return float(np.clip(Phi_z - correction, 0.0, 1.0))

    def sf(self, h: float) -> float:
        """Survival function P(H >= h)."""
        return 1.0 - self.cdf(h)

    def tail_probability(self, h: float) -> float:
        """Alias for sf(). Compute P(H >= h)."""
        return self.sf(h)

    def critical_value(self, alpha: float) -> float:
        """Compute critical value c such that P(H >= c) = alpha."""
        from scipy.optimize import brentq

        def objective(c):
            return self.sf(c) - alpha

        try:
            c_chi2 = stats.chi2.ppf(1 - alpha, self.df)
            c_star = brentq(objective, max(0.01, c_chi2 - 5), c_chi2 + 10, xtol=1e-8)
        except ValueError:
            c_star = stats.chi2.ppf(1 - alpha, self.df)

        return c_star

    def get_parameters(self) -> dict:
        """Get all approximation parameters."""
        return {
            'mu': self.mu,
            'sigma': self.sigma,
            'kappa3_asymp': self.kappa3,
            'kappa4_asymp': self.kappa4,
            'gamma1': self.gamma1,
            'gamma2': self.gamma2,
            'df': self.df,
            'N': self.N,
        }

    def __repr__(self) -> str:
        return (f"EdgeworthHallKolassa(k={self.k}, N={self.N}, "
                f"mu={self.mu:.4f}, sigma={self.sigma:.4f}, "
                f"kappa3={self.kappa3:.4f}, kappa4={self.kappa4:.4f})")


class EdgeworthApproximation:
    """
    Chi-square based Edgeworth expansion for the Kruskal-Wallis H statistic.

    Uses chi-square(p) as the base distribution with generalized Laguerre
    polynomial corrections:

        f(h) = chi2_pdf(h, p) * [1 + sum_{n=1}^M c_n * L_n^{(alpha)}(h/2)]

        F(h) = chi2_cdf(h, p) + chi2_pdf(h, p) * sum_{n=1}^M c_n * L_{n-1}^{(alpha+1)}(h/2)

    where alpha = p/2 - 1 and p = k - 1 (degrees of freedom).

    Parameters
    ----------
    sample_sizes : List[int]
        Sample sizes for each group [n1, n2, ..., nk]
    max_terms : int
        Number of correction terms M (default: 4, uses up to 4th raw moment)
    """

    def __init__(self, sample_sizes: List[int], max_terms: int = 4):
        self.sample_sizes = np.array(sample_sizes, dtype=int)
        self.k = len(sample_sizes)
        self.N = np.sum(self.sample_sizes)
        self.df = self.k - 1  # p = k - 1
        self.max_terms = max_terms

        # alpha for Laguerre polynomials: alpha = p/2 - 1
        self.alpha = self.df / 2.0 - 1.0

        # Compute moments (need raw moments up to max_terms)
        self.moments = KWMoments(sample_sizes, max_moment=max(max_terms + 2, 6))

        # Extract key statistics (for compatibility and reporting)
        self.mu = self.moments.get_mean()
        self.sigma = self.moments.get_std()
        self.lambda3 = self.moments.get_skewness()
        self.lambda4 = self.moments.get_kurtosis()

        # Compute Laguerre expansion coefficients
        self._coefficients = self._compute_coefficients()

    def _compute_coefficients(self) -> np.ndarray:
        """
        Compute expansion coefficients c_n from raw moments.

        c_n = (n! / Gamma(n + alpha + 1)) * sum_{j=0}^{n} a_{n,j} * mu'_j

        where:
        - a_{n,j} are polynomial coefficients of L_n^{(alpha)}
        - mu'_j = E[H^j] / 2^j  (scaled raw moments, since chi2 scale = 2)
        - c_0 = 1 by construction

        Returns
        -------
        np.ndarray
            Coefficients c_0, c_1, ..., c_M
        """
        M = self.max_terms
        alpha = self.alpha
        raw = self.moments.raw_moments

        # Scaled raw moments: mu'_j = E[H^j] / 2^j
        # (scaling because chi-square has scale beta=2, and Laguerre
        #  orthogonality is w.r.t. x^alpha * exp(-x), so x = h/2)
        scaled_moments = {}
        for j in range(M + 1):
            if j in raw:
                scaled_moments[j] = raw[j] / (2.0 ** j)
            else:
                # Fallback: use chi-square moment
                # E_chi2[X^j] = 2^j * Gamma(p/2 + j) / Gamma(p/2)
                chi2_raw_j = (2.0 ** j) * gamma_func(self.df / 2.0 + j) / gamma_func(self.df / 2.0)
                scaled_moments[j] = chi2_raw_j / (2.0 ** j)

        coeffs = np.zeros(M + 1)
        coeffs[0] = 1.0  # Normalization

        for n in range(1, M + 1):
            # Get polynomial coefficients of L_n^{(alpha)}
            lag_coeffs = laguerre_coefficients(n, alpha)

            # c_n = (n! / Gamma(n + alpha + 1)) * sum_j a_{n,j} * mu'_j
            norm = factorial(n, exact=True) / gamma_func(n + alpha + 1)

            total = 0.0
            for j in range(n + 1):
                if j in scaled_moments:
                    total += lag_coeffs[j] * scaled_moments[j]

            coeffs[n] = norm * total

        return coeffs

    def pdf(self, h: float) -> float:
        """
        Chi-square based Edgeworth density approximation.

        f(h) = chi2_pdf(h, p) * [1 + sum_{n=1}^M c_n * L_n^{(alpha)}(h/2)]

        Parameters
        ----------
        h : float
            Point at which to evaluate density

        Returns
        -------
        float
            Approximate density value
        """
        if h <= 0:
            return 0.0

        base_pdf = stats.chi2.pdf(h, self.df)
        x = h / 2.0  # Scaled argument for Laguerre polynomials

        correction = 1.0
        for n in range(1, self.max_terms + 1):
            correction += self._coefficients[n] * generalized_laguerre(n, self.alpha, x)

        density = base_pdf * correction
        return max(density, 0.0)

    def cdf(self, h: float) -> float:
        """
        Chi-square based Edgeworth CDF approximation.

        Using the identity d/dx L_n^{(alpha)}(x) = -L_{n-1}^{(alpha+1)}(x):

        F(h) = chi2_cdf(h, p) + chi2_pdf(h, p) * sum_{n=1}^M c_n * L_{n-1}^{(alpha+1)}(h/2)

        Parameters
        ----------
        h : float
            Point at which to evaluate CDF

        Returns
        -------
        float
            Approximate P(H <= h)
        """
        if h <= 0:
            return 0.0

        base_cdf = stats.chi2.cdf(h, self.df)
        base_pdf = stats.chi2.pdf(h, self.df)
        x = h / 2.0

        correction = 0.0
        for n in range(1, self.max_terms + 1):
            # L_{n-1}^{(alpha+1)}(x) — note alpha is shifted by +1
            correction += self._coefficients[n] * generalized_laguerre(n - 1, self.alpha + 1, x)

        cdf_value = base_cdf + base_pdf * correction

        return np.clip(cdf_value, 0.0, 1.0)

    def sf(self, h: float) -> float:
        """
        Survival function (tail probability) P(H >= h).

        Parameters
        ----------
        h : float
            Critical value

        Returns
        -------
        float
            Approximate P(H >= h)
        """
        return 1.0 - self.cdf(h)

    def tail_probability(self, h: float) -> float:
        """
        Alias for sf(). Compute P(H >= h).

        Parameters
        ----------
        h : float
            Critical value

        Returns
        -------
        float
            Approximate P(H >= h)
        """
        return self.sf(h)

    def cdf_normal_based(self, h: float) -> float:
        """
        Normal-based Edgeworth expansion (legacy, same as GC-A).

        Preserved for comparison. Uses Hermite polynomials with normal base.

        Parameters
        ----------
        h : float
            Point at which to evaluate CDF

        Returns
        -------
        float
            Approximate P(H <= h)
        """
        if h <= 0:
            return 0.0

        z = (h - self.mu) / self.sigma if self.sigma > 0 else 0.0

        Phi_z = stats.norm.cdf(z)
        phi_z = stats.norm.pdf(z)

        He2 = z**2 - 1
        He3 = z**3 - 3 * z
        He5 = z**5 - 10 * z**3 + 15 * z

        correction = phi_z * (
            self.lambda3 / 6 * He2 +
            self.lambda4 / 24 * He3 +
            self.lambda3**2 / 72 * He5
        )

        return np.clip(Phi_z - correction, 0.0, 1.0)

    def critical_value(self, alpha: float) -> float:
        """
        Compute critical value for significance level alpha.

        Uses numerical root finding to solve P(H >= c) = alpha.

        Parameters
        ----------
        alpha : float
            Significance level (e.g., 0.05 or 0.10)

        Returns
        -------
        float
            Critical value c
        """
        from scipy.optimize import brentq

        def objective(c):
            return self.sf(c) - alpha

        try:
            c_chi2 = stats.chi2.ppf(1 - alpha, self.df)
            c_star = brentq(objective, max(0.01, c_chi2 - 5), c_chi2 + 10, xtol=1e-8)
        except ValueError:
            c_star = stats.chi2.ppf(1 - alpha, self.df)

        return c_star

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse CDF).

        Parameters
        ----------
        q : float
            Probability (between 0 and 1)

        Returns
        -------
        float
            Value h such that P(H <= h) = q
        """
        return self.critical_value(1 - q)

    def get_parameters(self) -> dict:
        """
        Get all approximation parameters.

        Returns
        -------
        dict
            Dictionary containing mu, sigma, lambda3, lambda4, etc.
        """
        return {
            'mu': self.mu,
            'sigma': self.sigma,
            'lambda3': self.lambda3,
            'lambda4': self.lambda4,
            'alpha': self.alpha,
            'df': self.df,
            'N': self.N,
            'coefficients': self._coefficients.tolist()
        }

    def is_stable(self) -> bool:
        """
        Check if the Edgeworth approximation is likely stable.

        Returns
        -------
        bool
            True if approximation is likely stable
        """
        return (abs(self.lambda3) < 2.0 and
                abs(self.lambda4) < 6.0 and
                self.N >= 9)

    def __repr__(self) -> str:
        return (f"EdgeworthApproximation(k={self.k}, N={self.N}, "
                f"df={self.df}, mu={self.mu:.4f}, sigma={self.sigma:.4f}, "
                f"lambda3={self.lambda3:.4f}, lambda4={self.lambda4:.4f})")
