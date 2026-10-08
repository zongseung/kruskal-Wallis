"""
Polynomially Adjusted Gamma Method (PAM) for Kruskal-Wallis Statistic

Implements the semi-parametric density approximation method using a gamma 
baseline distribution with polynomial adjustments, as described in:

- Ha, H-T. and Provost, S.B. (2007). A viable alternative to resorting to 
  statistical tables.
- Provost, S.B., Jiang, M. and Ha, H-T. (2009). Moment-based approximations 
  of probability mass functions with applications involving order statistics.

References:
- Murakami & Ha: Higher Order Asymptotic Approximations of Kruskal-Wallis Statistics
"""

import numpy as np
import warnings
from typing import List, Tuple
from scipy import stats
from scipy.special import gammainc
from scipy.linalg import solve
from .moments import KWMoments


class PolynomialAdjustedGamma:
    """
    Polynomially Adjusted Gamma approximation for the Kruskal-Wallis H statistic.
    
    The approximate density is:
    
        f_H(x; d) = ψ(x) * Σ_{i=0}^{d} ξ_i * x^i
    
    where ψ(x) is the gamma baseline density and ξ_i are polynomial coefficients
    derived from moment matching.
    
    Parameters
    ----------
    sample_sizes : List[int]
        Sample sizes for each group [n1, n2, ..., nk]
    degree : int
        Degree of polynomial adjustment (default: 6)
        Higher degrees provide better approximations but may be unstable.
    
    Attributes
    ----------
    alpha : float
        Shape parameter of the baseline gamma distribution
    beta : float
        Scale parameter of the baseline gamma distribution
    xi : np.ndarray
        Polynomial coefficients
    """
    
    def __init__(self, sample_sizes: List[int], degree: int = 6):
        self.sample_sizes = np.array(sample_sizes, dtype=int)
        self.k = len(sample_sizes)
        self.N = np.sum(self.sample_sizes)
        self.degree = degree
        
        # Need moments up to degree d for the polynomial adjustment
        self.moments = KWMoments(sample_sizes, max_moment=max(degree + 2, 6))
        
        # Estimate gamma parameters from first two moments
        self.alpha, self.beta = self._estimate_gamma_params()
        
        # Compute polynomial coefficients
        self.xi = self._compute_polynomial_coefficients()
    
    def _estimate_gamma_params(self) -> Tuple[float, float]:
        """
        Estimate gamma distribution parameters from moments.
        
        If X ~ Gamma(α, β), then:
        - E[X] = α * β
        - Var[X] = α * β²
        
        Returns
        -------
        Tuple[float, float]
            (alpha, beta) - shape and scale parameters
        """
        mu = self.moments.get_mean()
        var = self.moments.get_variance()
        
        # Solve: mu = α*β, var = α*β²
        # => β = var/mu, α = mu²/var
        
        if var <= 0 or mu <= 0:
            # Fallback to chi-square parameters
            df = self.k - 1
            return df / 2, 2.0
        
        beta = var / mu
        alpha = mu**2 / var
        
        return alpha, beta
    
    def _gamma_baseline_moment(self, j: int) -> float:
        """
        Compute j-th raw moment of the gamma baseline distribution.
        
        m(j) = β^j * Γ(α + j) / Γ(α) = β^j * Π_{i=1}^{j} (α + j - i)
        
        Parameters
        ----------
        j : int
            Moment order
            
        Returns
        -------
        float
            E[X^j] for X ~ Gamma(α, β)
        """
        if j == 0:
            return 1.0
        
        # Use product formula for numerical stability
        result = self.beta ** j
        for i in range(1, j + 1):
            result *= (self.alpha + j - i)
        
        return result
    
    def _build_moment_matrix(self) -> np.ndarray:
        """
        Build the moment matrix M for polynomial coefficient estimation.
        
        Each row h (0 to d) contains moments m(h), m(h+1), ..., m(h+d).
        
        Returns
        -------
        np.ndarray
            (d+1) x (d+1) moment matrix
        """
        d = self.degree
        M = np.zeros((d + 1, d + 1))
        
        for h in range(d + 1):
            for i in range(d + 1):
                M[h, i] = self._gamma_baseline_moment(h + i)
        
        return M
    
    def _compute_polynomial_coefficients(self) -> np.ndarray:
        """
        Compute polynomial coefficients ξ_i via moment matching.
        
        Solves: (ξ_0, ..., ξ_d)' = M^{-1} * (μ_H(0), ..., μ_H(d))'
        
        Returns
        -------
        np.ndarray
            Polynomial coefficients
        """
        d = self.degree
        
        # Build moment matrix
        M = self._build_moment_matrix()
        
        # Get target moments of H
        mu_H = np.zeros(d + 1)
        mu_H[0] = 1.0  # μ_H(0) = 1 (normalization)
        
        raw_moments = self.moments.raw_moments
        for h in range(1, d + 1):
            if h in raw_moments:
                mu_H[h] = raw_moments[h]
            else:
                # Estimate higher moments using gamma approximation
                mu_H[h] = self._gamma_baseline_moment(h)
        
        # Warn about ill-conditioning but use direct solve which
        # works well enough for typical PAM moment matrices (cond ~1e12)
        cond = np.linalg.cond(M)
        if cond > 1e14:
            warnings.warn(
                f"PAM moment matrix severely ill-conditioned (cond={cond:.1e}, degree={d}). "
                f"Consider reducing polynomial degree.",
                RuntimeWarning
            )

        try:
            xi = solve(M, mu_H)
        except np.linalg.LinAlgError:
            # If matrix is singular, use pseudo-inverse
            xi = np.linalg.lstsq(M, mu_H, rcond=None)[0]

        return xi
    
    def gamma_pdf(self, x: float) -> float:
        """
        Gamma baseline density function.
        
        ψ(x) = (1 / (Γ(α) * β^α)) * x^{α-1} * exp(-x/β)
        
        Parameters
        ----------
        x : float
            Point at which to evaluate density
            
        Returns
        -------
        float
            Gamma density value
        """
        if x <= 0:
            return 0.0
        return stats.gamma.pdf(x, self.alpha, scale=self.beta)
    
    def pdf(self, x: float) -> float:
        """
        Polynomially adjusted gamma density approximation.
        
        f_H(x; d) = ψ(x) * Σ_{i=0}^{d} ξ_i * x^i
        
        Parameters
        ----------
        x : float
            Point at which to evaluate density
            
        Returns
        -------
        float
            Approximate density value
        """
        if x <= 0:
            return 0.0
        
        # Gamma baseline
        psi = self.gamma_pdf(x)
        
        # Polynomial adjustment
        poly_sum = np.sum([self.xi[i] * x**i for i in range(len(self.xi))])
        
        density = psi * poly_sum
        
        # Ensure non-negative
        return max(density, 0.0)
    
    def _incomplete_gamma_integral(self, i: int, c: float) -> float:
        """
        Compute the integral ∫_0^{c} x^i * ψ(x) dx.
        
        This equals (1/Γ(α)) * β^i * γ(i+α, c/β)
        
        where γ(a, x) is the lower incomplete gamma function.
        
        Parameters
        ----------
        i : int
            Power of x
        c : float
            Upper integration limit
            
        Returns
        -------
        float
            Integral value
        """
        if c <= 0:
            return 0.0
        
        # Using scipy's regularized incomplete gamma function
        # gammainc(a, x) = γ(a, x) / Γ(a)
        
        # ∫_0^c x^i ψ(x) dx = β^i * Γ(i+α) / Γ(α) * gammainc(i+α, c/β)
        
        a = self.alpha + i
        x = c / self.beta
        
        # Γ(i+α) / Γ(α) = Π_{j=0}^{i-1} (α + j)
        ratio = 1.0
        for j in range(i):
            ratio *= (self.alpha + j)
        
        result = self.beta**i * ratio * gammainc(a, x)
        
        return result
    
    def cdf(self, c: float) -> float:
        """
        Approximate CDF of the Kruskal-Wallis H statistic.
        
        F_H(c; d) = Σ_{i=0}^{d} ξ_i * ∫_0^{c} x^i * ψ(x) dx
        
        Parameters
        ----------
        c : float
            Point at which to evaluate CDF
            
        Returns
        -------
        float
            Approximate P(H <= c)
        """
        if c <= 0:
            return 0.0
        
        cdf_value = 0.0
        for i in range(len(self.xi)):
            cdf_value += self.xi[i] * self._incomplete_gamma_integral(i, c)
        
        return np.clip(cdf_value, 0.0, 1.0)
    
    def sf(self, c: float) -> float:
        """
        Survival function (tail probability) P(H >= c).
        
        Parameters
        ----------
        c : float
            Critical value
            
        Returns
        -------
        float
            Approximate P(H >= c)
        """
        return 1.0 - self.cdf(c)
    
    def tail_probability(self, c: float) -> float:
        """
        Alias for sf(). Compute P(H >= c).
        
        Parameters
        ----------
        c : float
            Critical value
            
        Returns
        -------
        float
            Approximate P(H >= c)
        """
        return self.sf(c)
    
    def critical_value(self, alpha: float) -> float:
        """
        Compute critical value for significance level alpha.
        
        Finds c such that P(H >= c) = alpha.
        
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
        
        # Search for critical value
        try:
            c_star = brentq(objective, 0.01, 50, xtol=1e-8)
        except ValueError:
            # Fallback to gamma critical value
            c_star = stats.gamma.ppf(1 - alpha, self.alpha, scale=self.beta)
        
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
            Value x such that P(H <= x) = q
        """
        return self.critical_value(1 - q)
    
    def get_parameters(self) -> dict:
        """
        Get all approximation parameters.
        
        Returns
        -------
        dict
            Dictionary containing alpha, beta, and polynomial coefficients
        """
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'degree': self.degree,
            'xi': self.xi.tolist()
        }
    
    def __repr__(self) -> str:
        return (f"PolynomialAdjustedGamma(k={self.k}, N={self.N}, "
                f"degree={self.degree}, alpha={self.alpha:.4f}, beta={self.beta:.4f})")
