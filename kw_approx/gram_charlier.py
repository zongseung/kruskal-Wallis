"""
Gram-Charlier Series Approximation for Kruskal-Wallis Statistic

Implements the Gram-Charlier Type A series expansion for approximating
the distribution of the Kruskal-Wallis H statistic using moments.

The method expands the distribution in terms of the normal distribution
and Hermite polynomials, incorporating skewness and kurtosis corrections.

References:
- Murakami & Ha: Higher Order Asymptotic Approximations of Kruskal-Wallis Statistics
"""

import numpy as np
from typing import List
from scipy import stats
from .moments import KWMoments


class GramCharlierApproximation:
    """
    Gram-Charlier Type A series approximation for the Kruskal-Wallis H statistic.
    
    The approximate density is:
    
        f_GC(h) = φ((h-μ)/σ) * [1 + Σ_{n=1}^{∞} A_n * H_n((h-μ)/σ) / n!]
    
    where:
    - φ is the standard normal PDF
    - H_n are Hermite polynomials
    - A_n are coefficients involving moments
    
    Typically truncated to include terms for skewness (γ₁) and kurtosis (γ₂):
    
        f_GC(h) ≈ φ(z) * [1 + (γ₁/6)*H_3(z) + (γ₂/24)*H_4(z) + (γ₁²/72)*H_6(z)]
    
    where z = (h - μ) / σ.
    
    Parameters
    ----------
    sample_sizes : List[int]
        Sample sizes for each group [n1, n2, ..., nk]
    
    Attributes
    ----------
    mu : float
        Mean of H under H0
    sigma : float
        Standard deviation of H under H0
    gamma1 : float
        Skewness of H
    gamma2 : float
        Excess kurtosis of H
    """
    
    def __init__(self, sample_sizes: List[int]):
        self.sample_sizes = np.array(sample_sizes, dtype=int)
        self.k = len(sample_sizes)
        self.N = np.sum(self.sample_sizes)
        
        # Compute moments (need at least 4 for skewness and kurtosis)
        self.moments = KWMoments(sample_sizes, max_moment=6)
        
        # Extract key statistics
        self.mu = self.moments.get_mean()
        self.sigma = self.moments.get_std()
        self.gamma1 = self.moments.get_skewness()  # Skewness
        self.gamma2 = self.moments.get_kurtosis()   # Excess kurtosis
    
    @staticmethod
    def hermite_polynomial(n: int, z: float) -> float:
        """
        Compute the probabilist's Hermite polynomial H_n(z).
        
        H_0(z) = 1
        H_1(z) = z
        H_2(z) = z² - 1
        H_3(z) = z³ - 3z
        H_4(z) = z⁴ - 6z² + 3
        H_5(z) = z⁵ - 10z³ + 15z
        H_6(z) = z⁶ - 15z⁴ + 45z² - 15
        
        Parameters
        ----------
        n : int
            Order of the polynomial
        z : float
            Argument
            
        Returns
        -------
        float
            H_n(z)
        """
        if n == 0:
            return 1.0
        elif n == 1:
            return z
        elif n == 2:
            return z**2 - 1
        elif n == 3:
            return z**3 - 3*z
        elif n == 4:
            return z**4 - 6*z**2 + 3
        elif n == 5:
            return z**5 - 10*z**3 + 15*z
        elif n == 6:
            return z**6 - 15*z**4 + 45*z**2 - 15
        else:
            # Recurrence relation: H_{n+1}(z) = z*H_n(z) - n*H_{n-1}(z)
            H_prev = 1.0  # H_0
            H_curr = z    # H_1
            for i in range(1, n):
                H_next = z * H_curr - i * H_prev
                H_prev = H_curr
                H_curr = H_next
            return H_curr
    
    def standardize(self, h: float) -> float:
        """
        Standardize h to z = (h - μ) / σ.
        
        Parameters
        ----------
        h : float
            Original value
            
        Returns
        -------
        float
            Standardized value z
        """
        if self.sigma <= 0:
            return 0.0
        return (h - self.mu) / self.sigma
    
    def pdf(self, h: float) -> float:
        """
        Gram-Charlier approximation to the PDF of H.
        
        f_GC(h) ≈ φ(z)/σ * [1 + (γ₁/6)*H_3(z) + (γ₂/24)*H_4(z) + (γ₁²/72)*H_6(z)]
        
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
        
        z = self.standardize(h)
        
        # Standard normal PDF at z (need to divide by sigma for transformation)
        phi_z = stats.norm.pdf(z)
        
        # Hermite polynomial terms
        H3 = self.hermite_polynomial(3, z)
        H4 = self.hermite_polynomial(4, z)
        H6 = self.hermite_polynomial(6, z)
        
        # Gram-Charlier expansion
        correction = (1 + 
                     self.gamma1 / 6 * H3 + 
                     self.gamma2 / 24 * H4 + 
                     self.gamma1**2 / 72 * H6)
        
        density = phi_z * correction / self.sigma
        
        # Gram-Charlier can produce negative values; truncate at zero
        return max(density, 0.0)
    
    def cdf(self, h: float) -> float:
        """
        Gram-Charlier approximation to the CDF of H.
        
        F_GC(h) ≈ Φ(z) - φ(z)*[(γ₁/6)*H_2(z) + (γ₂/24)*H_3(z) + (γ₁²/72)*H_5(z)]
        
        Note: Integration of f_GC leads to this form using:
        ∫φ(z)*H_n(z)dz = -φ(z)*H_{n-1}(z)
        
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
        
        z = self.standardize(h)
        
        # Standard normal CDF and PDF
        Phi_z = stats.norm.cdf(z)
        phi_z = stats.norm.pdf(z)
        
        # Hermite polynomials (note: one degree lower due to integration)
        H2 = self.hermite_polynomial(2, z)
        H3 = self.hermite_polynomial(3, z)
        H5 = self.hermite_polynomial(5, z)
        
        # CDF with Gram-Charlier correction
        correction = phi_z * (
            self.gamma1 / 6 * H2 + 
            self.gamma2 / 24 * H3 + 
            self.gamma1**2 / 72 * H5
        )
        
        cdf_value = Phi_z - correction
        
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
            # Start from chi-square quantile as initial bracket
            c_chi2 = stats.chi2.ppf(1 - alpha, self.k - 1)
            c_star = brentq(objective, max(0.01, c_chi2 - 5), c_chi2 + 10, xtol=1e-8)
        except ValueError:
            # Fallback to chi-square critical value
            c_star = stats.chi2.ppf(1 - alpha, self.k - 1)
        
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
            Dictionary containing mu, sigma, gamma1, gamma2
        """
        return {
            'mu': self.mu,
            'sigma': self.sigma,
            'gamma1': self.gamma1,
            'gamma2': self.gamma2
        }
    
    def is_stable(self) -> bool:
        """
        Check if the Gram-Charlier approximation is likely stable.
        
        The approximation can become unstable when |γ₁| or |γ₂| are large,
        potentially producing negative densities or probabilities > 1.
        
        Returns
        -------
        bool
            True if approximation is likely stable
        """
        # Rule of thumb: approximation is stable if skewness and kurtosis
        # are not too extreme
        return abs(self.gamma1) < 2.0 and abs(self.gamma2) < 4.0
    
    def __repr__(self) -> str:
        return (f"GramCharlierApproximation(k={self.k}, N={self.N}, "
                f"mu={self.mu:.4f}, sigma={self.sigma:.4f}, "
                f"gamma1={self.gamma1:.4f}, gamma2={self.gamma2:.4f})")
