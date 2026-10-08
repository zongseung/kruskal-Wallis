"""
Saddlepoint Approximation Module for Kruskal-Wallis Statistic

Implements various saddlepoint approximation methods:
- Daniels' saddlepoint density approximation (SD1, SD2)
- Lugannani-Rice tail probability approximation
- Continuity-corrected versions (SDC1, SDC2)
- Gamma-based saddlepoint approximation (Wood et al., 1993)

References:
- Daniels, H.E. (1954). Saddlepoint approximations in statistics
- Lugannani, R. and Rice, S.O. (1980). Saddlepoint approximation for the 
  distribution of the sum of independent random variables
- Murakami & Ha: Higher Order Asymptotic Approximations of Kruskal-Wallis Statistics
"""

import numpy as np
from typing import List
from scipy import stats
from scipy.optimize import brentq, newton
from .moments import KWMoments


class SaddlepointApproximation:
    """
    Saddlepoint approximations for the Kruskal-Wallis H statistic.
    
    Parameters
    ----------
    sample_sizes : List[int]
        Sample sizes for each group [n1, n2, ..., nk]
    cgf_method : str
        Method for cumulant generating function approximation.
        Options: 'exact', 'ER1', 'ER2'
        Default: 'ER1' (Easton-Ronchetti first approximation)
    
    Attributes
    ----------
    moments : KWMoments
        Moment calculator object
    """

    SUPPORTED_CGF = ('ER1', 'ER2', 'exact')

    def __init__(self, sample_sizes: List[int], cgf_method: str = 'ER1'):
        self.sample_sizes = np.array(sample_sizes, dtype=int)
        self.k = len(sample_sizes)
        self.N = np.sum(self.sample_sizes)

        # Compute moments (need at least 4 for saddlepoint)
        self.moments = KWMoments(sample_sizes, max_moment=6)

        if cgf_method not in self.SUPPORTED_CGF:
            raise ValueError(
                f"Unknown cgf_method {cgf_method!r}; "
                f"supported: {self.SUPPORTED_CGF}")
        self.cgf_method = cgf_method

        # Cache cumulants for CGF computations
        self._kappa = self.moments.cumulants

    def cumulant_generating_function(self, t: float) -> float:
        """
        Compute the cumulant generating function K_H(t).
        
        Uses approximations based on cumulants since exact CGF
        is complex for the Kruskal-Wallis statistic.
        
        Parameters
        ----------
        t : float
            Argument of the CGF
            
        Returns
        -------
        float
            K_H(t)
        """
        kappa = self._kappa
        
        if self.cgf_method == 'ER1':
            # Easton-Ronchetti first approximation
            # K_H(t) ≈ sum_{i=1}^{4} kappa_i * t^i / i!
            return (kappa[1] * t + 
                    kappa[2] * t**2 / 2 + 
                    kappa[3] * t**3 / 6 + 
                    kappa[4] * t**4 / 24)
        
        elif self.cgf_method == 'ER2':
            # Easton-Ronchetti second approximation
            # K_H(t) ≈ kappa_1*t + (kappa_2/2)*t^2 + 
            #          log(1 + (kappa_3/6)*t^3 + (3*kappa_4/72)*t^4 + (kappa_3^2/72)*t^6)
            inner = (1 + kappa[3] * t**3 / 6 + 
                     3 * kappa[4] * t**4 / 72 + 
                     kappa[3]**2 * t**6 / 72)
            if inner > 0:
                return (kappa[1] * t + kappa[2] * t**2 / 2 + np.log(inner))
            else:
                # Fallback to ER1 if log argument becomes non-positive
                return (kappa[1] * t + 
                        kappa[2] * t**2 / 2 + 
                        kappa[3] * t**3 / 6 + 
                        kappa[4] * t**4 / 24)
        
        else:
            # Default to ER1
            return (kappa[1] * t +
                    kappa[2] * t**2 / 2 + 
                    kappa[3] * t**3 / 6 + 
                    kappa[4] * t**4 / 24)
    
    def cgf_derivative1(self, t: float) -> float:
        """
        First derivative of CGF: K'_H(t).
        
        Parameters
        ----------
        t : float
            Argument
            
        Returns
        -------
        float
            K'_H(t)
        """
        kappa = self._kappa
        
        if self.cgf_method == 'ER1':
            return (kappa[1] + 
                    kappa[2] * t + 
                    kappa[3] * t**2 / 2 + 
                    kappa[4] * t**3 / 6)
        
        elif self.cgf_method == 'ER2':
            # Derivative of ER2 approximation
            inner = (1 + kappa[3] * t**3 / 6 + 
                     3 * kappa[4] * t**4 / 72 + 
                     kappa[3]**2 * t**6 / 72)
            d_inner = (kappa[3] * t**2 / 2 + 
                       kappa[4] * t**3 / 6 + 
                       kappa[3]**2 * t**5 / 12)
            if inner > 0:
                return kappa[1] + kappa[2] * t + d_inner / inner
            else:
                return (kappa[1] + 
                        kappa[2] * t + 
                        kappa[3] * t**2 / 2 + 
                        kappa[4] * t**3 / 6)
        
        else:
            return (kappa[1] +
                    kappa[2] * t +
                    kappa[3] * t**2 / 2 +
                    kappa[4] * t**3 / 6)
    
    def cgf_derivative2(self, t: float) -> float:
        """
        Second derivative of CGF: K''_H(t).
        
        Parameters
        ----------
        t : float
            Argument
            
        Returns
        -------
        float
            K''_H(t)
        """
        kappa = self._kappa
        
        if self.cgf_method in ['ER1', 'exact']:
            return kappa[2] + kappa[3] * t + kappa[4] * t**2 / 2
        else:
            # Numerical derivative as fallback
            h = 1e-6
            return (self.cgf_derivative1(t + h) - self.cgf_derivative1(t - h)) / (2 * h)
    
    def cgf_derivative3(self, t: float) -> float:
        """Third derivative of CGF: K'''_H(t)."""
        kappa = self._kappa

        if self.cgf_method in ['ER1', 'exact']:
            return kappa[3] + kappa[4] * t
        else:
            # Numerical derivative for ER2 and other methods
            h = 1e-6
            return (self.cgf_derivative2(t + h) - self.cgf_derivative2(t - h)) / (2 * h)
    
    def find_saddlepoint(self, x: float) -> float:
        """
        Find the saddlepoint t_hat that solves K'_H(t) = x.

        Uses dynamic bracket expansion with sign constraints to ensure
        the correct root is found (t_hat > 0 when x > mean, t_hat < 0
        when x < mean).

        Parameters
        ----------
        x : float
            Target value (typically the observed H statistic)

        Returns
        -------
        float
            The saddlepoint t_hat
        """
        def equation(t):
            return self.cgf_derivative1(t) - x

        mean = self.moments.get_mean()

        if abs(x - mean) < 1e-10:
            return 0.0

        # Sign constraint: x > mean => t_hat > 0, x < mean => t_hat < 0
        if x > mean:
            lo, hi = 0.0, 1.0
        else:
            lo, hi = -1.0, 0.0

        # Dynamic bracket expansion until sign change is found
        for _ in range(60):
            try:
                f_lo = equation(lo)
                f_hi = equation(hi)
                if (np.isfinite(f_lo) and np.isfinite(f_hi) and
                        np.sign(f_lo) != np.sign(f_hi)):
                    t_hat = brentq(equation, lo, hi, xtol=1e-12)
                    return t_hat
            except (ValueError, OverflowError):
                pass
            # Expand bracket outward
            if x > mean:
                hi *= 2.0
                if hi > 200:
                    break
            else:
                lo *= 2.0
                if lo < -200:
                    break

        # Newton fallback with sign-appropriate initial guess
        try:
            t0 = 0.1 if x > mean else -0.1
            t_hat = newton(equation, t0, fprime=self.cgf_derivative2,
                          tol=1e-10, maxiter=200)
            return t_hat
        except (RuntimeError, ValueError):
            return 0.0
    
    def density_approximation(self, x: float, continuity_correction: bool = False) -> float:
        """
        Daniels' saddlepoint density approximation.

        f_SP(x) = (2π K''_H(t_hat))^{-1/2} * exp(K_H(t_hat) - x*t_hat)

        Parameters
        ----------
        x : float
            Point at which to evaluate density
        continuity_correction : bool
            Whether to apply continuity correction for discrete distributions

        Returns
        -------
        float
            Approximate density f_H(x)
        """
        if continuity_correction:
            # Continuity correction for discrete distributions.
            #
            # The paper (Section 3.2, p.16) writes "replace x by x + 1/2",
            # but this is inconsistent with:
            #   (a) Standard CC convention: P(H >= v) ~ P_cont(H >= v - 0.5)
            #       (Butler, 2007, "Saddlepoint Approximations with Applications")
            #   (b) The paper's own Table 4.1: SDC1=0.124 > E-P=0.10, which
            #       matches v - 0.5 (increases tail prob), not v + 0.5.
            # Conclusion: the paper text likely has a sign typo. We use v - 0.5.
            x = x - 0.5

        t_hat = self.find_saddlepoint(x)

        K_t = self.cumulant_generating_function(t_hat)
        K2_t = self.cgf_derivative2(t_hat)

        if K2_t <= 0:
            return 0.0

        density = np.sqrt(1 / (2 * np.pi * K2_t)) * np.exp(K_t - x * t_hat)
        
        return max(density, 0.0)
    
    def tail_probability_lr(self, v: float, continuity_correction: bool = False) -> float:
        """
        Lugannani-Rice saddlepoint approximation for tail probability.

        P(H >= v) ≈ 1 - Φ(w_hat) + φ(w_hat) * (1/u_hat - 1/w_hat)

        where:
        - w_hat = sign(t_hat) * sqrt(2*(t_hat*v - K_H(t_hat)))
        - u_hat = t_hat * sqrt(K''_H(t_hat))

        For discrete distributions, continuity correction adjusts v to
        better approximate the discrete tail probability.

        Parameters
        ----------
        v : float
            Critical value
        continuity_correction : bool
            Whether to apply continuity correction (shifts v by -0.5
            for upper-tail P(H >= v) of discrete distributions)

        Returns
        -------
        float
            Approximate P(H >= v)
        """
        if continuity_correction:
            # Continuity correction for discrete distributions.
            # See detailed note in density_approximation() above.
            # Paper text says v + 0.5 but standard CC and paper tables
            # both confirm v - 0.5 is correct.
            v = v - 0.5

        t_hat = self.find_saddlepoint(v)

        # Special case: v = mean (t_hat = 0)
        if np.abs(t_hat) < 1e-10:
            K2_0 = self.cgf_derivative2(0)
            K3_0 = self.cgf_derivative3(0)

            term = (K3_0 * K2_0**(-3/2) / 6 - K2_0**(-1/2) / 2) / np.sqrt(2 * np.pi)
            return 0.5 - term

        K_t = self.cumulant_generating_function(t_hat)
        K2_t = self.cgf_derivative2(t_hat)

        if K2_t <= 0:
            # Fallback to chi-square
            from .kruskal_wallis import KruskalWallisStatistic
            kw = KruskalWallisStatistic(list(self.sample_sizes))
            return kw.chi_square_approximation(v)

        # Compute w_hat and u_hat
        arg_w = 2 * (t_hat * v - K_t)
        if arg_w < 0:
            # Can happen due to numerical issues
            arg_w = max(arg_w, 0)

        w_hat = np.sign(t_hat) * np.sqrt(arg_w)
        u_hat = t_hat * np.sqrt(K2_t)

        # Lugannani-Rice formula
        Phi_w = stats.norm.cdf(w_hat)
        phi_w = stats.norm.pdf(w_hat)

        if np.abs(u_hat) < 1e-10 or np.abs(w_hat) < 1e-10:
            return 0.5

        prob = 1 - Phi_w + phi_w * (1/u_hat - 1/w_hat)

        return np.clip(prob, 0, 1)
    
    def tail_probability_gamma_based(self, v: float,
                                     continuity_correction: bool = False) -> float:
        """
        Gamma-based saddlepoint approximation (Wood, Booth & Butler, 1993).

        General non-normal-base Lugannani-Rice approximation: the H-scale
        saddlepoint is mapped onto a Gamma(alpha, beta) reference (matched to
        the first two moments of H) by CGF-exponent matching:
            K_G(t_xi) - t_xi * xi  =  K_H(t_hat) - t_hat * v,
        with t_xi the gamma-scale saddlepoint solving K_G'(t_xi) = xi. With
        u_xi = t_hat * sqrt(K_H''(t_hat) / K_G''(t_xi)) the tail is
            P(H >= v) ~ 1 - G(xi) + g(xi) * (1/u_xi - 1/t_xi).
        Reduces to the standard normal-base LR when the base is normal.

        Parameters
        ----------
        v : float
            Critical value
        continuity_correction : bool
            Whether to apply continuity correction (shifts v by -0.5 for
            upper-tail P(H >= v) of discrete distributions; same convention
            as tail_probability_lr).

        Returns
        -------
        float
            Approximate P(H >= v)
        """
        if continuity_correction:
            # See note in tail_probability_lr(): upper-tail discrete CC uses v - 0.5.
            v = v - 0.5

        # Get gamma parameters matched to first two moments
        alpha, beta = self.moments.get_gamma_params()
        
        t_hat = self.find_saddlepoint(v)
        
        K_t = self.cumulant_generating_function(t_hat)
        K2_t = self.cgf_derivative2(t_hat)
        
        if K2_t <= 0:
            return stats.gamma.sf(v, alpha, scale=beta)
        
        target = K_t - v * t_hat

        # For Gamma(alpha,beta):
        # G(t) = -alpha*log(1-beta*t), G'(t)=xi => t_xi = 1/beta - alpha/xi
        # Matching equation:
        # G(t_xi) - xi*t_xi = target
        # -> F(xi)=alpha*log(xi/(alpha*beta)) - xi/beta + alpha - target = 0
        def f_xi(xi: float) -> float:
            if xi <= 0:
                return -np.inf
            return alpha * np.log(xi / (alpha * beta)) - xi / beta + alpha - target

        mean_g = alpha * beta
        xi_hat = mean_g
        found = False

        # Right-side root (upper tail)
        lo = max(mean_g * (1.0 + 1e-8), 1e-12)
        hi = max(mean_g * 2.0, 1.0)
        flo = f_xi(lo)
        fhi = f_xi(hi)
        for _ in range(80):
            if np.isfinite(flo) and np.isfinite(fhi) and np.sign(flo) != np.sign(fhi):
                xi_hat = brentq(f_xi, lo, hi, xtol=1e-10)
                found = True
                break
            hi *= 2.0
            fhi = f_xi(hi)

        if not found:
            # Left-side root fallback
            lo = max(1e-12, mean_g * 1e-8)
            hi = max(mean_g * (1.0 - 1e-8), lo * 2.0)
            flo = f_xi(lo)
            fhi = f_xi(hi)
            for _ in range(80):
                if np.isfinite(flo) and np.isfinite(fhi) and np.sign(flo) != np.sign(fhi):
                    xi_hat = brentq(f_xi, lo, hi, xtol=1e-10)
                    found = True
                    break
                lo = max(lo / 2.0, 1e-12)
                flo = f_xi(lo)

        if not np.isfinite(xi_hat) or xi_hat <= 0:
            return stats.gamma.sf(v, alpha, scale=beta)

        t_xi = 1.0 / beta - alpha / xi_hat
        g_pdf = stats.gamma.pdf(xi_hat, alpha, scale=beta)
        g_cdf = stats.gamma.cdf(xi_hat, alpha, scale=beta)

        # G''(t_xi) for gamma CGF
        denom = 1.0 - beta * t_xi
        if abs(denom) < 1e-12:
            return np.clip(1.0 - g_cdf, 0.0, 1.0)
        G2_txi = alpha * beta**2 / (denom**2)
        if G2_txi <= 0:
            return np.clip(1.0 - g_cdf, 0.0, 1.0)

        u_hat_xi = t_hat * np.sqrt(K2_t / G2_txi)
        if np.abs(u_hat_xi) < 1e-10 or np.abs(t_xi) < 1e-10:
            return np.clip(1.0 - g_cdf, 0.0, 1.0)

        prob = 1.0 - g_cdf + g_pdf * (1.0 / u_hat_xi - 1.0 / t_xi)
        return np.clip(prob, 0.0, 1.0)
    
    def critical_value(self, alpha: float, method: str = 'LR',
                       continuity_correction: bool = False) -> float:
        """
        Compute critical value for significance level alpha.
        
        Finds c such that P(H >= c) = alpha.
        
        Parameters
        ----------
        alpha : float
            Significance level (e.g., 0.05 or 0.10)
        method : str
            Approximation method
        continuity_correction : bool
            Whether to apply continuity correction
            
        Returns
        -------
        float
            Critical value c
        """
        def objective(c):
            if method == 'LR':
                return self.tail_probability_lr(c, continuity_correction) - alpha
            else:
                return self.tail_probability_gamma_based(c, continuity_correction) - alpha
        
        # Search for critical value
        # H is non-negative, and critical values are typically in range [0, 20]
        try:
            c_star = brentq(objective, 0.01, 50, xtol=1e-8)
        except ValueError:
            # Fallback to chi-square critical value
            c_star = stats.chi2.ppf(1 - alpha, self.k - 1)
        
        return c_star
    
    def __repr__(self) -> str:
        return (f"SaddlepointApproximation(k={self.k}, N={self.N}, "
                f"method='{self.cgf_method}')")
