"""
Kruskal-Wallis Higher Order Asymptotic Approximations Package

This package implements higher-order saddlepoint approximation methods for the
Kruskal-Wallis statistic as described in:

Murakami, H., Lee, J.-S. and Ha, H.-T. "Higher Order Asymptotic Approximations
of Kruskal-Wallis Statistics Based on Skewness and Kurtosis"

Saddlepoint methods: SD1 = ER1 CGF + Lugannani-Rice; SD2 = gamma-based
saddlepoint (Wood, Booth & Butler 1993). Continuity-corrected variants
SDC1/SDC2. ER2 CGF is also available.
All methods use exact finite-sample cumulants (Section 4 of the paper).

References:
- Kruskal, W.H. & Wallis, A. (1952). Use of ranks in one-criterion variance analysis.
- Daniels, H.E. (1954). Saddlepoint approximations in statistics.
- Lugannani, R. & Rice, S.O. (1980). Saddlepoint approximation for distributions.
- Iman, R.L., Quade, D. & Alexander, D.A. (1975). Exact probability levels.
"""

from .kruskal_wallis import KruskalWallisStatistic
from .moments import KWMoments
from .saddlepoint import SaddlepointApproximation
from .pam import PolynomialAdjustedGamma
from .gram_charlier import GramCharlierApproximation
from .edgeworth import EdgeworthApproximation, EdgeworthHallKolassa, hall_kolassa_cumulants
from .exact import ExactDistribution
from .simulation import MonteCarloSimulation
from .approximator import KWApproximator

__version__ = "0.3.0"
__author__ = "Based on Murakami, Lee & Ha (2026)"

__all__ = [
    "KruskalWallisStatistic",
    "KWMoments",
    "SaddlepointApproximation",
    "PolynomialAdjustedGamma",
    "GramCharlierApproximation",
    "EdgeworthApproximation",
    "EdgeworthHallKolassa",
    "hall_kolassa_cumulants",
    "ExactDistribution",
    "MonteCarloSimulation",
    "KWApproximator",
]
