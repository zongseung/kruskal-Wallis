"""
Test Module for Kruskal-Wallis Approximations

Tests the implementation against known values from the paper:
Murakami & Ha: Higher Order Asymptotic Approximations of Kruskal-Wallis Statistics
"""

import numpy as np
import pytest
from typing import List, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kw_approx import (
    KruskalWallisStatistic,
    KWMoments,
    SaddlepointApproximation,
    PolynomialAdjustedGamma,
    GramCharlierApproximation,
    EdgeworthApproximation,
    ExactDistribution,
    KWApproximator
)


class TestKruskalWallisStatistic:
    """Test the basic Kruskal-Wallis statistic computation."""
    
    def test_statistic_computation(self):
        """Test H statistic computation from rank sums."""
        # Example: 3 groups with equal sizes
        sample_sizes = [3, 3, 3]
        kw = KruskalWallisStatistic(sample_sizes)
        
        # N = 9, ranks 1-9
        # If R1=6, R2=15, R3=24 (sum = 45 = 9*10/2 ✓)
        rank_sums = [6, 15, 24]
        H = kw.compute_from_rank_sums(rank_sums)
        
        # Manual calculation:
        # H = (12/(9*10)) * (36/3 + 225/3 + 576/3) - 3*10
        # H = (12/90) * (12 + 75 + 192) - 30
        # H = 0.1333 * 279 - 30 = 37.2 - 30 = 7.2
        expected = 7.2
        assert np.isclose(H, expected, rtol=1e-6), f"Expected {expected}, got {H}"
    
    def test_mean_equals_df(self):
        """Test that E[H] = k-1 under H0."""
        for k in [3, 4, 5]:
            sample_sizes = [5] * k
            kw = KruskalWallisStatistic(sample_sizes)
            assert kw.mean() == k - 1
    
    def test_chi_square_approximation(self):
        """Test chi-square p-value computation."""
        sample_sizes = [5, 5, 5]
        kw = KruskalWallisStatistic(sample_sizes)
        
        # For H = 5.991, chi-square(2) should give p ≈ 0.05
        from scipy import stats
        H = stats.chi2.ppf(0.95, 2)
        p = kw.chi_square_approximation(H)
        assert np.isclose(p, 0.05, rtol=1e-3)


class TestKWMoments:
    """Test moment calculations."""
    
    def test_mean(self):
        """Test that mean = k - 1."""
        moments = KWMoments([3, 3, 3])
        assert np.isclose(moments.get_mean(), 2.0, rtol=1e-6)

        moments = KWMoments([5, 5, 5, 5])
        assert np.isclose(moments.get_mean(), 3.0, rtol=1e-6)
    
    def test_variance_positive(self):
        """Test that variance is positive."""
        for sizes in [[3, 3, 3], [5, 5, 5], [3, 4, 5]]:
            moments = KWMoments(sizes)
            assert moments.get_variance() > 0
    
    def test_gamma_params(self):
        """Test gamma parameter estimation."""
        moments = KWMoments([5, 5, 5])
        alpha, beta = moments.get_gamma_params()
        
        # Check that alpha * beta = mean
        assert np.isclose(alpha * beta, moments.get_mean(), rtol=1e-6)
        
        # Check that alpha * beta^2 = variance
        assert np.isclose(alpha * beta**2, moments.get_variance(), rtol=1e-3)
    
    def test_skewness_positive(self):
        """Test that skewness is positive (right-skewed like chi-square)."""
        moments = KWMoments([5, 5, 5])
        assert moments.get_skewness() > 0


class TestSaddlepointApproximation:
    """Test saddlepoint approximation methods."""
    
    def test_tail_probability_bounds(self):
        """Test that tail probabilities are in [0, 1]."""
        sp = SaddlepointApproximation([3, 3, 3])
        
        for h in [1.0, 2.0, 4.0, 6.0, 8.0]:
            p = sp.tail_probability_lr(h)
            assert 0 <= p <= 1, f"P(H >= {h}) = {p} out of bounds"
    
    def test_tail_probability_monotonic(self):
        """Test that P(H >= h) decreases as h increases."""
        sp = SaddlepointApproximation([5, 5, 5])
        
        h_values = [2.0, 4.0, 6.0, 8.0, 10.0]
        probs = [sp.tail_probability_lr(h) for h in h_values]
        
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i+1], \
                f"Not monotonic: P(H >= {h_values[i]}) = {probs[i]} < P(H >= {h_values[i+1]}) = {probs[i+1]}"
    
    def test_continuity_correction(self):
        """Test that continuity correction changes results."""
        sp = SaddlepointApproximation([3, 3, 3])
        
        h = 4.62
        p_no_cc = sp.tail_probability_lr(h, continuity_correction=False)
        p_cc = sp.tail_probability_lr(h, continuity_correction=True)
        
        # They should be different
        assert p_no_cc != p_cc

    def test_cgf_origin_derivatives_match_h_cumulants(self):
        """Test that H-scale polynomial CGFs have the right mean and variance."""
        for method in ['ER1', 'ER2']:
            sp = SaddlepointApproximation([3, 3, 3], cgf_method=method)

            assert np.isclose(sp.cgf_derivative1(0.0), sp.moments.get_mean())
            assert np.isclose(sp.cgf_derivative2(0.0), sp.moments.get_variance())


class TestPolynomialAdjustedGamma:
    """Test PAM approximation."""
    
    def test_pdf_integrates_to_one(self):
        """Test that PDF approximately integrates to 1."""
        pam = PolynomialAdjustedGamma([5, 5, 5], degree=4)
        
        # Numerical integration
        from scipy import integrate
        integral, _ = integrate.quad(pam.pdf, 0, 50)
        
        assert np.isclose(integral, 1.0, rtol=0.05), \
            f"PDF integral = {integral}, expected 1.0"
    
    def test_cdf_bounds(self):
        """Test that CDF is in [0, 1]."""
        pam = PolynomialAdjustedGamma([3, 3, 3], degree=6)
        
        for h in [0.0, 1.0, 2.0, 5.0, 10.0]:
            cdf = pam.cdf(h)
            assert 0 <= cdf <= 1, f"CDF({h}) = {cdf} out of bounds"
    
    def test_cdf_monotonic(self):
        """Test that CDF is monotonically increasing."""
        pam = PolynomialAdjustedGamma([5, 5, 5], degree=4)
        
        h_values = np.linspace(0.1, 15, 50)
        cdfs = [pam.cdf(h) for h in h_values]
        
        for i in range(len(cdfs) - 1):
            assert cdfs[i] <= cdfs[i+1] + 1e-10, \
                f"CDF not monotonic at h={h_values[i]}"
    
    def test_higher_degree_different(self):
        """Test that different polynomial degrees give different results."""
        pam4 = PolynomialAdjustedGamma([3, 3, 3], degree=4)
        pam6 = PolynomialAdjustedGamma([3, 3, 3], degree=6)
        
        h = 4.62
        assert pam4.tail_probability(h) != pam6.tail_probability(h)


class TestGramCharlierApproximation:
    """Test Gram-Charlier approximation."""
    
    def test_hermite_polynomials(self):
        """Test Hermite polynomial computation."""
        gc = GramCharlierApproximation([5, 5, 5])
        
        # H_3(0) = 0
        assert gc.hermite_polynomial(3, 0) == 0
        
        # H_4(0) = 3
        assert gc.hermite_polynomial(4, 0) == 3
        
        # H_6(0) = -15
        assert gc.hermite_polynomial(6, 0) == -15
        
        # H_3(1) = 1 - 3 = -2
        assert gc.hermite_polynomial(3, 1) == -2
    
    def test_pdf_nonnegative(self):
        """Test that PDF is non-negative (may fail for extreme cases)."""
        gc = GramCharlierApproximation([5, 5, 5])
        
        # In the reasonable range
        for h in np.linspace(0.5, 10, 20):
            pdf = gc.pdf(h)
            assert pdf >= 0, f"PDF({h}) = {pdf} < 0"
    
    def test_stability_check(self):
        """Test stability indicator."""
        gc = GramCharlierApproximation([5, 5, 5])
        # For moderate samples, should be stable
        assert gc.is_stable()


class TestEdgeworthApproximation:
    """Test Edgeworth expansion approximation (chi-square based, Laguerre)."""

    def test_tail_probability_bounds(self):
        """Test that tail probabilities are in [0, 1]."""
        ed = EdgeworthApproximation([5, 5, 5])

        for h in [1.0, 2.0, 4.0, 6.0, 8.0]:
            p = ed.tail_probability(h)
            assert 0 <= p <= 1, f"P(H >= {h}) = {p} out of bounds"

    def test_cdf_monotonic(self):
        """Test that CDF is monotonically increasing."""
        ed = EdgeworthApproximation([5, 5, 5])

        h_values = np.linspace(0.5, 10, 20)
        cdfs = [ed.cdf(h) for h in h_values]

        for i in range(len(cdfs) - 1):
            assert cdfs[i] <= cdfs[i+1] + 1e-6, \
                f"CDF not monotonic at h={h_values[i]}"

    def test_cdf_monotonic_small_sample(self):
        """Test CDF monotonicity for small samples (3,3,3)."""
        ed = EdgeworthApproximation([3, 3, 3])

        h_values = np.linspace(0.5, 10, 30)
        cdfs = [ed.cdf(h) for h in h_values]

        for i in range(len(cdfs) - 1):
            assert cdfs[i] <= cdfs[i+1] + 1e-6, \
                f"CDF not monotonic at h={h_values[i]}"

    def test_stability_check(self):
        """Test stability indicator."""
        ed = EdgeworthApproximation([5, 5, 5])
        # For moderate samples, should be stable
        assert ed.is_stable()

    def test_parameters(self):
        """Test parameter extraction."""
        ed = EdgeworthApproximation([5, 5, 5])
        params = ed.get_parameters()

        assert 'mu' in params
        assert 'sigma' in params
        assert 'lambda3' in params
        assert 'lambda4' in params
        assert params['df'] == 2
        assert 'coefficients' in params
        assert 'alpha' in params

    def test_ed_differs_from_gc_a(self):
        """Test that ED (chi-square Laguerre) differs from GC-A (normal Hermite)."""
        ed = EdgeworthApproximation([3, 3, 3])
        gc = GramCharlierApproximation([3, 3, 3])

        h = 4.62222
        ed_p = ed.tail_probability(h)
        gc_p = gc.tail_probability(h)

        # They must be different (this was the original bug)
        assert abs(ed_p - gc_p) > 0.001, \
            f"ED ({ed_p:.6f}) should differ from GC-A ({gc_p:.6f})"

    def test_ed_normal_based_matches_gc_a(self):
        """Test that the legacy normal-based ED matches GC-A."""
        ed = EdgeworthApproximation([3, 3, 3])
        gc = GramCharlierApproximation([3, 3, 3])

        h = 4.62222
        ed_normal = 1.0 - ed.cdf_normal_based(h)
        gc_p = gc.tail_probability(h)

        assert np.isclose(ed_normal, gc_p, rtol=1e-4), \
            f"Normal-based ED ({ed_normal:.6f}) should match GC-A ({gc_p:.6f})"

    def test_ed_near_paper_value_333(self):
        """Test ED for (3,3,3) H=4.62222 is near paper value 0.090."""
        ed = EdgeworthApproximation([3, 3, 3])
        p = ed.tail_probability(4.62222)

        # Paper value is 0.090, allow reasonable tolerance
        assert 0.05 < p < 0.15, \
            f"ED P-value {p:.6f} far from expected ~0.090"

    def test_ed_four_groups(self):
        """Test ED for (3,2,2,5) produces reasonable results."""
        ed = EdgeworthApproximation([3, 2, 2, 5])
        p = ed.tail_probability(5.587179)

        assert 0 < p < 1, f"ED P-value {p:.6f} out of bounds"
        # Paper value is 0.112
        assert 0.05 < p < 0.20, \
            f"ED P-value {p:.6f} far from expected ~0.112"


class TestExactDistribution:
    """Test exact distribution computation."""

    def test_small_sample(self):
        """Test exact computation for very small sample."""
        exact = ExactDistribution([2, 2, 2])

        # Total probability should be 1
        total = sum(exact.distribution.values())
        assert np.isclose(total, 1.0, rtol=1e-10), \
            f"Total probability = {total}, expected 1.0"
    
    def test_probabilities_positive(self):
        """Test that all probabilities are positive."""
        exact = ExactDistribution([2, 2, 2])
        
        for h, p in exact.distribution.items():
            assert p > 0, f"P(H = {h}) = {p} <= 0"
    
    def test_cdf_at_max(self):
        """Test that CDF at maximum H is 1."""
        exact = ExactDistribution([2, 2, 2])
        max_h = max(exact.distribution.keys())
        
        assert np.isclose(exact.cdf(max_h), 1.0, rtol=1e-10)
    
    def test_sf_plus_cdf(self):
        """Test that SF(h) + CDF(h) - PMF(h) = 1."""
        exact = ExactDistribution([2, 2, 2])
        
        for h in exact.distribution.keys():
            # P(H < h) + P(H > h) + P(H = h) = 1
            # CDF(h) includes P(H=h), SF(h) includes P(H=h)
            # So CDF(h) + SF(h) - PMF(h) = 1
            result = exact.cdf(h) + exact.sf(h) - exact.pmf(h)
            assert np.isclose(result, 1.0, rtol=1e-10)


class TestKWApproximator:
    """Test the unified approximator interface."""
    
    def test_all_methods_work(self):
        """Test that all methods can be called without error."""
        approx = KWApproximator([3, 3, 3])
        h = 4.0

        for method in ['chi_square', 'saddlepoint',
                       'saddlepoint_sd2',   # SD2 = gamma-based saddlepoint (WBB 1993)
                       'saddlepoint_cc',
                       'saddlepoint_cc2',   # SDC2 = SD2 + continuity correction
                       'SD2', 'SDC2',
                       'pam', 'pam6', 'gram_charlier', 'edgeworth', 'exact']:
            try:
                p = approx.tail_probability(h, method)
                assert 0 <= p <= 1, f"{method}: P = {p} out of bounds"
            except Exception as e:
                pytest.fail(f"Method {method} failed: {e}")
    
    def test_compare_methods(self):
        """Test method comparison."""
        approx = KWApproximator([3, 3, 3])
        results = approx.compare_methods(4.62)
        
        # Should have results for chi-square, active CGF methods, and references.
        assert len(results) >= 4
        
        # All values should be probabilities
        for method, p in results.items():
            if p is not None:
                assert 0 <= p <= 1
    
    def test_summary(self):
        """Test summary output."""
        approx = KWApproximator([5, 5, 5])
        summary = approx.summary()
        
        assert 'mean_H' in summary
        assert 'var_H' in summary
        assert 'recommended_method' in summary
        assert summary['k'] == 3
        assert summary['N'] == 15
    
    def test_recommendation(self):
        """Test method recommendation."""
        # Small sample -> exact
        approx = KWApproximator([3, 3, 3])
        assert approx.recommend_method() == 'exact'

        # Medium sample -> ER1 (saddlepoint)
        approx = KWApproximator([8, 8, 8])
        assert approx.recommend_method() == 'ER1'

        # Large sample -> ER1 (saddlepoint)
        approx = KWApproximator([50, 50, 50])
        assert approx.recommend_method() == 'ER1'


class TestPaperExamples:
    """Test against specific examples from the paper."""
    
    def test_table_4_1_three_groups(self):
        """
        Test against Table 4.1 from the paper.
        
        n1=3, n2=3, n3=3, H=4.62222, 10% significance level
        
        Paper values (approximate):
        - E-P: 0.0842833
        - SD1: 0.0789061
        - PAM: 0.098122
        - PAM(6): 0.0933829
        """
        approx = KWApproximator([3, 3, 3])
        h = 4.62222
        
        # Get approximations
        results = approx.compare_methods(h)
        
        # Check PAM(6) - should be close to paper value
        pam6_p = results.get('pam6')
        if pam6_p is not None:
            # Allow some tolerance due to implementation differences
            assert 0.05 < pam6_p < 0.15, \
                f"PAM(6) P-value {pam6_p} far from expected ~0.0934"
        
        # Check that exact distribution can be computed
        exact_p = results.get('exact')
        if exact_p is not None:
            assert 0.05 < exact_p < 0.15, \
                f"Exact P-value {exact_p} unexpected"
    
    def test_four_groups(self):
        """
        Test four groups case from Table 4.4.
        
        n1=3, n2=2, n3=2, n4=5
        """
        approx = KWApproximator([3, 2, 2, 5])
        
        # Should be able to compute all approximations
        results = approx.compare_methods(5.0)
        
        # All should be valid probabilities
        for method, p in results.items():
            if p is not None:
                assert 0 <= p <= 1


def run_quick_test():
    """Run a quick demonstration test."""
    from kw_approx.approximator import quick_test
    
    print("\n" + "="*60)
    print("QUICK TEST: 3 Groups (n1=3, n2=3, n3=3)")
    print("="*60)
    quick_test([3, 3, 3], h=4.62222, alpha=0.10)
    
    print("\n" + "="*60)
    print("QUICK TEST: 4 Groups (n1=3, n2=2, n3=2, n4=5)")
    print("="*60)
    quick_test([3, 2, 2, 5], h=5.587179, alpha=0.10)


if __name__ == '__main__':
    # Run quick demonstration
    run_quick_test()
    
    print("\n" + "="*60)
    print("Running pytest...")
    print("="*60)
    
    # Run pytest
    pytest.main([__file__, '-v', '--tb=short'])
