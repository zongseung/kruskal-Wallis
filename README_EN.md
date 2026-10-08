**English** | [**Korean**](README.md)

# kw-approx: Higher Order Asymptotic Approximations for Kruskal-Wallis Statistics

A Python package implementing higher-order asymptotic approximations to the null distribution and critical values of the Kruskal-Wallis test statistic.

## Paper

> **Lee, J.-S., Murakami, H., & Ha, H.-T. (2026).** *Higher Order Asymptotic Approximations of Kruskal-Wallis Statistics Based on Skewness and Kurtosis.* Preprint submitted to Journals.

## Background

The Kruskal-Wallis test is a rank-based, distribution-free alternative to one-way ANOVA for testing the equality of location parameters across $k$ treatment groups.

### Kruskal-Wallis H Statistic

$$H = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1)$$

- $N = \sum_{i=1}^{k} n_i$: total sample size
- $n_i$: sample size of group $i$
- $R_i$: rank sum of group $i$

The conventional $\chi^2_{k-1}$ approximation can be inaccurate for small or unbalanced samples. This package provides saddlepoint, Edgeworth, Gram-Charlier, and polynomially adjusted gamma approximations that incorporate exact skewness and kurtosis corrections.

## Project Structure

```
KS-test/
|
|-- kw_approx/                    # Core package
|   |-- __init__.py               #   Package init (v0.3.0)
|   |-- kruskal_wallis.py         #   H statistic computation
|   |-- moments.py                #   Moments/cumulants (exact/simulation)
|   |-- cumulants_closed_form.py  #   Closed-form k3/k4 (Thm 3.12, no enumeration)
|   |-- saddlepoint.py            #   Saddlepoint approx (ER1/ER2 + L-R, WBB gamma-base = SD2)
|   |-- edgeworth.py              #   Edgeworth expansion (chi-sq + Laguerre)
|   |-- gram_charlier.py          #   Gram-Charlier Type A (Hermite)
|   |-- pam.py                    #   PAG(d) polynomially adjusted gamma (arbitrary degree)
|   |-- exact.py                  #   Exact distribution (recursive enumeration)
|   |-- simulation.py             #   Monte Carlo simulation
|   +-- approximator.py           #   Unified interface (KWApproximator)
|
|-- examples/
|   |-- reproduce_paper_tables.py       # Reproduce paper Tables 4.1-4.7 + extensions
|   |-- tables_to_latex.py              # Raw text -> LaTeX (sorted by n1,n2,n3,n4)
|   |-- verify_cumulants_closed_form.py # Closed-form k3/k4 <-> Table 2 <-> enumeration + figure
|   |-- verify_cumulants_exact_identity.py # Rational enumeration proves the closed form is an identity
|   |-- plot_k3_k4_axes.py              # k3(x) vs k4(y) scatter
|   +-- plot_gamma1_gamma2_axes.py      # gamma1(x) vs gamma2(y) scatter
|
|-- tests/
|   +-- test_approximations.py    #   37 test cases
|
|
|-- README.md                     # Korean documentation
|-- README_EN.md                  # English documentation
|-- CLAUDE.md
|-- pyproject.toml
+-- Kruskal_Wallis_Test.pdf       # Paper PDF
```

## Core Pipeline

```
                         +------------------+
                         |   sample_sizes   |
                         | (n1, n2, ..., nk)|
                         +--------+---------+
                                  |
                                  v
                     +------------+-------------+
                     |        KWMoments          |
                     |  (exact enum / MC sim)    |
                     |                           |
                     |  k<=3, N<=15: exact       |
                     |  k>=4, N<=13: exact       |
                     |  otherwise:   simulation  |
                     +------------+-------------+
                                  |
                                  v
                       +--------------------+
                       | Cumulants k1...k4  |
                       | (exact finite-     |
                       |  sample, NOT       |
                       |  asymptotic chi-sq)|
                       +----+----+----+----+
                            |    |    |    |
            +---------------+    |    |    +----------------+
            |           +--------+    +---------+           |
            v           v                       v           v
     +-----------+ +-----------+         +-----------+ +-----------+
     |Saddlepoint| |Edgeworth  |         |Gram-      | |   PAG(d)  |
     |  ER1/ER2  | |(chi-sq +  |         |Charlier   | | (gamma x  |
     |           | | Laguerre) |         |Type A     | | polynomial|
     +-----------+ +-----------+         |(Hermite)  | |  any d)   |
     |  +- CC    |                       +-----------+ +-----------+
     |  variants |
     +-----+-----+
           |
           v
    +-------------+
    |Lugannani-   |
    |Rice tail    |---------->  P(H >= v)
    |probability  |
    +-------------+
```

## Installation

```bash
# uv (recommended) — after cloning, install dependencies + the package (editable)
uv sync
uv sync --extra dev     # also include dev tools (pytest, etc.)

# Using pip instead (numpy/scipy resolved from pyproject dependencies)
pip install -e ".[dev]"
```

Run scripts with `uv run python <script>`.

## Quick Start

```python
from kw_approx import KWApproximator

# Three groups, 3 observations each
approx = KWApproximator([3, 3, 3])

# Tail probability P(H >= 4.62)
p_value = approx.tail_probability(4.62, method='ER1')
print(f"ER1 P-value: {p_value:.6f}")

# Compare all methods
results = approx.compare_methods(4.62)
for method, p in results.items():
    print(f"{method}: {p:.6f}")
```

## Approximation Methods

### Method Naming (Paper <-> Code)

```
+----------+--------------------+----------------------+-----------------------+
| Paper    | Code method        | Base / CGF           | Note                  |
+----------+--------------------+----------------------+-----------------------+
| SD1      | 'ER1' / 'SD1'      | K_ER1 (4-term poly)  | Saddlepoint + L-R     |
| SD2      | 'gamma' / 'SD2'    | gamma base (WBB '93) | Non-normal base L-R   |
| SDC1     | 'ER1_cc' / 'SDC1'  | ER1 + CC             | Continuity corrected  |
| SDC2     | 'gamma_cc' / 'SDC2'| gamma base + CC      | WBB + cont. correction|
| CHI      | 'chi_square'       | -                    | Baseline (chi-sq)     |
| ED       | 'edgeworth'        | -                    | Laguerre-based        |
| GC-A     | 'gram_charlier'    | -                    | Hermite-based         |
| PAG(4)   | 'pam'              | -                    | Gamma x poly (d=4)    |
| PAG(6)   | 'pam6'             | -                    | Gamma x poly (d=6)    |
| PAG(d)   | 'pam<d>'           | -                    | Arbitrary d (e.g 'pam8')|
| -        | 'exact'            | -                    | Small N enumeration   |
| -        | 'simulation'       | -                    | Monte Carlo           |
+----------+--------------------+----------------------+-----------------------+
```

> **Arbitrary PAG degree:** besides `'pam'`=PAG(4) and `'pam6'`=PAG(6), any degree is callable as
> `'pam<d>'` (e.g. `'pam8'`, `'pam10'`). Very high degrees make the moment matrix ill-conditioned
> (a warning is emitted).

> **SD2 / disabled CGF notes:**
>
> - **SD2 / SDC2 = Wood-Booth-Butler (1993) gamma-based saddlepoint.** The non-normal-base Lugannani--Rice approximation already described and cited in paper §4.2. Implemented in `saddlepoint.py` as `tail_probability_gamma_based(v, continuity_correction=False)`. Table columns are `SD2(WBB)` / `SDC2(WBB)`.
> - **Wang** damped CGF (former SD2): $K''(\hat t)$ collapses toward 0 deep in the tail near $N \approx 15$, blowing up Lugannani--Rice (e.g. (5,5,5) at $\alpha = 0.05$ returns 0.504 vs. reference ~0.05). **Removed from the code** (kept only as a historical note).
> - **KT** $(1+\kappa_2)$ polynomial CGF: not present in Kakizawa-Taniguchi (1994) — that paper studies higher-order Edgeworth↔saddlepoint relations, not CGFs. The $(1+\kappa_2)$ shift is also asymptotically inconsistent (inflates variance by ~30% even as $N \to \infty$), so it was **removed from the code**.
> - Supported CGF names are only `'ER1'`, `'ER2'`, `'exact'`; passing anything else (e.g. `'Wang'`) to `SaddlepointApproximation(cgf_method=...)` raises `ValueError`.

### Cumulants -- Exact Finite-Sample

Following Section 4 of the paper, all approximation methods use **exact finite-sample cumulants**, not asymptotic chi-square cumulants:

- Small N (k<=3, N<=15 or k>=4, N<=13): computed from exact distribution
- Large N: Monte Carlo simulation (50,000 iterations, deterministic seed)

> **Note:** Asymptotic chi-sq cumulants show large errors for small samples.
> E.g., for (3,3,3): kappa_2: exact 2.72 vs asymptotic 4.0

### Saddlepoint Approximation -- Lugannani-Rice

Daniels (1954) saddlepoint density approximation:

$$f_{SP}(x) = \left(2\pi K_H^{(2)}(\hat{t})\right)^{-1/2} \exp(K_H(\hat{t}) - x\hat{t})$$

**Lugannani-Rice tail probability:**

$$\Pr(H \geq v) \approx 1 - \Phi(\hat{w}) + \phi(\hat{w})\left(\frac{1}{\hat{u}} - \frac{1}{\hat{w}}\right)$$

where:
- $\hat{w} = \text{sgn}(\hat{t})\sqrt{2(\hat{t}v - K_H(\hat{t}))}$
- $\hat{u} = \hat{t}\sqrt{K_H^{(2)}(\hat{t})}$

### CGF Approximation Methods

**ER1 (Easton-Ronchetti 1st):**
$$K_H(t) \approx \sum_{i=1}^{4} \frac{\kappa_i t^i}{i!}$$

**ER2 (Easton-Ronchetti 2nd):**
$$K_H(t) \approx \kappa_1 t + \frac{\kappa_2}{2}t^2 + \log\left(1 + \frac{\kappa_3}{6}t^3 + \frac{3\kappa_4}{72}t^4 + \frac{\kappa_3^2}{72}t^6\right)$$

> **Note (removed):** the Wang damped CGF once used for SD2,
> $K_H(t)\approx\kappa_1 t+\tfrac{\kappa_2}{2}t^2+(\tfrac{\kappa_3}{6}t^3+\tfrac{\kappa_4}{24}t^4)\,\eta_p(t)$
> with $\eta_p(t)=\exp(-\kappa_2 p^2 t^2/2)$, was removed from the code (blows up near $N\approx15$). SD2 is now the WBB gamma-base below.

**SD2 / SDC2 — gamma-based saddlepoint (Wood, Booth & Butler 1993):**

Unlike SD1, which substitutes a polynomial CGF into the *normal*-base Lugannani--Rice formula, SD2 changes the **base distribution itself** from normal to gamma (matched to the first two moments of H). The H-scale saddlepoint $\hat t$ is mapped onto the gamma reference family by CGF-exponent matching:

$$K_G(t_\xi)\, -\, t_\xi\, \xi\ =\ K_H(\hat t)\, -\, \hat t\, v.$$

With a curvature-ratio correction $u_{\hat\xi} = \hat t\,\sqrt{K_H^{(2)}(\hat t)\,/\,K_G^{(2)}(t_{\hat\xi})}$,

$$\Pr(H \geq v)\ \approx\ 1 - G(\hat\xi)\ +\ g(\hat\xi)\left(\tfrac{1}{u_{\hat\xi}} - \tfrac{1}{t_{\hat\xi}}\right).$$

Because H has no closed-form CGF, $K_H$ in the formula is the ER1 polynomial (an unavoidable structural limitation). Nonetheless the gamma base aligns with the $\chi^2$-like target shape and avoids Wang-type blow-up.

```python
from kw_approx import KWApproximator

approx = KWApproximator([5, 5, 5])
p_sd2  = approx.tail_probability(4.5, 'SD2')    # gamma-WBB
p_sdc2 = approx.tail_probability(4.5, 'SDC2')   # + continuity correction
```

**Disabled KT note:**
An earlier attempt used $K_{\rm KT}(t)=\kappa_1 t+(1+\kappa_2)t^2/2+\kappa_3 t^3/6+\kappa_4 t^4/24$ as SD2, but this formula does not appear in Kakizawa--Taniguchi (1994) — that paper is a theoretical study of the Edgeworth↔saddlepoint relation, not a CGF proposal. The $(1+\kappa_2)$ term also makes the approximation asymptotically inconsistent (variance error ~30% at the $\chi^2$ limit), so it was removed.

**Accuracy note (for reference):** SD1 and SD2(WBB) are comparable (mean absolute error ~0.008 over 196 rows), with **PAG(4) the most accurate overall** (MAE ~0.004). PAG fits moments directly to a gamma×polynomial density without going through CGF truncation; saddlepoint methods inherit a structural ceiling because $K_H$ must be approximated by a polynomial in the absence of a closed-form KW CGF.

### Edgeworth Expansion

Chi-square baseline with generalized Laguerre polynomial corrections for skewness and kurtosis:

$$F_{ED}(x) = G_{\nu}(x) - g_{\nu}(x)\left[\frac{\gamma_1}{6}L_3^{(\nu/2-1)}(x/2) + \frac{\gamma_2}{24}L_4^{(\nu/2-1)}(x/2) + \frac{\gamma_1^2}{72}L_6^{(\nu/2-1)}(x/2)\right]$$

### Gram-Charlier Type A

Normal baseline with Hermite polynomial corrections ($H_3, H_4, H_6$).

### PAG(d) -- Polynomially Adjusted Gamma

Gamma density multiplied by a degree-$d$ polynomial:

$$f_{PAG}(x; d) = \psi(x) \sum_{i=0}^{d} \xi_i x^i$$

Coefficients $\xi_0, \ldots, \xi_d$ are determined by matching the first $d+1$ moments via moment matrix inversion. **Any degree** is available (`'pam<d>'` or `PolynomialAdjustedGamma(sample_sizes, degree=d)`); the paper uses $d=4$ (`'pam'`) and $d=6$ (`'pam6'`).

```python
from kw_approx import KWApproximator
approx = KWApproximator([5, 5, 5])
for m in ['pam', 'pam6', 'pam8', 'pam10']:      # PAG(4), (6), (8), (10)
    print(m, approx.tail_probability(8.0, m))
```

## Moments and Cumulants

Under $H_0$:

- **Mean**: $E(H) = k - 1$ (exact, independent of $n_i$)
- **Variance** (Wallace 1959 / paper Thm 3.9 exact closed form):

$$\text{Var}(H) = 2(k-1) - \frac{2A_W}{5\,N(N+1)} - \frac{6}{5}\sum_{i=1}^{k}\frac{1}{n_i},\qquad A_W = 3k(k-2) + N(2k^2 - 6k + 1).$$

**Cumulants** (exact finite-sample):
- $\kappa_1 = E(H) = k - 1$
- $\kappa_2 = \text{Var}(H)$
- $\kappa_3 = \mu_3 - 3\mu_2\mu_1 + 2\mu_1^3$
- $\kappa_4 = \mu_4 - 4\mu_3\mu_1 - 3\mu_2^2 + 12\mu_2\mu_1^2 - 6\mu_1^4$

The default pipeline (`KWMoments`) obtains $\kappa_1,\dots,\kappa_4$ from exact enumeration (small samples) or simulated finite-sample moments (large samples).

### Closed-form combinatorial cumulants (Theorem 3.12)

Per paper Theorem 3.12, $\kappa_3(H)=c^3\kappa_3(Q)$ and $\kappa_4(H)=c^4\kappa_4(Q)$ ($c=12/[N(N+1)]$, $Q=\sum_i U_i^2/n_i$) are computed **without enumeration** from the even joint moments of the centred rank sums (multivariate hypergeometric master formula). `cumulants_closed_form.py` implements this in exact rationals (`fractions.Fraction`) and **reproduces the exact Table 2 values, not an approximation** ($\Delta=0$ against rational enumeration).

```python
from kw_approx.cumulants_closed_form import cumulants_closed_form, kappa_H

c = cumulants_closed_form([5, 5, 5])
print(c["k3"], c["k4"], c["gamma1"], c["gamma2"])
#   8.1469  21.0201  1.3969  2.0024   <- equals paper Table 2 for (5,5,5)
print(kappa_H([5, 5, 5], 3))   # exact Fraction(14257, 1750)
```

## Detailed Usage

### Individual Approximation Classes

```python
from kw_approx import SaddlepointApproximation, ExactDistribution, MonteCarloSimulation

sample_sizes = [3, 3, 3]

# Saddlepoint with ER1 CGF
sp_er1 = SaddlepointApproximation(sample_sizes, cgf_method='ER1')

print(f"ER1: {sp_er1.tail_probability_lr(4.62):.6f}")

# Exact (small samples only)
exact = ExactDistribution(sample_sizes)
print(f"Exact: {exact.tail_probability(4.62):.6f}")

# Monte Carlo (large samples)
sim = MonteCarloSimulation(sample_sizes, n_simulations=10000, seed=42)
print(f"Simulation: {sim.tail_probability(4.62):.6f}")
```

### Critical Value Computation

```python
from kw_approx import KWApproximator

approx = KWApproximator([5, 5, 5])

for alpha in [0.10, 0.05, 0.01]:
    cv_exact = approx.critical_value(alpha, method='exact')
    cv_chi2 = approx.critical_value(alpha, method='chi_square')
    cv_er1 = approx.critical_value(alpha, method='ER1')
    print(f"alpha={alpha}: Exact={cv_exact:.4f}, Chi2={cv_chi2:.4f}, ER1={cv_er1:.4f}")
```

## Recommended Methods by Sample Size

```
+---------------+----------+---------------+---------------------------+
| Sample size N | Groups k | Recommended   | Note                      |
+---------------+----------+---------------+---------------------------+
| N <= 15       | k <= 3   | exact         | Exact distribution        |
| N <= 13       | k = 4    | exact         | Exact distribution        |
| 15 < N < 100  | any      | ER1           | Saddlepoint + L-R         |
| N >= 100      | any      | chi_square    | Asymptotic sufficient     |
| any           | any      | simulation    | MC reference (slow)       |
+---------------+----------+---------------+---------------------------+
```

## Testing

```bash
# Run all tests (37 cases)
python -m pytest tests/test_approximations.py -v

# Run a single test class
python -m pytest tests/test_approximations.py::TestKWMoments -v

# Reproduce paper tables (save as raw text)
uv run python examples/reproduce_paper_tables.py > result/paper_tables_raw.txt

# Convert to LaTeX (sorted by n1, n2, n3, n4; SD2/SDC2(WBB) columns)
uv run python examples/tables_to_latex.py
# -> writes result/paper_tables.tex
```

## Dependencies

- Python >= 3.13
- NumPy >= 2.4.1
- SciPy >= 1.17.0

## References

1. Kruskal, W. H. & Wallis, A. (1952). Use of ranks in one-criterion variance analysis. *JASA*, 47, 583-621.
2. Wallace, D. L. (1959). Simplified beta-approximations to the Kruskal-Wallis H test. *JASA*, 54, 225-230.
3. Daniels, H. E. (1954). Saddlepoint approximations in statistics. *Annals of Mathematical Statistics*, 25, 631-650.
4. Iman, R. L., Quade, D., & Alexander, D. A. (1975). Exact probability levels for the Kruskal-Wallis test.
5. Lugannani, R. & Rice, S. O. (1980). Saddlepoint approximation for the distribution of the sum of independent random variables. *Advances in Applied Probability*, 12, 475-490.
6. Easton, G. S. & Ronchetti, E. (1986). General saddlepoint approximations with applications to L statistics. *JASA*, 81, 420-430.
7. Wang, S. (1992). General saddlepoint approximations in the bootstrap. *Statistics & Probability Letters*, 13, 61-66.
8. Wood, A. T. A., Booth, J. G., & Butler, R. W. (1993). Saddlepoint approximations to the CDF of some statistics with nonnormal limit distributions. *JASA*, 88, 680-686. *(basis for SD2 / SDC2)*
9. Ha, H.-T. & Provost, S. B. (2007). A viable alternative to resorting to statistical tables. *Communications in Statistics*, 36, 1135-1151.
10. Hall, P. (1992). *The Bootstrap and Edgeworth Expansion*. Springer.

## License

MIT License
