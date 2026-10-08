[**English**](README_EN.md) | **Korean**

# kw-approx: Higher Order Asymptotic Approximations for Kruskal-Wallis Statistics

Kruskal-Wallis 검정 통계량의 고차 점근 근사를 구현한 Python 패키지입니다.

## 논문 정보

> **Lee, J.-S., Murakami, H., & Ha, H.-T. (2026).** *Higher Order Asymptotic Approximations of Kruskal-Wallis Statistics Based on Skewness and Kurtosis.* Preprint submitted to Journals.

## 배경

Kruskal-Wallis 검정은 k개 집단의 위치 모수 동일성을 검정하는 비모수적 방법으로, 일원분산분석(One-way ANOVA)의 순위 기반 대안입니다.

### Kruskal-Wallis H 통계량

$$H = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1)$$

- $N = \sum_{i=1}^{k} n_i$: 전체 표본 크기
- $n_i$: $i$번째 집단의 표본 크기
- $R_i$: $i$번째 집단의 순위합

전통적으로 $H$의 귀무분포는 자유도 $k-1$인 카이제곱 분포로 근사하지만, 소표본에서는 정확도가 떨어집니다. 이 패키지는 새들포인트, Edgeworth, Gram-Charlier, PAG 근사를 통해 더 정확한 근사를 제공합니다.

## 프로젝트 구조

```
KS-test/
|
|-- kw_approx/                    # 핵심 패키지
|   |-- __init__.py               #   패키지 초기화 (v0.3.0)
|   |-- kruskal_wallis.py         #   H 통계량 계산
|   |-- moments.py                #   모멘트/큐뮬런트 (exact/simulation)
|   |-- cumulants_closed_form.py  #   k3/k4 닫힌형 조합식 (Thm 3.12, 열거 불필요)
|   |-- saddlepoint.py            #   새들포인트 근사 (ER1/ER2 + L-R, WBB gamma-base = SD2)
|   |-- edgeworth.py              #   Edgeworth 전개 (chi-sq 기반 Laguerre)
|   |-- gram_charlier.py          #   Gram-Charlier Type A (Hermite)
|   |-- pam.py                    #   PAG(d) 다항식 조정 감마 근사 (임의 차수)
|   |-- exact.py                  #   정확 분포 (재귀 열거, 소표본)
|   |-- simulation.py             #   Monte Carlo 시뮬레이션
|   +-- approximator.py           #   통합 인터페이스 (KWApproximator)
|
|-- examples/
|   |-- reproduce_paper_tables.py       # 논문 Table 4.1~4.7 재현 + 확장
|   |-- tables_to_latex.py              # raw 출력 -> LaTeX (n1,n2,n3,n4 오름차순)
|   |-- verify_cumulants_closed_form.py # 닫힌형 k3/k4 <-> Table 2 <-> 열거 3중 대조 + 그림
|   |-- verify_cumulants_exact_identity.py # 유리수 열거로 닫힌형이 '항등'임을 증명 (Δ=0)
|   |-- plot_k3_k4_axes.py              # k3(x) vs k4(y) 산점도
|   +-- plot_gamma1_gamma2_axes.py      # gamma1(x) vs gamma2(y) 산점도
|
|-- tests/
|   +-- test_approximations.py    #   37개 테스트 케이스
|
|
|-- README.md
|-- README_EN.md
|-- CLAUDE.md
|-- pyproject.toml
+-- Kruskal_Wallis_Test.pdf       # 논문 원본
```

## 핵심 파이프라인

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

## 설치

```bash
# uv (권장) — 저장소 클론 후, 의존성 + 패키지를 editable 모드로 설치
uv sync
uv sync --extra dev     # pytest 등 개발 도구까지 포함

# pip 를 쓰는 경우 (numpy/scipy 는 pyproject 의존성으로 자동 설치)
pip install -e ".[dev]"
```

설치 후 스크립트는 `uv run python <script>` 로 실행합니다.

## 빠른 시작

```python
from kw_approx import KWApproximator

# 3개 집단, 각 3명씩
approx = KWApproximator([3, 3, 3])

# H = 4.62에서 꼬리확률 P(H >= 4.62)
p_value = approx.tail_probability(4.62, method='ER1')
print(f"ER1 P-value: {p_value:.6f}")

# 여러 방법 비교
results = approx.compare_methods(4.62)
for method, p in results.items():
    print(f"{method}: {p:.6f}")
```

## 구현된 근사 방법

### 방법 일람 (논문 <-> 코드)

```
+----------+--------------------+----------------------+-----------------------+
| 논문     | 코드 method        | 기저 / CGF           | 비고                  |
+----------+--------------------+----------------------+-----------------------+
| SD1      | 'ER1' / 'SD1'      | K_ER1 (4항 다항식)   | Saddlepoint + L-R     |
| SD2      | 'gamma' / 'SD2'    | gamma base (WBB '93) | 비정규 base L-R       |
| SDC1     | 'ER1_cc' / 'SDC1'  | ER1 + CC             | 연속성 보정           |
| SDC2     | 'gamma_cc' / 'SDC2'| gamma base + CC      | WBB + 연속성 보정     |
| CHI      | 'chi_square'       | -                    | Baseline (chi-sq)     |
| ED       | 'edgeworth'        | -                    | Laguerre 기반         |
| GC-A     | 'gram_charlier'    | -                    | Hermite 기반          |
| PAG(4)   | 'pam'              | -                    | 감마 x 다항식 (d=4)   |
| PAG(6)   | 'pam6'             | -                    | 감마 x 다항식 (d=6)   |
| PAG(d)   | 'pam<d>'           | -                    | 임의 차수 (예 'pam8') |
| -        | 'exact'            | -                    | 소표본 전용 (열거)    |
| -        | 'simulation'       | -                    | MC 시뮬레이션         |
+----------+--------------------+----------------------+-----------------------+
```

> **PAG 임의 차수:** `'pam'`=PAG(4), `'pam6'`=PAG(6) 외에도 `'pam<d>'`(예 `'pam8'`, `'pam10'`)로
> 임의 차수를 바로 호출할 수 있습니다. 차수가 높아지면 moment matrix가 병증(ill-conditioned)이라
> 경고를 냅니다.

> **SD2 / 폐기 CGF 메모:**
>
> - **SD2 / SDC2 = Wood-Booth-Butler (1993) gamma-based saddlepoint.** 논문 §4.2가 이미 인용·기술하는 비정규 base Lugannani--Rice. 구현은 `saddlepoint.py`의 `tail_probability_gamma_based(v, continuity_correction=False)`. 표 컬럼은 `SD2(WBB)` / `SDC2(WBB)`.
> - **Wang** damped CGF (과거 SD2)는 N≈15 부근에서 $K''(\hat t)$ 가 0 근방으로 붕괴해 Lugannani--Rice가 폭주합니다 (예: (5,5,5) α=0.05 → 0.504, 참값 ~0.05). **코드에서 제거**했습니다(과거 사유 기록용 메모만 유지).
> - **KT** $(1+\kappa_2)$ 다항식 CGF는 Kakizawa-Taniguchi (1994)에 그 식이 없습니다 (그 논문은 Edgeworth↔saddlepoint 차수 관계 이론). 점근적으로도 비일관(분산을 ~30% 고정 부풀림)이라 **코드에서 제거**.
> - 지원 CGF 이름은 `'ER1'`, `'ER2'`, `'exact'` 뿐이며, 그 외(`'Wang'` 등)를 `SaddlepointApproximation(cgf_method=...)`에 넘기면 `ValueError`가 발생합니다.

### 큐뮬런트 (Cumulants) -- Exact Finite-Sample

논문 Section 4에 따라, 모든 근사 방법은 **정확한 유한표본 큐뮬런트**를 사용합니다:

- 소표본 (k<=3, N<=15 또는 k>=4, N<=13): exact distribution으로 계산
- 대표본: Monte Carlo simulation (50,000 iterations, deterministic seed)

> **참고:** 점근적 chi-sq 큐뮬런트는 소표본에서 큰 오차를 보이므로 사용하지 않습니다.
> 예: (3,3,3)에서 kappa_2: exact 2.72 vs asymptotic 4.0

### Saddlepoint Approximation -- Lugannani-Rice

Daniels (1954)의 새들포인트 밀도 근사:

$$f_{SP}(x) = \left(2\pi K_H^{(2)}(\hat{t})\right)^{-1/2} \exp(K_H(\hat{t}) - x\hat{t})$$

**Lugannani-Rice 꼬리확률 근사:**

$$\Pr(H \geq v) \approx 1 - \Phi(\hat{w}) + \phi(\hat{w})\left(\frac{1}{\hat{u}} - \frac{1}{\hat{w}}\right)$$

여기서:
- $\hat{w} = \text{sgn}(\hat{t})\sqrt{2(\hat{t}v - K_H(\hat{t}))}$
- $\hat{u} = \hat{t}\sqrt{K_H^{(2)}(\hat{t})}$

### CGF 근사 방법

**ER1 (Easton-Ronchetti 1st):**
$$K_H(t) \approx \sum_{i=1}^{4} \frac{\kappa_i t^i}{i!}$$

**ER2 (Easton-Ronchetti 2nd):**
$$K_H(t) \approx \kappa_1 t + \frac{\kappa_2}{2}t^2 + \log\left(1 + \frac{\kappa_3}{6}t^3 + \frac{3\kappa_4}{72}t^4 + \frac{\kappa_3^2}{72}t^6\right)$$

> **참고 (제거됨):** 과거 SD2에 쓰였던 Wang damped CGF
> $K_H(t)\approx\kappa_1 t+\tfrac{\kappa_2}{2}t^2+(\tfrac{\kappa_3}{6}t^3+\tfrac{\kappa_4}{24}t^4)\,\eta_p(t)$,
> $\eta_p(t)=\exp(-\kappa_2 p^2 t^2/2)$ 는 N≈15 부근 폭주로 코드에서 제거했습니다. 이제 SD2는 아래 WBB gamma-base입니다.

**SD2 / SDC2 — gamma-based saddlepoint (Wood, Booth & Butler 1993):**

다항식 CGF를 정규 base Lugannani--Rice에 대입하는 SD1과 달리, SD2는 **base 분포를 정규에서 감마(첫 두 모멘트 매칭)로 바꾼** 비정규 base saddlepoint입니다. H-스케일 새들포인트 $\hat t$ 를 감마 기준족에 CGF 지수 매칭으로 사상:

$$K_G(t_\xi)\, -\, t_\xi\, \xi\ =\ K_H(\hat t)\, -\, \hat t\, v.$$

곡률 비를 반영한 $u_{\hat\xi} = \hat t\,\sqrt{K_H^{(2)}(\hat t)\,/\,K_G^{(2)}(t_{\hat\xi})}$ 로

$$\Pr(H \geq v)\ \approx\ 1 - G(\hat\xi)\ +\ g(\hat\xi)\left(\tfrac{1}{u_{\hat\xi}} - \tfrac{1}{t_{\hat\xi}}\right).$$

H의 정확 CGF는 닫힌형이 없어 $K_H$ 자리에 ER1 다항식을 씁니다(불가피한 본질적 한계). 그래도 base가 χ² 형태에 맞춰져 정규 base보다 부드럽고, Wang처럼 폭주하지도 않습니다.

```python
from kw_approx import KWApproximator

approx = KWApproximator([5, 5, 5])
p_sd2  = approx.tail_probability(4.5, 'SD2')    # gamma-WBB
p_sdc2 = approx.tail_probability(4.5, 'SDC2')   # + 연속성 보정
```

**폐기된 KT 메모:**
한때 $K_{\rm KT}(t)=\kappa_1 t+(1+\kappa_2)t^2/2+\kappa_3 t^3/6+\kappa_4 t^4/24$ 형태를 SD2로 시도했으나, Kakizawa--Taniguchi (1994)에 이 식이 없습니다(그 논문은 Edgeworth↔saddlepoint 차수 관계 이론 논문). $(1+\kappa_2)$ 항은 χ² 극한에서 분산을 약 30% 부풀리는 비일관 근사라 폐기했습니다.

**정확도 관찰 (참고):** SD1과 SD2(WBB)는 동등 수준이며 (196행 평균 절대오차 ~0.008), **PAG(4)가 가장 정확합니다** (MAE ~0.004). 이는 PAG가 CGF 절단 단계를 거치지 않고 모멘트를 직접 감마×다항식에 fitting하기 때문이며, KW에 닫힌형 CGF가 없는 한 saddlepoint 계열의 구조적 한계입니다.

### 연속성 보정 (Continuity Correction)

이산 분포의 특성을 보정하기 위해 $v$를 $v - 1/2$로 대체하여 CC 변형 계산.

### Edgeworth Expansion

카이제곱 분포를 기저로 사용하며, 일반화된 Laguerre 다항식을 통해 skewness/kurtosis 보정:

$$F_{ED}(x) = G_{\nu}(x) - g_{\nu}(x)\left[\frac{\gamma_1}{6}L_3^{(\nu/2-1)}(x/2) + \frac{\gamma_2}{24}L_4^{(\nu/2-1)}(x/2) + \frac{\gamma_1^2}{72}L_6^{(\nu/2-1)}(x/2)\right]$$

### Gram-Charlier Type A

정규분포를 기저로 사용하며, Hermite 다항식 $H_3, H_4, H_6$을 통해 보정.

### PAG(d) -- Polynomially Adjusted Gamma

감마 분포에 $d$차 다항식을 곱한 밀도 근사:

$$f_{PAG}(x; d) = \psi(x) \sum_{i=0}^{d} \xi_i x^i$$

처음 $d+1$개 모멘트를 일치시키는 계수 $\xi_0, \ldots, \xi_d$를 moment matrix 역행렬로 결정합니다. 차수는 **임의로 지정 가능**하며(`'pam<d>'` 또는 `PolynomialAdjustedGamma(sample_sizes, degree=d)`), 논문 표는 $d=4$(`'pam'`), $d=6$(`'pam6'`)를 사용합니다.

```python
from kw_approx import KWApproximator
approx = KWApproximator([5, 5, 5])
for m in ['pam', 'pam6', 'pam8', 'pam10']:      # PAG(4), (6), (8), (10)
    print(m, approx.tail_probability(8.0, m))
```

## 모멘트와 큐뮬런트

귀무가설 하에서 H 통계량의 기본 모멘트:

- **평균**: $E(H) = k - 1$ (정확, $n_i$ 무관)
- **분산** (Wallace 1959 / 논문 Thm 3.9 정확 닫힌형):

$$\text{Var}(H) = 2(k-1) - \frac{2A_W}{5\,N(N+1)} - \frac{6}{5}\sum_{i=1}^{k}\frac{1}{n_i},\qquad A_W = 3k(k-2) + N(2k^2 - 6k + 1).$$

**큐뮬런트** (exact finite-sample):
- $\kappa_1 = E(H) = k - 1$
- $\kappa_2 = \text{Var}(H)$
- $\kappa_3 = \mu_3 - 3\mu_2\mu_1 + 2\mu_1^3$
- $\kappa_4 = \mu_4 - 4\mu_3\mu_1 - 3\mu_2^2 + 12\mu_2\mu_1^2 - 6\mu_1^4$

기본 파이프라인(`KWMoments`)은 $\kappa_1,\dots,\kappa_4$ 를 소표본에서는 정확 분포 열거, 대표본에서는 시뮬레이션의 유한표본 모멘트에서 구합니다.

### 닫힌형 조합식 큐뮬런트 (Theorem 3.12)

논문 Theorem 3.12에 따라 $\kappa_3(H)=c^3\kappa_3(Q)$, $\kappa_4(H)=c^4\kappa_4(Q)$ ($c=12/[N(N+1)]$, $Q=\sum_i U_i^2/n_i$) 를 **열거 없이** 중심순위합의 짝수 결합모멘트(다변량 초기하 master 공식)로 직접 계산합니다. `cumulants_closed_form.py`에 유리수(`fractions.Fraction`)로 구현되어, Table 2의 exact 값을 **근사가 아니라 그대로 재현**합니다(유리수 열거와 $\Delta=0$).

```python
from kw_approx.cumulants_closed_form import cumulants_closed_form, kappa_H

c = cumulants_closed_form([5, 5, 5])
print(c["k3"], c["k4"], c["gamma1"], c["gamma2"])
#   8.1469  21.0201  1.3969  2.0024   <- 논문 Table 2 (5,5,5)와 동일
print(kappa_H([5, 5, 5], 3))   # 정확 유리수 Fraction(14257, 1750)
```

검증/그림 재생성:

```bash
uv run python examples/verify_cumulants_exact_identity.py   # 21개 설계 모두 Δ=0
uv run python examples/verify_cumulants_closed_form.py      # 3중 대조 + 수렴 그림
uv run python examples/plot_k3_k4_axes.py                   # k3 vs k4
uv run python examples/plot_gamma1_gamma2_axes.py           # gamma1 vs gamma2
```

## 상세 사용법

### 개별 근사 클래스 사용

```python
from kw_approx import SaddlepointApproximation, ExactDistribution, MonteCarloSimulation

sample_sizes = [3, 3, 3]

# Saddlepoint with ER1 CGF
sp_er1 = SaddlepointApproximation(sample_sizes, cgf_method='ER1')

print(f"ER1: {sp_er1.tail_probability_lr(4.62):.6f}")

# Exact (소표본)
exact = ExactDistribution(sample_sizes)
print(f"Exact: {exact.tail_probability(4.62):.6f}")

# Monte Carlo (대표본)
sim = MonteCarloSimulation(sample_sizes, n_simulations=10000, seed=42)
print(f"Simulation: {sim.tail_probability(4.62):.6f}")
```

### 임계값 계산

```python
from kw_approx import KWApproximator

approx = KWApproximator([5, 5, 5])

for alpha in [0.10, 0.05, 0.01]:
    cv_exact = approx.critical_value(alpha, method='exact')
    cv_chi2 = approx.critical_value(alpha, method='chi_square')
    cv_er1 = approx.critical_value(alpha, method='ER1')
    print(f"alpha={alpha}: Exact={cv_exact:.4f}, Chi2={cv_chi2:.4f}, ER1={cv_er1:.4f}")
```

### 실제 데이터에 적용

```python
import numpy as np
from scipy import stats
from kw_approx import KWApproximator

np.random.seed(42)
group1 = np.random.normal(10, 2, 5)
group2 = np.random.normal(12, 2, 5)
group3 = np.random.normal(11, 2, 5)

result = stats.kruskal(group1, group2, group3)
H = result.statistic

approx = KWApproximator([5, 5, 5])
print(f"H statistic: {H:.4f}")
print(f"SciPy p-value: {result.pvalue:.4f}")

for method in ['exact', 'chi_square', 'ER1', 'edgeworth', 'gram_charlier', 'pam']:
    p = approx.tail_probability(H, method)
    print(f"{method}: {p:.4f}")
```

## 방법별 권장 사용 상황

```
+---------------+----------+---------------+---------------------------+
| 표본 크기 (N) | 그룹 (k) | 권장 방법     | 비고                      |
+---------------+----------+---------------+---------------------------+
| N <= 15       | k <= 3   | exact         | 정확 분포 계산 가능       |
| N <= 13       | k = 4    | exact         | 정확 분포 계산 가능       |
| 15 < N < 100  | any      | ER1           | 새들포인트 + L-R          |
| N >= 100      | any      | chi_square    | 점근 근사 충분            |
| any           | any      | simulation    | MC 참조값 (느림)          |
+---------------+----------+---------------+---------------------------+
```

## 테스트

```bash
# 전체 테스트 실행 (37개)
python -m pytest tests/test_approximations.py -v

# 단일 테스트 클래스
python -m pytest tests/test_approximations.py::TestKWMoments -v

# 논문 테이블 재현 (raw 텍스트로 저장)
uv run python examples/reproduce_paper_tables.py > result/paper_tables_raw.txt

# LaTeX로 변환 (n1, n2, n3, n4 오름차순 정렬, SD2/SDC2(WBB) 컬럼)
uv run python examples/tables_to_latex.py
# -> result/paper_tables.tex 생성
```

## 의존성

- Python >= 3.13
- NumPy >= 2.4.1
- SciPy >= 1.17.0

## 참고문헌

1. Kruskal, W. H. & Wallis, A. (1952). Use of ranks in one-criterion variance analysis. *JASA*, 47, 583-621.
2. Wallace, D. L. (1959). Simplified beta-approximations to the Kruskal-Wallis H test. *JASA*, 54, 225-230.
3. Daniels, H. E. (1954). Saddlepoint approximations in statistics. *Annals of Mathematical Statistics*, 25, 631-650.
4. Iman, R. L., Quade, D., & Alexander, D. A. (1975). Exact probability levels for the Kruskal-Wallis test.
5. Lugannani, R. & Rice, S. O. (1980). Saddlepoint approximation for the distribution of the sum of independent random variables. *Advances in Applied Probability*, 12, 475-490.
6. Easton, G. S. & Ronchetti, E. (1986). General saddlepoint approximations with applications to L statistics. *JASA*, 81, 420-430.
7. Wang, S. (1992). General saddlepoint approximations in the bootstrap. *Statistics & Probability Letters*, 13, 61-66.
8. Wood, A. T. A., Booth, J. G., & Butler, R. W. (1993). Saddlepoint approximations to the CDF of some statistics with nonnormal limit distributions. *JASA*, 88, 680-686. *(SD2 / SDC2의 근거)*
9. Ha, H.-T. & Provost, S. B. (2007). A viable alternative to resorting to statistical tables. *Communications in Statistics*, 36, 1135-1151.
10. Hall, P. (1992). *The Bootstrap and Edgeworth Expansion*. Springer.

## 라이선스

MIT License
