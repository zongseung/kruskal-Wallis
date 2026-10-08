"""
Closed-form combinatorial cumulants of the Kruskal-Wallis statistic H.

This module implements Theorem 3.12 of Murakami, Lee & Ha (2026) and the
accompanying "Explicit derivation of the third and fourth cumulants" note:
the third and fourth cumulants of H are obtained *without enumeration* from
the exact even joint moments of the centred rank sums, via the combinatorial
multivariate hypergeometric ("master joint-moment") formula.

Pipeline (all exact, using rationals):

    H = c * Q,           c = 12 / [N(N+1)],     Q = sum_i U_i^2 / n_i
    kappa_m(H) = c^m * kappa_m(Q)                              (Lemma: affine reduction)
    kappa_r(Q) = sum_{i_1..i_r} (1/ n_{i_1}..n_{i_r}) kappa(U_{i_1}^2, .., U_{i_r}^2)
                                                              (multilinear expansion)
    kappa(U^2, ..) : moment-cumulant inversion over set partitions
    E[ prod_i U_i^{a_i} ] : master formula (eq:master + eq:distinct)

Every quantity is computed as an exact ``fractions.Fraction`` and only cast to
float at the boundary, so the output reproduces the enumerated cumulants of
Table 2 to full machine precision (they are the same number, computed two
different ways).

References
----------
- Murakami, Lee & Ha (2026), Theorem 3.12 (third and fourth cumulants).
- "Explicit derivation of the third and fourth cumulants of H" (companion note):
  Lemmas affine / multilinear / inversion / master joint-moment formula.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from math import factorial
from typing import Dict, List, Sequence, Tuple

__all__ = [
    "power_sum",
    "joint_moment",
    "joint_cumulant_of_squares",
    "kappa_Q",
    "kappa_H",
    "cumulants_closed_form",
]


# --------------------------------------------------------------------------- #
# Set-partition utilities
# --------------------------------------------------------------------------- #
def _set_partitions(elements: Sequence) -> List[List[list]]:
    """Enumerate all set partitions of ``elements`` as lists of blocks."""
    elements = list(elements)
    if not elements:
        return [[]]
    first, rest = elements[0], elements[1:]
    partitions = []
    for smaller in _set_partitions(rest):
        # insert `first` into each existing block ...
        for i in range(len(smaller)):
            partitions.append(smaller[:i] + [[first] + smaller[i]] + smaller[i + 1:])
        # ... or as its own new block
        partitions.append([[first]] + smaller)
    return partitions


def _falling_factorial(x: int, t: int) -> int:
    """x^{(t)} = x (x-1) ... (x-t+1)."""
    result = 1
    for j in range(t):
        result *= (x - j)
    return result


# --------------------------------------------------------------------------- #
# Population power sums  P_m = sum_{r=1}^N (r - (N+1)/2)^m
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def power_sum(N: int, m: int) -> Fraction:
    """Exact centred-rank power sum ``P_m = sum_r (r - (N+1)/2)**m``.

    Computed directly (not via the closed polynomials) so it is correct for
    every ``m``; the paper's P2, P4, P6, P8 formulas are a special case.
    """
    mu = Fraction(N + 1, 2)
    total = Fraction(0)
    for r in range(1, N + 1):
        total += (Fraction(r) - mu) ** m
    return total


# --------------------------------------------------------------------------- #
# Distinct-index power-sum reduction  (eq:distinct)
#   sum_{distinct r_1..r_m} prod_l y_{r_l}^{b_l}
#     = sum_{rho} prod_{C in rho} (-1)^{|C|-1} (|C|-1)! P_{sum_{l in C} b_l}
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _distinct_sum(N: int, block_exps: Tuple[int, ...]) -> Fraction:
    m = len(block_exps)
    total = Fraction(0)
    for rho in _set_partitions(range(m)):
        term = Fraction(1)
        for C in rho:
            b = sum(block_exps[l] for l in C)
            sign = (-1) ** (len(C) - 1)
            term *= sign * factorial(len(C) - 1) * power_sum(N, b)
        total += term
    return total


# --------------------------------------------------------------------------- #
# Master joint-moment formula  (eq:master)
#   E[ prod_i U_i^{a_i} ] = sum_{pi monochromatic}
#         ( distinct-sum over blocks ) * prod_i n_i^{(t_i)} / N^{(|pi|)}
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _joint_moment_cached(N: int, n: Tuple[int, ...], exps: Tuple[int, ...]) -> Fraction:
    # Slots, each tagged by its group colour.
    slots: List[int] = []
    for g, a in enumerate(exps):
        slots.extend([g] * a)

    D = len(slots)
    if D == 0:
        return Fraction(1)
    if D % 2 == 1:
        # Odd total degree -> vanishes by reflection symmetry r -> N+1-r.
        return Fraction(0)

    total = Fraction(0)
    for pi in _set_partitions(range(D)):
        # A block must be monochromatic (xi_{i,r} xi_{j,r} = 0 for i != j).
        colours = []
        block_sizes = []
        ok = True
        for block in pi:
            cols = {slots[s] for s in block}
            if len(cols) != 1:
                ok = False
                break
            colours.append(next(iter(cols)))
            block_sizes.append(len(block))
        if not ok:
            continue

        # Probability factor prod_i n_i^{(t_i)} / N^{(|pi|)}.
        prob_num = 1
        for g in range(len(n)):
            t_g = colours.count(g)
            if t_g:
                prob_num *= _falling_factorial(n[g], t_g)
        prob_den = _falling_factorial(N, len(pi))
        if prob_den == 0:
            continue

        inner = _distinct_sum(N, tuple(sorted(block_sizes)))
        total += inner * Fraction(prob_num, prob_den)

    return total


def joint_moment(N: int, n: Sequence[int], exps: Sequence[int]) -> Fraction:
    """E[ prod_i U_i^{exps[i]} ] under H0 (exact, rational)."""
    return _joint_moment_cached(N, tuple(int(x) for x in n), tuple(int(x) for x in exps))


# --------------------------------------------------------------------------- #
# Joint cumulant of squared rank sums  kappa(U_{g1}^2, ..., U_{gr}^2)
#   standard moment->cumulant set-partition inversion, each factor an even
#   joint moment of the U_i.
# --------------------------------------------------------------------------- #
def _moment_of_squares(N: int, n: Sequence[int], groups: Sequence[int]) -> Fraction:
    """E[ prod_{g in groups} U_g^2 ]  (each listed group contributes exponent 2)."""
    exps = [0] * len(n)
    for g in groups:
        exps[g] += 2
    return joint_moment(N, n, exps)


def joint_cumulant_of_squares(N: int, n: Sequence[int], groups: Sequence[int]) -> Fraction:
    """kappa(U_{groups[0]}^2, ..., U_{groups[r-1]}^2) via partition inversion."""
    r = len(groups)
    total = Fraction(0)
    for pi in _set_partitions(range(r)):
        sign = (-1) ** (len(pi) - 1)
        coeff = sign * factorial(len(pi) - 1)
        term = Fraction(coeff)
        for block in pi:
            term *= _moment_of_squares(N, n, [groups[i] for i in block])
        total += term
    return total


# --------------------------------------------------------------------------- #
# Cumulants of Q and H
# --------------------------------------------------------------------------- #
def kappa_Q(n: Sequence[int], order: int) -> Fraction:
    """kappa_order(Q) with Q = sum_i U_i^2 / n_i  (exact, rational)."""
    n = [int(x) for x in n]
    k = len(n)
    N = sum(n)
    total = Fraction(0)
    # sum over all ordered index tuples (i_1, ..., i_order) in {0..k-1}^order
    for tup in _index_tuples(k, order):
        denom = 1
        for i in tup:
            denom *= n[i]
        total += joint_cumulant_of_squares(N, n, tup) / denom
    return total


def _index_tuples(k: int, r: int) -> List[Tuple[int, ...]]:
    if r == 0:
        return [()]
    smaller = _index_tuples(k, r - 1)
    return [t + (i,) for t in smaller for i in range(k)]


def kappa_H(n: Sequence[int], order: int) -> Fraction:
    """kappa_order(H) = c^order * kappa_order(Q), c = 12 / [N(N+1)] (exact)."""
    n = [int(x) for x in n]
    N = sum(n)
    if order == 1:
        # E[H] = k - 1 exactly (affine reduction gives kappa_1(Q) then c*.. ,
        # but simplest exact statement is k-1).
        return Fraction(len(n) - 1)
    c = Fraction(12, N * (N + 1))
    return c ** order * kappa_Q(n, order)


def cumulants_closed_form(n: Sequence[int]) -> Dict[str, float]:
    """Return kappa1..4, gamma1, gamma2 of H as floats via the closed form.

    Parameters
    ----------
    n : sequence of int
        Group sample sizes.

    Returns
    -------
    dict with keys ``k1, k2, k3, k4, gamma1, gamma2`` (floats), plus the exact
    ``Fraction`` values under ``k1_exact .. k4_exact``.
    """
    k1 = kappa_H(n, 1)
    k2 = kappa_H(n, 2)
    k3 = kappa_H(n, 3)
    k4 = kappa_H(n, 4)
    # Degenerate designs (e.g. every group of size 1) make H constant, so
    # kappa2 = 0 and the standardized shapes are undefined; report NaN rather
    # than dividing by zero, while still returning the exact cumulants.
    if k2 == 0:
        gamma1 = gamma2 = float("nan")
    else:
        gamma1 = float(k3) / float(k2) ** 1.5
        gamma2 = float(k4) / float(k2) ** 2
    return {
        "k1": float(k1), "k2": float(k2), "k3": float(k3), "k4": float(k4),
        "gamma1": gamma1, "gamma2": gamma2,
        "k1_exact": k1, "k2_exact": k2, "k3_exact": k3, "k4_exact": k4,
    }


if __name__ == "__main__":
    for design in [(3, 3, 3), (5, 5, 5), (2, 3, 7), (3, 3, 3, 3)]:
        c = cumulants_closed_form(design)
        print(design, {k: round(v, 4) for k, v in c.items() if not k.endswith("exact")})
