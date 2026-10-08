"""
Prove the closed form is an IDENTITY, not an approximation.

Table 2's skewness/kurtosis are exact enumeration values. The combinatorial
closed form does not approximate them -- it produces the *same rational number*.
To show this we redo the full enumeration of the H null distribution in exact
rational arithmetic (fractions.Fraction, no floating point, no rounding),
compute kappa3/kappa4/gamma1/gamma2 exactly, and compare to the closed form:
the difference is exactly 0 for every one of the 21 Table-2 designs.

Run:
    uv run python examples/verify_cumulants_exact_identity.py
"""

from __future__ import annotations

from fractions import Fraction

from kw_approx.exact import ExactDistribution
from kw_approx.cumulants_closed_form import kappa_H

DESIGNS = [
    (2, 2, 2), (2, 2, 5), (2, 3, 4), (3, 3, 3), (2, 3, 7), (2, 4, 8),
    (2, 5, 8), (3, 4, 5), (4, 4, 4), (3, 5, 7), (4, 5, 6), (5, 5, 5),
    (2, 2, 2, 2), (2, 2, 2, 4), (2, 2, 3, 3), (2, 2, 2, 6), (2, 2, 3, 4),
    (2, 2, 3, 5), (2, 3, 3, 3), (2, 3, 3, 4), (3, 3, 3, 3),
]


def exact_cumulants_by_enumeration(n):
    """kappa1..4 of H as exact Fractions, via full rational enumeration."""
    N = sum(n)
    ed = ExactDistribution(list(n))
    counts = ed._enumerate_rank_sums()          # {rank_sums: integer count}
    total = ed._multinomial_coefficient()       # integer
    c = Fraction(12, N * (N + 1))

    # exact rational raw moments E[H^m]
    raw = {m: Fraction(0) for m in range(1, 5)}
    for rank_sums, cnt in counts.items():
        A = sum(Fraction(R * R, ni) for R, ni in zip(rank_sums, n))
        H = c * A - 3 * (N + 1)                 # exact Fraction value of H
        w = Fraction(cnt, total)
        Hp = H
        for m in range(1, 5):
            raw[m] += w * Hp
            Hp *= H

    mu = raw[1]
    m2 = raw[2] - mu**2
    m3 = raw[3] - 3 * mu * raw[2] + 2 * mu**3
    m4 = raw[4] - 4 * mu * raw[3] + 6 * mu**2 * raw[2] - 3 * mu**4
    k1, k2, k3 = mu, m2, m3
    k4 = m4 - 3 * m2**2
    return k1, k2, k3, k4


def main():
    print(f"{'design':<12}{'k3 identical':>14}{'k4 identical':>14}"
          f"{'gamma1 (closed, 12 dp)':>26}{'gamma2 (closed, 12 dp)':>26}")
    print("-" * 92)
    all_identical = True
    for n in DESIGNS:
        # exact enumeration (rational)
        _, _, k3e, k4e = exact_cumulants_by_enumeration(n)
        # closed form (rational)
        k3c = kappa_H(n, 3)
        k4c = kappa_H(n, 4)
        k2c = kappa_H(n, 2)

        same3 = (k3e == k3c)
        same4 = (k4e == k4c)
        all_identical = all_identical and same3 and same4

        # gamma from the exact rationals, printed to 12 decimals
        g1 = float(k3c) / float(k2c) ** 1.5
        g2 = float(k4c) / float(k2c) ** 2

        label = ",".join(map(str, n))
        print(f"{label:<12}{('YES' if same3 else 'NO'):>14}"
              f"{('YES' if same4 else 'NO'):>14}"
              f"{g1:>26.12f}{g2:>26.12f}")

    print("-" * 92)
    print(f"All 21 designs: kappa3 and kappa4 are EXACTLY equal (Δ = 0)? "
          f"{'YES' if all_identical else 'NO'}")

    # Spotlight the (5,5,5) row the reviewer mentioned.
    n = (5, 5, 5)
    k2, k3, k4 = kappa_H(n, 2), kappa_H(n, 3), kappa_H(n, 4)
    print(f"\n(5,5,5) exact rationals:")
    print(f"  kappa2 = {k2}  = {float(k2):.12f}")
    print(f"  kappa3 = {k3}  = {float(k3):.12f}")
    print(f"  kappa4 = {k4}  = {float(k4):.12f}")
    print(f"  gamma1 = kappa3 / kappa2^1.5 = {float(k3)/float(k2)**1.5:.12f}  "
          f"(Table 2: 1.3969)")
    print(f"  gamma2 = kappa4 / kappa2^2   = {float(k4)/float(k2)**2:.12f}  "
          f"(Table 2: 2.0024)")


if __name__ == "__main__":
    main()
