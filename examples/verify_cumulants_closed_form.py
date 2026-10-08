"""
Verify the closed-form combinatorial third/fourth cumulants of H against
(a) the paper's Table 2 (exact, full enumeration) and
(b) the code's own exact enumeration (kw_approx.moments.KWMoments),
then plot the kappa3 / kappa4 agreement ("the 3rd and 4th cumulants coincide").

Theorem 3.12 (Murakami, Lee & Ha 2026) + companion "Explicit derivation" note
give kappa_m(H) = c^m kappa_m(Q) with the even joint moments of the centred
rank sums evaluated in closed form by the multivariate hypergeometric formula.
This script shows that route is numerically identical to enumeration.

Run:
    uv run python examples/verify_cumulants_closed_form.py
"""

from __future__ import annotations

import numpy as np

from kw_approx.cumulants_closed_form import cumulants_closed_form, kappa_H
from kw_approx.moments import KWMoments


# Table 2 of the paper: n-tuple -> (k2, k3, k4, gamma1, gamma2), all exact.
TABLE2 = {
    (2, 2, 2): (2.0571, 0.7184, -5.5492, 0.2435, -1.3113),
    (2, 2, 5): (2.4800, 2.9201, -1.7514, 0.7477, -0.2848),
    (2, 3, 4): (2.6200, 3.6882, -0.2633, 0.8697, -0.0384),
    (3, 3, 3): (2.7200, 4.1881, 0.1059, 0.9336, 0.0143),
    (2, 3, 7): (2.7747, 4.7777, 4.7218, 1.0337, 0.6133),
    (2, 4, 8): (2.9062, 5.7792, 10.1639, 1.1665, 1.2034),
    (2, 5, 8): (2.9700, 6.3094, 13.6032, 1.2327, 1.5422),
    (3, 4, 5): (3.0062, 6.2635, 9.8608, 1.2017, 1.0912),
    (4, 4, 4): (3.0462, 6.5402, 10.9622, 1.2302, 1.1814),
    (3, 5, 7): (3.1486, 7.4471, 17.3736, 1.3330, 1.7525),
    (4, 5, 6): (3.2200, 7.9870, 20.1054, 1.3823, 1.9391),
    (5, 5, 5): (3.2400, 8.1469, 21.0201, 1.3969, 2.0024),
    (2, 2, 2, 2): (3.0667, 0.7905, -9.8805, 0.1472, -1.0506),
    (2, 2, 2, 4): (3.4855, 2.9152, -7.3784, 0.4480, -0.6074),
    (2, 2, 3, 3): (3.5855, 3.3556, -7.6041, 0.4943, -0.5915),
    (2, 2, 2, 6): (3.6615, 3.9959, -4.0232, 0.5703, -0.3001),
    (2, 2, 3, 4): (3.7273, 4.2349, -4.9435, 0.5885, -0.3558),
    (2, 2, 3, 5): (3.8215, 4.8599, -2.5655, 0.6505, -0.1757),
    (2, 3, 3, 3): (3.8273, 4.6883, -5.1392, 0.6261, -0.3508),
    (2, 3, 3, 4): (3.9615, 5.5780, -1.8953, 0.7074, -0.1208),
    (3, 3, 3, 3): (4.0615, 6.0420, -2.0627, 0.7382, -0.1250),
}


def balanced_k3_formula(k: int, n: int) -> float:
    """Closed form eq:k3-balanced from the companion note (balanced n_i = n)."""
    num = (8 * (k - 1) * (n - 1) *
           (35 * k**3 * n**4 - 70 * k**3 * n**3 + 91 * k**2 * n**3
            - 148 * k**2 * n**2 + 62 * k * n**2 - 30 * k * n + 60))
    den = 35 * k * n**3 * (k * n + 1)**2
    return num / den


def main() -> None:
    rows = []
    print(f"{'design':<14}{'':2}"
          f"{'k3_table':>10}{'k3_closed':>11}{'k3_enum':>10}{'|Δ|max':>9}   "
          f"{'k4_table':>10}{'k4_closed':>11}{'k4_enum':>10}{'|Δ|max':>9}")
    print("-" * 108)

    max_diff_all = 0.0
    for n, (k2, k3, k4, g1, g2) in TABLE2.items():
        cf = cumulants_closed_form(n)                       # closed-form (Thm 3.12)
        mom = KWMoments(list(n), max_moment=4, use_exact=True)  # code enumeration
        cum = mom.cumulants
        k3_enum, k4_enum = cum[3], cum[4]

        d3 = max(abs(cf["k3"] - k3), abs(cf["k3"] - k3_enum))
        d4 = max(abs(cf["k4"] - k4), abs(cf["k4"] - k4_enum))
        max_diff_all = max(max_diff_all, abs(cf["k3"] - k3_enum), abs(cf["k4"] - k4_enum))

        label = ",".join(map(str, n))
        print(f"{label:<14}{'':2}"
              f"{k3:>10.4f}{cf['k3']:>11.4f}{k3_enum:>10.4f}{d3:>9.1e}   "
              f"{k4:>10.4f}{cf['k4']:>11.4f}{k4_enum:>10.4f}{d4:>9.1e}")

        rows.append(dict(design=label, k=len(n),
                         k3_table=k3, k3_closed=cf["k3"], k3_enum=k3_enum,
                         k4_table=k4, k4_closed=cf["k4"], k4_enum=k4_enum,
                         g1_table=g1, g1_closed=cf["gamma1"],
                         g2_table=g2, g2_closed=cf["gamma2"]))

    print("-" * 108)
    print(f"Max |closed-form − enumeration| over all designs: {max_diff_all:.3e}")

    # Balanced-case explicit formula check (eq:k3-balanced).
    print("\nBalanced eq:k3-balanced vs combinatorial closed form:")
    for (k, n) in [(3, 3), (3, 5), (4, 3), (3, 10), (4, 8)]:
        design = tuple([n] * k)
        cf = float(kappa_H(design, 3))
        formula = balanced_k3_formula(k, n)
        print(f"  k={k}, n={n:<3} -> eq:k3-balanced={formula:.6f}  "
              f"combinatorial={cf:.6f}  |Δ|={abs(cf-formula):.2e}")

    _plot(rows)


def _plot(rows) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"\n[plot skipped: matplotlib unavailable: {exc}]")
        return

    labels = [r["design"] for r in rows]
    x = np.arange(len(rows))
    k3_tab = [r["k3_table"] for r in rows]
    k3_cf = [r["k3_closed"] for r in rows]
    k4_tab = [r["k4_table"] for r in rows]
    k4_cf = [r["k4_closed"] for r in rows]
    colors = ["#1f77b4" if r["k"] == 3 else "#d62728" for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    # (0,0) kappa3 across designs
    ax = axes[0, 0]
    ax.plot(x, k3_tab, "o", ms=9, mfc="none", mec="k", mew=1.6,
            label="Table 2 (exact enumeration)")
    ax.plot(x, k3_cf, "x", ms=8, color="#1f77b4",
            label="Closed form (Thm 3.12)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel(r"$\kappa_3(H)$"); ax.set_title(r"Third cumulant $\kappa_3(H)$")
    ax.legend(fontsize=9); ax.grid(alpha=.3)

    # (0,1) kappa4 across designs
    ax = axes[0, 1]
    ax.plot(x, k4_tab, "o", ms=9, mfc="none", mec="k", mew=1.6,
            label="Table 2 (exact enumeration)")
    ax.plot(x, k4_cf, "x", ms=8, color="#d62728",
            label="Closed form (Thm 3.12)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel(r"$\kappa_4(H)$"); ax.set_title(r"Fourth cumulant $\kappa_4(H)$")
    ax.legend(fontsize=9); ax.grid(alpha=.3)

    # (1,0) identity scatter: closed form vs enumeration (both k3 and k4)
    ax = axes[1, 0]
    allx = k3_tab + k4_tab
    ally = k3_cf + k4_cf
    ax.scatter(k3_tab, k3_cf, c=colors, marker="^", s=55, label=r"$\kappa_3$")
    ax.scatter(k4_tab, k4_cf, c=colors, marker="s", s=45, label=r"$\kappa_4$")
    lo, hi = min(allx + ally), max(allx + ally)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xlabel("Exact enumeration (Table 2)")
    ax.set_ylabel("Closed form (Thm 3.12)")
    ax.set_title("Closed form vs enumeration\n(blue = 3 groups, red = 4 groups)")
    ax.legend(fontsize=9); ax.grid(alpha=.3)

    # (1,1) asymptotic convergence of balanced k=3 to chi^2_{k-1}
    ax = axes[1, 1]
    ns = list(range(2, 41))
    k3_bal = [float(kappa_H(tuple([n] * 3), 3)) for n in ns]
    k4_bal = [float(kappa_H(tuple([n] * 3), 4)) for n in ns]
    ax.plot(ns, k3_bal, "-o", ms=3, color="#1f77b4", label=r"$\kappa_3(H)$, $n{=}n{=}n$")
    ax.plot(ns, k4_bal, "-o", ms=3, color="#d62728", label=r"$\kappa_4(H)$, $n{=}n{=}n$")
    ax.axhline(8 * (3 - 1), ls="--", color="#1f77b4", alpha=.7,
               label=r"$\kappa_3(\chi^2_2)=16$")
    ax.axhline(48 * (3 - 1), ls="--", color="#d62728", alpha=.7,
               label=r"$\kappa_4(\chi^2_2)=96$")
    ax.set_xlabel("balanced group size $n$")
    ax.set_ylabel("cumulant")
    ax.set_title(r"Convergence to $\chi^2_{k-1}$ (Thm 3.13), $k=3$")
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    fig.suptitle("Kruskal–Wallis: closed-form combinatorial "
                 r"$\kappa_3,\kappa_4$ reproduce the exact cumulants",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = "claudedocs/cumulants_closed_form_verification.png"
    import os
    os.makedirs("claudedocs", exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nFigure written to {out}")


if __name__ == "__main__":
    main()
