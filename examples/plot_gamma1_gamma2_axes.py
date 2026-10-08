"""
Scatter of standardized skewness gamma1 (x-axis) vs excess kurtosis gamma2
(y-axis) for the two parts of Table 2: three-group and four-group designs.
Closed-form points (Thm 3.12) are overlaid on the exact Table-2 values; they
coincide because substituting the formulas reproduces the exact numbers.

Run:
    uv run python examples/plot_gamma1_gamma2_axes.py
"""

from __future__ import annotations

from kw_approx.cumulants_closed_form import cumulants_closed_form

# Table 2: n-tuple -> (gamma1, gamma2) exact enumeration values.
THREE = {
    (2, 2, 2): (0.2435, -1.3113), (2, 2, 5): (0.7477, -0.2848),
    (2, 3, 4): (0.8697, -0.0384), (3, 3, 3): (0.9336, 0.0143),
    (2, 3, 7): (1.0337, 0.6133), (2, 4, 8): (1.1665, 1.2034),
    (2, 5, 8): (1.2327, 1.5422), (3, 4, 5): (1.2017, 1.0912),
    (4, 4, 4): (1.2302, 1.1814), (3, 5, 7): (1.3330, 1.7525),
    (4, 5, 6): (1.3823, 1.9391), (5, 5, 5): (1.3969, 2.0024),
}
FOUR = {
    (2, 2, 2, 2): (0.1472, -1.0506), (2, 2, 2, 4): (0.4480, -0.6074),
    (2, 2, 3, 3): (0.4943, -0.5915), (2, 2, 2, 6): (0.5703, -0.3001),
    (2, 2, 3, 4): (0.5885, -0.3558), (2, 2, 3, 5): (0.6505, -0.1757),
    (2, 3, 3, 3): (0.6261, -0.3508), (2, 3, 3, 4): (0.7074, -0.1208),
    (3, 3, 3, 3): (0.7382, -0.1250),
}


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    for table, color, name in [(THREE, "#1f77b4", "k=3"),
                               (FOUR, "#d62728", "k=4")]:
        tx = [v[0] for v in table.values()]        # exact gamma1
        ty = [v[1] for v in table.values()]        # exact gamma2
        cf = [cumulants_closed_form(n) for n in table]
        cx = [c["gamma1"] for c in cf]             # closed-form gamma1
        cy = [c["gamma2"] for c in cf]             # closed-form gamma2

        ax.scatter(tx, ty, s=120, facecolors="none", edgecolors=color,
                   linewidths=1.8, label=f"{name} exact")
        ax.scatter(cx, cy, s=40, marker="x", color=color,
                   label=f"{name} closed form")

    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel(r"$\gamma_1$")
    ax.set_ylabel(r"$\gamma_2$")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=.3)
    fig.tight_layout()

    import os
    os.makedirs("claudedocs", exist_ok=True)
    out = "claudedocs/gamma1_vs_gamma2.png"
    fig.savefig(out, dpi=140)
    print(f"Figure written to {out}")


if __name__ == "__main__":
    main()
