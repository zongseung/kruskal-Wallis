"""
Scatter of the third cumulant (x-axis) vs the fourth cumulant (y-axis) for the
two parts of Table 2: three-group and four-group designs. Closed-form points
(Thm 3.12) are overlaid on the exact Table-2 values to show they coincide.

Run:
    uv run python examples/plot_k3_k4_axes.py
"""

from __future__ import annotations

from kw_approx.cumulants_closed_form import cumulants_closed_form

# Table 2: n-tuple -> (k3, k4) exact enumeration values.
THREE = {
    (2, 2, 2): (0.7184, -5.5492), (2, 2, 5): (2.9201, -1.7514),
    (2, 3, 4): (3.6882, -0.2633), (3, 3, 3): (4.1881, 0.1059),
    (2, 3, 7): (4.7777, 4.7218), (2, 4, 8): (5.7792, 10.1639),
    (2, 5, 8): (6.3094, 13.6032), (3, 4, 5): (6.2635, 9.8608),
    (4, 4, 4): (6.5402, 10.9622), (3, 5, 7): (7.4471, 17.3736),
    (4, 5, 6): (7.9870, 20.1054), (5, 5, 5): (8.1469, 21.0201),
}
FOUR = {
    (2, 2, 2, 2): (0.7905, -9.8805), (2, 2, 2, 4): (2.9152, -7.3784),
    (2, 2, 3, 3): (3.3556, -7.6041), (2, 2, 2, 6): (3.9959, -4.0232),
    (2, 2, 3, 4): (4.2349, -4.9435), (2, 2, 3, 5): (4.8599, -2.5655),
    (2, 3, 3, 3): (4.6883, -5.1392), (2, 3, 3, 4): (5.5780, -1.8953),
    (3, 3, 3, 3): (6.0420, -2.0627),
}


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    for table, color, name in [(THREE, "#1f77b4", "k=3"),
                               (FOUR, "#d62728", "k=4")]:
        tx = [v[0] for v in table.values()]        # exact k3
        ty = [v[1] for v in table.values()]        # exact k4
        cf = [cumulants_closed_form(n) for n in table]
        cx = [c["k3"] for c in cf]                  # closed-form k3
        cy = [c["k4"] for c in cf]                  # closed-form k4

        ax.scatter(tx, ty, s=120, facecolors="none", edgecolors=color,
                   linewidths=1.8, label=f"{name} exact")
        ax.scatter(cx, cy, s=40, marker="x", color=color,
                   label=f"{name} closed form")

    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel(r"$\kappa_3$")
    ax.set_ylabel(r"$\kappa_4$")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=.3)
    fig.tight_layout()

    import os
    os.makedirs("claudedocs", exist_ok=True)
    out = "claudedocs/k3_vs_k4.png"
    fig.savefig(out, dpi=140)
    print(f"Figure written to {out}")


if __name__ == "__main__":
    main()
