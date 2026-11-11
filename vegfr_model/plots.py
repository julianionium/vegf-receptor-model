"""Figures. Each function takes the DataFrame from analysis/surrogate and saves a PNG."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def timecourse(df, path, title):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df.time_hours, df.VEGFR1_bound_fraction, label="VEGFR1 bound fraction")
    ax.plot(df.time_hours, df.VEGFR2_bound_fraction, label="VEGFR2 bound fraction")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Fraction of receptors bound")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def dose_response(df, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (name, g), color in zip(df.groupby("cells", sort=False), ["tab:blue", "tab:red"]):
        label = f"{name} ({g.VEGFR1_per_cell.iloc[0]/1000:.0f}k / {g.VEGFR2_per_cell.iloc[0]/1000:.0f}k)"
        ax.plot(g.dose_pM, g.VEGFR2_bound_with_R1, color=color, label=f"{label}, with VEGFR1")
        ax.plot(g.dose_pM, g.VEGFR2_bound_no_R1, color=color, ls="--", label=f"{label}, no VEGFR1")
    ax.set_xscale("log")
    ax.set_xlabel("VEGF added (pM)")
    ax.set_ylabel("Fraction of VEGFR2 bound")
    ax.set_title("VEGFR1 only competes when receptors are plentiful")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def competition_map(df, path):
    levels = sorted(df.VEGFR2_per_cell.unique())
    fig, axes = plt.subplots(1, len(levels), figsize=(11, 4.5), sharey=True)
    for ax, r2 in zip(axes, levels):
        g = df[df.VEGFR2_per_cell == r2]
        grid = g.pivot(index="VEGFR1_to_VEGFR2_ratio", columns="dose_pM", values="percent_drop")
        im = ax.pcolormesh(grid.columns, grid.index, grid.values, shading="auto",
                           cmap="viridis", vmin=0, vmax=100)
        cs = ax.contour(grid.columns, grid.index, grid.values, levels=[10, 25, 50],
                        colors="white", linewidths=1)
        ax.clabel(cs, fmt="%d%%", fontsize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("VEGF added (pM)")
        ax.set_title(f"VEGFR2 = {r2:,.0f} per cell")
    axes[0].set_ylabel("VEGFR1 : VEGFR2 ratio")
    fig.colorbar(im, ax=axes, label="% drop in VEGFR2 binding caused by VEGFR1")
    fig.suptitle("VEGFR1 acts as a decoy only with many receptors and little VEGF")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def sensitivity(df, path, subtitle):
    order = df.assign(size=np.maximum(df.percent_change_at_minus.abs(),
                                      df.percent_change_at_plus.abs())).sort_values("size")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(order.input, order.percent_change_at_minus, color="tab:blue", label="input -50%")
    ax.barh(order.input, order.percent_change_at_plus, color="tab:orange", label="input +50%")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("% change in VEGFR2 binding")
    ax.set_title(f"What VEGFR2 binding is most sensitive to ({subtitle})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def surrogate_fit(test, scores, path):
    cols = {"Linear regression": "linear_regression_prediction",
            "Neural network": "neural_network_prediction"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, (name, col) in zip(axes, cols.items()):
        mae = scores.set_index("model").loc[name, "test_MAE"]
        ax.scatter(test.true_vegfr2_bound, test[col], s=4, alpha=0.4)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_title(f"{name}\nMAE = {mae:.3f}")
        ax.set_xlabel("True VEGFR2 bound (model)")
    axes[0].set_ylabel("Predicted VEGFR2 bound")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
