"""Visualize the G1 combined 1D family — heat / advection / burgers / diff-sorp.

Reads the *_g1.h5 files produced by combine_g1.py and shows:
  - 4 sample heatmaps per system (sorted by coefficient when available)
  - Line plots at 3 time slices (t=0, T/2, T-1) with all 4 samples overlaid

Output: data/1d/g1_sample.png
"""
from __future__ import annotations
import os, sys
import numpy as np
import h5py
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "1d")


def load_g1(path: str, n_samples: int = 4):
    """Load the combined `*_g1.h5` file. Picks 4 evenly-spaced quantiles of coeff."""
    with h5py.File(path, "r") as f:
        coeff = np.asarray(f["coeff"][:])
        coeff_name = str(f.attrs.get("coeff_name", "?"))
        pde_class = int(f.attrs.get("pde_class", -1))
        pde_name = str(f.attrs.get("pde_name", "?"))
        n_total = f["u"].shape[0]
        if coeff.std() > 1e-6:
            order = np.argsort(coeff)
            picks = order[np.linspace(0, n_total - 1, n_samples).astype(int)]
        else:
            picks = np.linspace(0, n_total - 1, n_samples).astype(int)
        u = np.asarray([f["u"][i] for i in picks])
        c = coeff[picks]
        const_label = ""
        if coeff.std() < 1e-6:
            const_label = f"(coeff fixed; only IC varies)"
    return u, c, coeff_name, pde_class, pde_name, const_label


def plot_row(axes_row, name, eq, u, coeffs, cname, const_label, has_var):
    n_samples, T, X = u.shape
    t_picks = [0, T // 2, T - 1]
    t_labels = [f"t = {i}" for i in t_picks]

    ax = axes_row[0]
    lines = [name, "", eq, ""]
    if has_var:
        lines.append("samples (sorted by coeff):")
        for k, c in enumerate(coeffs):
            lines.append(f"  s{k}: {cname} = {c:.4f}")
    else:
        lines.append(const_label)
    lines.append("")
    lines.append(f"shape: {u.shape}")
    ax.text(0.0, 0.5, "\n".join(lines), transform=ax.transAxes,
              fontsize=10, va="center", family="monospace")
    ax.axis("off")

    vmax = float(np.abs(u).max())
    for s in range(n_samples):
        ax = axes_row[1 + s]
        ax.imshow(u[s], cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax,
                    extent=[0, X, T, 0])
        title = (f"s{s}: {cname}={coeffs[s]:.3g}" if has_var else f"s{s}")
        ax.set_title(title, fontsize=9)
        if s == 0: ax.set_ylabel("t", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    palette = plt.cm.viridis(np.linspace(0, 1, n_samples))
    for col_offset, t_pick, t_lab in zip(range(5, 8), t_picks, t_labels):
        ax = axes_row[col_offset]
        for s in range(n_samples):
            label = (f"{cname}={coeffs[s]:.3g}" if has_var else f"s{s}")
            ax.plot(u[s, t_pick], color=palette[s], linewidth=1.2,
                      label=label, alpha=0.85)
        ax.set_title(t_lab, fontsize=10)
        ax.set_xlim(0, X - 1)
        if col_offset == 5:
            ax.legend(fontsize=7, loc="best", framealpha=0.7)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
        if col_offset == 5: ax.set_ylabel("u(x)", fontsize=9)


def main():
    systems = [
        ("Heat",      os.path.join(OUT, "heat",      "heat_g1.h5"),
          r"$u_t = \nu u_{xx}$  ($\nu$ varies per traj)"),
        ("Advection", os.path.join(OUT, "advection", "advection_g1.h5"),
          r"$u_t + \beta u_x = 0$  (3 $\beta$ levels)"),
        ("Burgers",   os.path.join(OUT, "burgers",   "burgers_g1.h5"),
          r"$u_t + u u_x = \nu u_{xx}$  (4 $\nu$ levels)"),
        ("Diff-Sorp", os.path.join(OUT, "diff_sorp", "diff_sorp_g1.h5"),
          r"$u_t = D u_{xx} - R(u)$  (single configuration, IC varies)"),
    ]

    n_show = 4
    fig, axes = plt.subplots(len(systems), 8, figsize=(24, 3.2 * len(systems)),
                              gridspec_kw={"width_ratios":
                                          [1.6, 1, 1, 1, 1, 1.4, 1.4, 1.4]})

    for r, (name, path, eq) in enumerate(systems):
        if not os.path.exists(path):
            for c in range(8):
                axes[r, c].text(0.5, 0.5, "MISSING", ha="center", va="center",
                                  transform=axes[r, c].transAxes, fontsize=12, color="red")
                axes[r, c].axis("off")
            continue
        u, c, cname, pde_class, pde_name, const_label = load_g1(path, n_samples=n_show)
        has_var = bool(c.std() > 1e-6)
        plot_row(axes[r], name, eq, u, c, cname, const_label, has_var)

    fig.suptitle("G1 — combined 1D PDE family: 4 samples per system (heatmap + 3 time slices)",
                  fontsize=14, y=1.005)
    plt.tight_layout()
    out_path = os.path.join(OUT, "g1_sample.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
