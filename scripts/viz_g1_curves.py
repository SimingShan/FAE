"""1D u(x) line plots for the G1 v2 dataset. Per system: 6 curves (low→high
coefficient), each showing u(x) overlaid at 6 timesteps. Also a separate
"trajectory waterfall" plot per system at fixed coefficient extremes.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/data/1d"

SYSTEMS = [
    ("heat", "Heat:  $u_t = \\nu u_{xx}$", "ν", "logU[1e-3, 1e-1]"),
    ("advection", "Advection:  $u_t = -\\beta u_x$", "β", "U[0.1, 4.0]"),
    ("burgers", "Burgers:  $u_t + u u_x = \\nu u_{xx}$", "ν", "logU[1e-4, 1e-2]"),
    ("reaction_diffusion", "Allen-Cahn:  $u_t = D u_{xx} + u - u^3$", "D",
       "logU[1e-4, 1e-2]"),
]

# =====================================================================
# Figure 1: per system, 6 columns × 6 coefficient samples,
#           each column shows u(x) overlaid at 6 timesteps
# =====================================================================
T_SAMPLES = [0, 20, 40, 60, 80, 99]
N_COLS = 6

fig, axes = plt.subplots(4, N_COLS, figsize=(22, 12), sharex=True)
for r, (sys_name, title, sym, rng_txt) in enumerate(SYSTEMS):
    with h5py.File(f"{OUT}/{sys_name}/{sys_name}_g1.h5", "r") as f:
        coeffs = f["coeff"][:]
        order = np.argsort(coeffs)
        positions = np.linspace(50, len(order) - 50, N_COLS).astype(int)
        picks = [int(order[p]) for p in positions]
        X = f["u"].shape[-1]
        x = np.linspace(0, 1, X, endpoint=False)
        for c, idx in enumerate(picks):
            ax = axes[r, c]
            u = f["u"][idx]
            colors = cm.viridis(np.linspace(0.1, 0.95, len(T_SAMPLES)))
            for ti, color in zip(T_SAMPLES, colors):
                ax.plot(x, u[ti], color=color, linewidth=1.0,
                          alpha=0.85)
            ax.set_title(f"{sym}={coeffs[idx]:.4g}", fontsize=9)
            ax.set_ylim(u.min() - 0.05 * abs(u.min()),
                          u.max() + 0.05 * abs(u.max()))
            ax.grid(alpha=0.25, linewidth=0.5)
            if c == 0:
                ax.set_ylabel(f"u(x, t)\n{sys_name}", fontsize=10)
            if r == 3:
                ax.set_xlabel("x")
# Legend across the top
for ti, color in zip(T_SAMPLES, cm.viridis(np.linspace(0.1, 0.95, len(T_SAMPLES)))):
    axes[0, -1].plot([], [], color=color, label=f"t-idx {ti}")
axes[0, -1].legend(fontsize=7, loc="upper right", framealpha=0.85,
                      title="time", title_fontsize=7)
fig.suptitle("G1 v2 — 1D field u(x) at 6 timesteps, sorted columns by coefficient",
              fontsize=14, y=1.005)
plt.tight_layout()
plt.savefig(f"{OUT}/g1_curves.png", dpi=110, bbox_inches="tight")
print(f"saved {OUT}/g1_curves.png")
plt.close(fig)

# =====================================================================
# Figure 2: waterfall — show how u(x, t) evolves for one
#           low-coefficient and one high-coefficient sample per system
# =====================================================================
fig, axes = plt.subplots(4, 2, figsize=(16, 14))
WATER_T = list(range(0, 100, 8))                                # 13 lines
for r, (sys_name, title, sym, rng_txt) in enumerate(SYSTEMS):
    with h5py.File(f"{OUT}/{sys_name}/{sys_name}_g1.h5", "r") as f:
        coeffs = f["coeff"][:]
        order = np.argsort(coeffs)
        idx_lo = int(order[100])                                # near-min
        idx_hi = int(order[-100])                               # near-max
        X = f["u"].shape[-1]
        x = np.linspace(0, 1, X, endpoint=False)
        for c, (idx, label) in enumerate([(idx_lo, "low"), (idx_hi, "high")]):
            ax = axes[r, c]
            u = f["u"][idx]
            offsets = np.arange(len(WATER_T)) * (u.max() - u.min()) * 0.35
            colors = cm.viridis(np.linspace(0.05, 0.95, len(WATER_T)))
            for off, ti, color in zip(offsets, WATER_T, colors):
                ax.plot(x, u[ti] + off, color=color, linewidth=0.9)
                ax.text(-0.02, off + u[ti].mean(),
                          f"t={ti}", ha="right", va="center", fontsize=7,
                          color=color)
            ax.set_title(f"{title}    {sym} = {coeffs[idx]:.4g} ({label})",
                          fontsize=10)
            ax.set_xlim(-0.05, 1.0)
            ax.set_xlabel("x"); ax.grid(alpha=0.25, linewidth=0.5)
            ax.set_yticks([])
fig.suptitle("G1 v2 — Time-waterfall of u(x, t) (low vs high coefficient extremes)",
              fontsize=14, y=1.005)
plt.tight_layout()
plt.savefig(f"{OUT}/g1_waterfall.png", dpi=110, bbox_inches="tight")
print(f"saved {OUT}/g1_waterfall.png")
plt.close(fig)

# =====================================================================
# Figure 3: IC + final-frame line plots, 4 systems × 6 coefficients
# =====================================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 7))
for c, (sys_name, title, sym, _) in enumerate(SYSTEMS):
    with h5py.File(f"{OUT}/{sys_name}/{sys_name}_g1.h5", "r") as f:
        coeffs = f["coeff"][:]
        order = np.argsort(coeffs)
        positions = np.linspace(50, len(order) - 50, 6).astype(int)
        picks = [int(order[p]) for p in positions]
        X = f["u"].shape[-1]
        x = np.linspace(0, 1, X, endpoint=False)
        for r, ti in [(0, 0), (1, 99)]:
            ax = axes[r, c]
            colors = cm.plasma(np.linspace(0.05, 0.95, len(picks)))
            for idx, color in zip(picks, colors):
                u = f["u"][idx, ti]
                ax.plot(x, u, color=color, linewidth=1.2, alpha=0.9,
                          label=f"{sym}={coeffs[idx]:.3g}")
            ax.set_title(f"{title}   (t-idx {ti})", fontsize=10)
            ax.grid(alpha=0.25, linewidth=0.5)
            ax.set_xlabel("x")
            if c == 0:
                ax.set_ylabel(f"u(x, t={ti})", fontsize=10)
            if r == 0 and c == 0:
                ax.legend(fontsize=6, loc="best", ncol=1)
fig.suptitle("G1 v2 — initial (top) vs final (bottom) field, 6 coefficient sweeps",
              fontsize=14, y=1.005)
plt.tight_layout()
plt.savefig(f"{OUT}/g1_ic_vs_final.png", dpi=110, bbox_inches="tight")
print(f"saved {OUT}/g1_ic_vs_final.png")
plt.close(fig)
