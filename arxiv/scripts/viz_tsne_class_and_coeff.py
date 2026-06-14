"""t-SNE grid: rows = {class, heat ν, advection β, burgers ν}, cols = 5 methods.
Same shuffle as evaluate_1d.py so the class indices align with cached Z_val.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.data.g1 import load_g1_system, PDE_NAMES, PDE_CLASS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "probes", "g1")
METHODS = [("fae_recon", "V3-recon"), ("fae_vicreg", "V3+VICReg"),
            ("mlp", "MLP"), ("cnn", "CNN"), ("mae", "MAE")]

# Reproduce evaluate_1d's val coeff per system (same shuffle seed=0)
rng = np.random.default_rng(0)
val_coeff_per_sys = {}
val_count_per_sys = {}
for n in PDE_NAMES:
    d = load_g1_system(n)
    u = d["u"]; coeff = d["coeff"]
    n_total = u.shape[0]; n_val = n_total // 5
    perm = rng.permutation(n_total)
    coeff = coeff[perm]
    val_coeff_per_sys[n] = coeff[-n_val:]
    val_count_per_sys[n] = n_val

# In evaluate_1d, val_u/val_cls is concatenated in PDE_NAMES order
offsets = {}; s = 0
for n in PDE_NAMES:
    offsets[n] = (s, s + val_count_per_sys[n]); s += val_count_per_sys[n]

# t-SNE per method (cache to npz)
tsne_cache = {}
for m, _ in METHODS:
    cache = os.path.join(OUT, f"tsne2d_{m}.npz")
    if os.path.exists(cache):
        d = np.load(cache); tsne_cache[m] = d["tsne"]; continue
    emb_path = os.path.join(OUT, f"emb_{m}.npz")
    if not os.path.exists(emb_path): continue
    d = np.load(emb_path); Z = d["Z_val"]
    print(f"[{m}] PCA+TSNE on {Z.shape}", flush=True)
    Z50 = PCA(n_components=min(50, Z.shape[1])).fit_transform(Z)
    t = TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(Z50)
    np.savez(cache, tsne=t)
    tsne_cache[m] = t

# Plot: 4 rows x 5 cols (rows = class, heat ν, adv β, burg ν)
methods_present = [m for m, _ in METHODS if m in tsne_cache]
n = len(methods_present)
fig, axes = plt.subplots(4, n, figsize=(3.8 * n, 14))
row_titles = ["By PDE class", "Heat ν (log)", "Advection β (log)", "Burgers ν (log)"]
class_palette = ["#d62728", "#1f77b4", "#2ca02c", "#7f7f7f"]
class_labels = ["Heat", "Advection", "Burgers", "Diff-Sorp"]

for col, m in enumerate(methods_present):
    t = tsne_cache[m]
    # Load class labels
    d = np.load(os.path.join(OUT, f"emb_{m}.npz"))
    cls = d["val_cls"]
    label_disp = dict(METHODS)[m]

    # Row 0: class
    ax = axes[0, col]
    for cid in range(4):
        mask = cls == cid
        ax.scatter(t[mask, 0], t[mask, 1], s=8, c=class_palette[cid],
                    label=class_labels[cid], alpha=0.65, edgecolors="none")
    ax.set_title(f"{label_disp}", fontsize=12, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    if col == 0:
        ax.set_ylabel(row_titles[0], fontsize=11)
        ax.legend(fontsize=8, loc="best", markerscale=1.5)

    # Rows 1-3: per-system coefficient coloring (others gray)
    for sys_idx, (sys_name, coeff_name) in enumerate([
        ("heat", "ν"), ("advection", "β"), ("burgers", "ν")]):
        ax = axes[sys_idx + 1, col]
        lo, hi = offsets[sys_name]
        # Gray out non-system points
        not_sys = (cls != PDE_CLASS[sys_name])
        ax.scatter(t[not_sys, 0], t[not_sys, 1], s=5, c="lightgray",
                    alpha=0.25, edgecolors="none")
        # Color system points by log coefficient
        sys_mask = (cls == PDE_CLASS[sys_name])
        sys_t = t[sys_mask]
        # Match coefficient order — sys_mask preserves order in val_u, which
        # was concatenated in PDE_NAMES order, so sys indices map to
        # val_coeff_per_sys[sys_name] directly.
        sys_coeff = val_coeff_per_sys[sys_name][:sys_t.shape[0]]
        log_c = np.log10(np.maximum(sys_coeff, 1e-8))
        sc = ax.scatter(sys_t[:, 0], sys_t[:, 1], s=10, c=log_c,
                          cmap="viridis", alpha=0.85, edgecolors="none")
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(row_titles[sys_idx + 1], fontsize=11)
        if col == n - 1:
            cbar = plt.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
            cbar.set_label(f"log₁₀ {coeff_name}", fontsize=9)

plt.tight_layout()
out = os.path.join(OUT, "tsne_class_and_coeff.png")
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"saved {out} ({os.path.getsize(out)/1024:.0f} KB)")
