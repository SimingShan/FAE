"""Visualize the point-FAE local-neighbourhood grouping on NS data.

Shows how `LocalGroupEmbed` (src/models/fae.py) works: sparse sensors are drawn over the field, each
sensor token gathers its k nearest neighbours (the SAME `knn_index` used in training), and the mini-
PointNet embeds [relative-coord, neighbour-value] per group. No model weights needed — pure geometry.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data.preprocessed import PDEDataset
from src.data.well2d import make_coords_2d_hw
from src.models.fae import knn_index

SAMPLE, N_SENS, K, SEED = 7, 300, 8, 0
rng = np.random.default_rng(SEED)

ds = PDEDataset("ns", "train", mode="single")
x, _ = ds[SAMPLE]                                            # (C,H,W) normalized
field = x[0].numpy()                                         # smoke channel
H, W = field.shape

idx = rng.choice(H * W, N_SENS, replace=False)              # discrete sensors = random grid subset
coords = make_coords_2d_hw(H, W)[idx]                        # (N,2) normalized (row,col), as the model sees
srow, scol = idx // W, idx % W                               # pixel positions for plotting
svals = field[srow, scol]                                   # sensor values

nbr = knn_index(coords[None], K)[0].numpy()                 # (N,K) faithful k-NN (incl. self)
queries = [40, 150, 250]                                     # a few example sensor tokens
qcol = ["#00e5ff", "#ffe600", "#ff2d95"]

fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.4))

# panel 1: field + all sparse sensors
ax[0].imshow(field, cmap="magma", interpolation="nearest")
ax[0].scatter(scol, srow, s=12, c="white", edgecolor="k", linewidths=0.3)
ax[0].set_title(f"NS smoke field + {N_SENS} sparse sensors\n(discrete = random grid subset)", fontsize=11)

# panel 2: each query gathers its k-NN group (lines centre -> neighbours)
ax[1].imshow(field, cmap="magma", interpolation="nearest", alpha=0.55)
ax[1].scatter(scol, srow, s=8, c="white", edgecolor="none", alpha=0.5)
for q, col in zip(queries, qcol):
    for j in nbr[q]:
        ax[1].plot([scol[q], scol[j]], [srow[q], srow[j]], c=col, lw=1.3, alpha=0.9, zorder=2)
    ax[1].scatter(scol[nbr[q]], srow[nbr[q]], s=42, facecolor="none", edgecolor=col, linewidths=1.8, zorder=3)
    ax[1].scatter(scol[q], srow[q], s=90, marker="*", c=col, edgecolor="k", linewidths=0.5, zorder=4)
ax[1].set_title(f"Per-token k-NN groups (k={K})\n★ = token centre, ○ = its neighbours", fontsize=11)

# panel 3: one group in RELATIVE coords coloured by value = the mini-PointNet input
q = queries[1]
rel = (coords[nbr[q]] - coords[q]).numpy()                  # (K,2) relative (row,col)
gv = field[srow[nbr[q]], scol[nbr[q]]]
sc = ax[2].scatter(rel[:, 1], -rel[:, 0], c=gv, s=240, cmap="magma", edgecolor="k", linewidths=0.6)
ax[2].scatter(0, 0, s=320, marker="*", c=qcol[1], edgecolor="k", linewidths=0.8, zorder=5)
ax[2].axhline(0, c="0.7", lw=0.6); ax[2].axvline(0, c="0.7", lw=0.6)
ax[2].set_aspect("equal"); ax[2].set_title("One group, relative coords\nPointNet input = [Δcoord, value]", fontsize=11)
ax[2].set_xlabel("Δcol"); ax[2].set_ylabel("Δrow")
plt.colorbar(sc, ax=ax[2], fraction=0.046, label="neighbour value")

for a in ax[:2]:
    a.set_xticks([]); a.set_yticks([])
plt.tight_layout()
out = "results/figures/knn_grouping_ns.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved {out}  (field {H}x{W}, {N_SENS} sensors, k={K})")
