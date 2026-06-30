"""Schematic of the FAE training objective on NS: what the encoder SEES (sparse INPUT sensors) vs what
the decoder must PREDICT (OUTPUT query targets) — query_mode 'global' vs 'neighborhood'.
Mirrors scripts/train/train_fae.py exactly: neighborhood = DENSE filled disks around a few centre sensors.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data.preprocessed import PDEDataset

SAMPLE, N_SENS, N_QUERY, RADIUS, SEED = 7, 256, 1024, 8, 0
torch.manual_seed(SEED)
ds = PDEDataset("ns", "train", mode="single")
x, _ = ds[SAMPLE]; field = x[0].numpy(); H, W = field.shape; NPIX = H * W

iA = torch.randperm(NPIX)[:N_SENS]                                  # INPUT sensors (encoder sees)
sr, sc = (iA // W).numpy(), (iA % W).numpy()

q_global = torch.randperm(NPIX)[:N_QUERY]                           # GLOBAL: uniform full-field targets
gr, gc = (q_global // W).numpy(), (q_global % W).numpy()

# NEIGHBORHOOD: DENSE filled disks around N_CENTERS centre sensors (identical to train_fae query_idx)
o = torch.arange(-RADIUS, RADIUS + 1)
dg = torch.stack(torch.meshgrid(o, o, indexing="ij"), -1).reshape(-1, 2)
DISK = dg[(dg ** 2).sum(-1) <= RADIUS ** 2]                         # (D,2)
N_CENTERS = max(1, N_QUERY // DISK.size(0))
centers = iA[torch.randperm(N_SENS)[:N_CENTERS]]
cr, cc = (centers // W)[:, None], (centers % W)[:, None]
nr = (cr + DISK[:, 0][None]).clamp(0, H - 1).reshape(-1).numpy()
nc = (cc + DISK[:, 1][None]).clamp(0, W - 1).reshape(-1).numpy()
ex_r, ex_c = int(centers[0] // W), int(centers[0] % W)             # one example centre to annotate

fig, ax = plt.subplots(1, 2, figsize=(12.6, 6.3))
# --- GLOBAL ---
ax[0].imshow(field, cmap="magma", interpolation="nearest", alpha=0.5)
ax[0].scatter(gc, gr, s=6, c="#00e5ff", alpha=0.55, linewidths=0, label=f"OUTPUT: decoder targets (n_query={N_QUERY})")
ax[0].scatter(sc, sr, s=42, c="white", edgecolor="k", linewidths=0.9, label=f"INPUT: encoder sensors ({N_SENS})")
ax[0].set_title("query_mode = GLOBAL\ndecoder predicts the WHOLE field (sparse-in -> dense-out)", fontsize=11.5)
# --- NEIGHBORHOOD (dense) ---
ax[1].imshow(field, cmap="magma", interpolation="nearest", alpha=0.5)
ax[1].scatter(nc, nr, s=7, c="#ffe600", alpha=0.7, linewidths=0, label=f"OUTPUT: dense disks ({N_CENTERS} centres x ~{DISK.size(0)} pts)")
ax[1].scatter(sc, sr, s=42, c="white", edgecolor="k", linewidths=0.9, label=f"INPUT: encoder sensors ({N_SENS})")
ax[1].add_patch(plt.Circle((ex_c, ex_r), RADIUS, fill=False, ec="#00e5ff", lw=2.0, zorder=5))
ax[1].annotate("one sensor ->\npredict its dense\nneighbourhood", (ex_c, ex_r), (ex_c + 16, ex_r - 18),
               color="#00e5ff", fontsize=9, ha="left", arrowprops=dict(arrowstyle="->", color="#00e5ff"))
ax[1].set_title(f"query_mode = NEIGHBORHOOD (r={RADIUS})\ndense local patch around each centre sensor", fontsize=11.5)
for a in ax:
    a.set_xticks([]); a.set_yticks([]); a.legend(loc="upper right", fontsize=8, framealpha=0.92)
plt.suptitle("FAE objective on NS — INPUT sensors (encoder, o) vs OUTPUT targets (decoder, .)", fontsize=12.5, y=1.02)
plt.tight_layout()
out = "results/figures/query_schematic_ns.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved {out}  (sensors={N_SENS}, n_query={N_QUERY}, radius={RADIUS}, centres={N_CENTERS}, disk={DISK.size(0)})")
