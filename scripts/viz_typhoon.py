"""Visualize the Digital Typhoon dataset (typhoon 198901): time frames + a trajectory,
each at native 512 and area-downsampled 256 / 128. IR brightness temperature (K), key 'Infrared'.
"""
import os, glob, datetime
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import h5py

PEEK = os.path.expanduser("~/scratch/typhoon_peek/WP/image/198901")
OUT = "/gpfs/radev/scratch/lu_lu/ss5235/WFAE/results/figures"; os.makedirs(OUT, exist_ok=True)
RES = [512, 256, 128]; CMAP = "turbo"
fs = sorted(glob.glob(os.path.join(PEEK, "*.h5")))
print(f"{len(fs)} frames")

def load(f):
    with h5py.File(f, "r") as h:
        return np.array(h["Infrared"]).astype(np.float32)

def down(img, r):
    if r == 512: return img
    t = torch.from_numpy(img)[None, None]
    return F.interpolate(t, size=(r, r), mode="area")[0, 0].numpy()

def ts(f):                                   # YYYYMMDDHH-... -> "MM-DD HHh"
    s = os.path.basename(f)[:10]
    return f"{s[4:6]}-{s[6:8]} {s[8:10]}h"

# shared color scale across everything (consistent K range)
allmin = min(load(fs[i]).min() for i in (0, len(fs)//2, -1))
allmax = max(load(fs[i]).max() for i in (0, len(fs)//2, -1))

# ---- Figure 1: different time frames (rows) x resolutions (cols) ----
pick = [0, len(fs)//3, 2*len(fs)//3, len(fs)-1]
fig, ax = plt.subplots(len(pick), len(RES), figsize=(3*len(RES), 3*len(pick)))
for i, idx in enumerate(pick):
    img = load(fs[idx])
    for j, r in enumerate(RES):
        ax[i, j].imshow(down(img, r), cmap=CMAP, vmin=allmin, vmax=allmax)
        ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        if i == 0: ax[i, j].set_title(f"{r}x{r}", fontsize=15, fontweight="bold")
    ax[i, 0].set_ylabel(ts(fs[idx]), fontsize=13)
fig.suptitle("Digital Typhoon 198901 — frames at native 512 vs downsampled 256 / 128", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(os.path.join(OUT, "typhoon_frames.png"), dpi=130)
print("wrote typhoon_frames.png")

# ---- Figure 2: trajectory (resolutions as rows) x time (cols) ----
T = 8; tidx = np.linspace(0, len(fs)-1, T).astype(int)
fig, ax = plt.subplots(len(RES), T, figsize=(2.1*T, 2.1*len(RES)))
imgs = [load(fs[k]) for k in tidx]
for r_i, r in enumerate(RES):
    for c, (k, img) in enumerate(zip(tidx, imgs)):
        ax[r_i, c].imshow(down(img, r), cmap=CMAP, vmin=allmin, vmax=allmax)
        ax[r_i, c].set_xticks([]); ax[r_i, c].set_yticks([])
        if r_i == 0: ax[r_i, c].set_title(ts(fs[k]), fontsize=10)
    ax[r_i, 0].set_ylabel(f"{r}x{r}", fontsize=14, fontweight="bold")
fig.suptitle("Digital Typhoon 198901 — trajectory (lifecycle) at 512 / 256 / 128", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(os.path.join(OUT, "typhoon_trajectory.png"), dpi=130)
print("wrote typhoon_trajectory.png")
print(f"K range used: [{allmin:.0f}, {allmax:.0f}]")
