"""What MAE/JEPA actually ingest in the sparse regime: linear interpolation of K sensors -> dense grid
(the architecture-axis handicap behind fig3). Per dataset: ground truth | interp@64 | @256 | @1024 | @4096.
Shear is shown at native 128x256.  python scripts/figs/fig_interp_input.py --dataset ns
"""
import os, sys, glob, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from src.plotstyle import apply
apply()
from scripts.eval.probe_all import get_data, _frame0, fae_hw

CMAP = {"shear": "coolwarm", "ns": "coolwarm", "typhoon": "Greys_r"}
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["shear", "typhoon", "ns"], required=True)
ap.add_argument("--ch", type=int, default=0)
args = ap.parse_args()
ds = args.dataset; SIDE = 128; CH = args.ch; cmap = CMAP[ds]
KS = [64, 256, 1024, 4096]

fck = sorted(glob.glob(f"results/checkpoints/{ds}/fae/*_s0.pt"))[0]
H, W = fae_hw(fck, SIDE)
dset = get_data(ds, "valid" if ds == "shear" else "test", list((H, W)) if (H, W) != (SIDE, SIDE) else SIDE)
field = _frame0(dset[len(dset) // 3][0].unsqueeze(0))[0, CH].numpy()      # (H,W)


def norm(x): return (x - x.mean()) / (x.std() + 1e-8) if cmap != "Greys_r" else x


vis = norm(field)
if cmap == "Greys_r":
    lo, hi = np.percentile(vis, 1), np.percentile(vis, 99)
else:
    hi = float(np.percentile(np.abs(vis), 99)) or 1.0; lo = -hi
kw = dict(cmap=cmap, vmin=lo, vmax=hi, aspect="equal")

fig, ax = plt.subplots(1, len(KS) + 1, figsize=(2.6 * (len(KS) + 1), 2.6 * (H / W) + 0.6))
ax[0].imshow(vis, **kw); ax[0].set_title("ground truth", fontsize=13)
gy, gx = np.mgrid[0:H, 0:W]; flat = field.reshape(-1); g = np.random.default_rng(0)
for i, K in enumerate(KS):
    idx = g.choice(H * W, K, replace=False); ys, xs = np.divmod(idx, W)
    grid = griddata(np.stack([ys, xs], 1), flat[idx], (gy, gx), method="linear", fill_value=float(flat[idx].mean()))
    ax[i + 1].imshow(norm(grid), **kw); ax[i + 1].set_title(f"{K} sensors", fontsize=13)
for a in ax:
    a.set_xticks([]); a.set_yticks([])
fig.suptitle(f"{ds} — what MAE / JEPA ingest: linear interp from K sensors  (FAE uses the points directly)", fontsize=12)
out = f"results/figs/{ds}/fig_interp_input.png"; os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout(rect=[0, 0, 1, 0.92]); fig.savefig(out, dpi=200); print(f"wrote {out}", flush=True)
