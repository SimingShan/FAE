"""Animate one spherical shallow-water trajectory (vorticity) -> GIF. Equirectangular (phi x theta)."""
import os, argparse, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
ap = argparse.ArgumentParser(); ap.add_argument("traj", type=int, nargs="?", default=0); s = ap.parse_args().traj
d = np.load(os.path.expanduser("~/scratch/sw_data/shallow_water.npz"))
vort = d["outputs"][s].astype(np.float32)                 # (T, 256, 256)
a, b = d["params"][s]
vort = vort[:, ::2, ::2]                                   # 256->128 for a lighter gif
T = vort.shape[0]; vmax = float(np.percentile(np.abs(vort), 99))
fig, ax = plt.subplots(figsize=(6, 3)); ax.set_xticks([]); ax.set_yticks([])
im = ax.imshow(vort[0].T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower", aspect="auto")
tt = ax.set_title("", fontsize=11)
def upd(i): im.set_data(vort[i].T); tt.set_text(f"SW vorticity (sample {s}, α={a:.2f} β={b:.2f})  hour {i*5}"); return im, tt
os.makedirs("results/figs/sw", exist_ok=True)
FuncAnimation(fig, upd, frames=T, blit=False).save(f"results/figs/sw/sw_traj{s}.gif", writer=PillowWriter(fps=8))
print(f"wrote results/figs/sw/sw_traj{s}.gif  ({T} frames, sample {s}, vort range +-{vmax:.1f})", flush=True)
