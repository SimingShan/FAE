"""Full-trajectory gifs at NATIVE resolution, stride=1 (every frame) — ns / shear / sw.
To eyeball per-frame change (decorrelation) for principled Δt selection.
  python scripts/eval/viz_traj_gif.py
"""
import os, sys, glob, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

OUT = "results/figs"


def render(ds, frames, title, fps=12):
    os.makedirs(f"{OUT}/{ds}", exist_ok=True)
    frames = np.asarray(frames, dtype=np.float32)
    v = float(np.percentile(np.abs(frames - frames.mean()), 99)) or 1.0
    m = frames.mean()
    H, W = frames.shape[1:]; fw = 6.0; fh = max(2.2, fw * H / W)
    paths = []
    for t in range(len(frames)):
        fig, ax = plt.subplots(figsize=(fw, fh))
        ax.imshow(frames[t], cmap="coolwarm", vmin=m - v, vmax=m + v, aspect="equal")
        ax.axis("off"); ax.set_title(f"{title}   t={t}/{len(frames)-1}", fontsize=9)
        p = f"{OUT}/{ds}/_f{t:03d}.png"; fig.savefig(p, dpi=100, bbox_inches="tight"); plt.close(fig); paths.append(p)
    gif = f"{OUT}/{ds}/traj_stride1_native.gif"
    imageio.mimsave(gif, [imageio.imread(p) for p in paths], fps=fps, loop=0)
    for p in paths: os.remove(p)
    print(f"wrote {gif}  ({len(frames)} frames, native {H}x{W})", flush=True)


# --- NS: highest-buoyancy train trajectory (most vigorous plume), smoke channel ---
nsf = sorted(glob.glob(os.path.expanduser("~/scratch/ns_data/*_train_*.h5")),
             key=lambda f: float(re.search(r"_([0-9.]+)\.h5", f).group(1)))[-1]
buo = float(re.search(r"_([0-9.]+)\.h5", nsf).group(1))
with h5py.File(nsf, "r") as h:
    render("ns", h["train/u"][0], f"NS smoke (buoyancy={buo:.2f})")

# --- SHEAR: most turbulent (Re=5e5, Sc=10), tracer channel, native 256x512 ---
shf = os.path.expanduser("~/scratch/the_well_data/shear_flow/data/train/shear_flow_Reynolds_5e5_Schmidt_1e1.hdf5")
with h5py.File(shf, "r") as h:
    render("shear", h["t0_fields/tracer"][0], "shear tracer (Re=5e5, Sc=10)", fps=15)

# --- SW: a developed trajectory, vorticity, native 256x256 ---
d = np.load(os.path.expanduser("~/scratch/sw_data/shallow_water_full.npz"), mmap_mode="r")
render("sw", d["outputs"][125], "SW vorticity")
