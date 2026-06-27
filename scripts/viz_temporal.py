"""Temporal-cadence check: for each dataset, one trajectory's frames AT THE TRAINING STRIDE ->
animated GIF + an inline filmstrip PNG. Consecutive frames = the gap the FAE predictor learns across,
so you can eyeball whether dt (frame_stride x dt_max) shows meaningful-but-learnable change.
  python scripts/viz_temporal.py --dataset shear
"""
import os, sys, glob, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

CMAP = {"shear": "coolwarm", "ns": "coolwarm", "typhoon": "Greys_r"}   # typhoon = grayscale IR (remote-sensing convention)

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["shear", "typhoon", "ns"], required=True)
ap.add_argument("--stride", type=int, default=4, help="training frame_stride (typhoon=1)")
ap.add_argument("--nframes", type=int, default=16)
args = ap.parse_args()
ds = args.dataset


def load_seq():
    """Return (T, H, W) channel-0 sequence at the training stride + a label."""
    if ds == "shear":
        import h5py
        f = sorted(glob.glob("/gpfs/radev/scratch/lu_lu/ss5235/the_well_data/shear_flow/data/valid/*.hdf5"))[0]
        with h5py.File(f) as h:
            tr = h["t0_fields/tracer"][0]                              # (nf, H, W) trajectory 0
            Re = float(h.attrs["Reynolds"])
        seq = tr[::args.stride][:args.nframes]
        return seq, f"shear tracer | stride={args.stride} of {tr.shape[0]} frames | Re={Re:.0f}"
    if ds == "typhoon":
        # pick a storm with many frames
        cs = sorted(glob.glob(os.path.expanduser("~/scratch/typhoon_cache/*.npz")))
        for f in cs:
            d = np.load(f)
            if d["x"].shape[0] >= args.nframes:
                x = d["x"][:, 0] if d["x"].ndim == 4 else d["x"]       # (T,H,W)
                return x[::args.stride][:args.nframes], f"typhoon IR | stride={args.stride} (hourly) | {os.path.basename(f)} {d['x'].shape[0]} frames"
    # ns
    import h5py
    f = sorted(glob.glob(os.path.expanduser("~/scratch/ns_data/*train*.h*5")) + glob.glob(os.path.expanduser("~/scratch/ns_data/*.h*5")))[0]
    with h5py.File(f) as h:
        grp = h[list(h.keys())[0]]
        u = grp["u"][0]                                                # (T,H,W) smoke, traj 0
        t = grp["t"][0] if "t" in grp else None
    gap = f"; physical Δt≈{(t[args.stride]-t[0]):.2f}" if t is not None else ""
    return u[::args.stride][:args.nframes], f"ns smoke | stride={args.stride} of {u.shape[0]}{gap}"


seq, title = load_seq()
seq = np.asarray(seq, dtype=np.float32)
seq = torch.nn.functional.interpolate(torch.from_numpy(seq)[:, None], size=(128, 128),
                                      mode="bilinear", align_corners=False)[:, 0].numpy()   # LOADED resolution = 128
cmap = CMAP[ds]
if cmap == "Greys_r":                                            # sequential IR -> data range (1-99 pct)
    vlo, vhi = float(np.percentile(seq, 1)), float(np.percentile(seq, 99))
else:                                                            # cool-warm: standardize (as training does) so non-negative
    seq = (seq - seq.mean()) / (seq.std() + 1e-8)               # fields (e.g. NS smoke) become zero-centered -> blue & red
    vhi = float(np.percentile(np.abs(seq), 99)) or 1.0; vlo = -vhi
os.makedirs(f"results/figs/{ds}", exist_ok=True)

# 1) animated GIF
fig, ax = plt.subplots(figsize=(4, 4)); ax.set_xticks([]); ax.set_yticks([])
im = ax.imshow(seq[0], cmap=cmap, vmin=vlo, vmax=vhi); tt = ax.set_title(f"{ds}  t=0", fontsize=11)
def upd(i):
    im.set_data(seq[i]); tt.set_text(f"{ds}  frame {i} (= {i*args.stride} native)"); return im, tt
FuncAnimation(fig, upd, frames=len(seq), blit=False).save(f"results/figs/{ds}/temporal.gif", writer=PillowWriter(fps=3))
plt.close(fig)

# 2) inline filmstrip (8 evenly-spaced frames) + per-step change
k = min(8, len(seq)); idx = np.linspace(0, len(seq) - 1, k).astype(int)
fig, axs = plt.subplots(1, k, figsize=(2 * k, 2.4))
for j, i in enumerate(idx):
    axs[j].imshow(seq[i], cmap=cmap, vmin=vlo, vmax=vhi); axs[j].set_xticks([]); axs[j].set_yticks([])
    axs[j].set_title(f"f{i}", fontsize=10)
# mean relative change between consecutive TRAINING frames (the prediction gap)
dchg = [np.linalg.norm(seq[i + 1] - seq[i]) / (np.linalg.norm(seq[i]) + 1e-8) for i in range(len(seq) - 1)]
fig.suptitle(f"{title}   |   mean per-step relΔ = {np.mean(dchg):.3f}", fontsize=11)
fig.tight_layout(); fig.savefig(f"results/figs/{ds}/temporal_strip.png", dpi=120, bbox_inches="tight"); plt.close(fig)
print(f"wrote results/figs/{ds}/temporal.gif + temporal_strip.png  | per-step relΔ mean={np.mean(dchg):.3f} (min {np.min(dchg):.3f} max {np.max(dchg):.3f})", flush=True)
