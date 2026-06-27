"""Shear trajectory diversity: drift vs frame for several (Re,Sc), a FULL-200-frame filmstrip comparing
dynamic vs static trajectories, and full-span GIFs. Shows shear dynamics are Re/Sc-dependent (not weak).
  python scripts/viz_shear_trajectories.py
"""
import os, sys, glob, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from src.plotstyle import apply, NPG
apply()

FILES = sorted(glob.glob("/gpfs/radev/scratch/lu_lu/ss5235/the_well_data/shear_flow/data/valid/*.hdf5"))
def find(Re, Sc):
    for f in FILES:
        with h5py.File(f) as h:
            if abs(float(h.attrs["Reynolds"]) - Re) < 1 and abs(float(h.attrs["Schmidt"]) - Sc) < 0.01:
                return f
    return None

def load(Re, Sc, seed=0):                                            # (200,128,128) standardized tracer
    with h5py.File(find(Re, Sc)) as h:
        tr = h["t0_fields/tracer"][seed].astype(np.float32)          # (200,H,W)
    tr = torch.nn.functional.interpolate(torch.from_numpy(tr)[:, None], size=(128, 128), mode="bilinear", align_corners=False)[:, 0].numpy()
    return (tr - tr.mean()) / (tr.std() + 1e-8)

SEL = [(100000, 10.0, "very dynamic"), (50000, 1.0, "dynamic"), (500000, 1.0, "moderate"), (10000, 0.1, "near-static")]

# 1) drift-vs-frame curves (full 200 frames)
fig, ax = plt.subplots(figsize=(6, 5))
for k, (Re, Sc, lab) in enumerate(SEL):
    s = load(Re, Sc); drift = [np.linalg.norm(s[i] - s[0]) / (np.linalg.norm(s[0]) + 1e-8) for i in range(len(s))]
    ax.plot(range(len(s)), drift, color=NPG[k], lw=2, label=f"Re={Re:.0e} Sc={Sc} ({lab})")
ax.set_xlabel("native frame (of 200)"); ax.set_ylabel("drift from t0 (relL2)"); ax.set_box_aspect(1)
ax.set_title("shear: dynamics are Re/Sc-dependent"); ax.legend(fontsize=10)
fig.tight_layout(); fig.savefig("results/figs/shear/drift_curves.png", dpi=120, bbox_inches="tight"); plt.close(fig)

# 2) FULL-200-frame filmstrip: rows = trajectories, cols = frames evenly spaced 0..199
idx = np.linspace(0, 199, 10).astype(int)
fig, axs = plt.subplots(len(SEL), len(idx), figsize=(1.7 * len(idx), 1.7 * len(SEL)))
for r, (Re, Sc, lab) in enumerate(SEL):
    s = load(Re, Sc); vmax = float(np.percentile(np.abs(s), 99))
    for c, i in enumerate(idx):
        axs[r, c].imshow(s[i], cmap="coolwarm", vmin=-vmax, vmax=vmax); axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
        if r == 0: axs[r, c].set_title(f"f{i}", fontsize=10)
    axs[r, 0].set_ylabel(f"Re={Re:.0e}\nSc={Sc}", fontsize=10)
fig.suptitle("shear FULL trajectory (0-199), 4 (Re,Sc) — dynamic (top) to near-static (bottom)", fontsize=13)
fig.tight_layout(); fig.savefig("results/figs/shear/trajectories_compare.png", dpi=120, bbox_inches="tight"); plt.close(fig)

# 3) full-span GIFs (40 frames over 0..199) for the dynamic + static
for (Re, Sc, lab), name in [(SEL[0], "dynamic"), (SEL[-1], "static")]:
    s = load(Re, Sc); fr = np.linspace(0, 199, 40).astype(int); vmax = float(np.percentile(np.abs(s), 99))
    fig, ax = plt.subplots(figsize=(4, 4)); ax.set_xticks([]); ax.set_yticks([])
    im = ax.imshow(s[fr[0]], cmap="coolwarm", vmin=-vmax, vmax=vmax); tt = ax.set_title("", fontsize=11)
    def upd(j, s=s, fr=fr, im=im, tt=tt, Re=Re, Sc=Sc): im.set_data(s[fr[j]]); tt.set_text(f"Re={Re:.0e} Sc={Sc}  native f{fr[j]}"); return im, tt
    FuncAnimation(fig, upd, frames=len(fr), blit=False).save(f"results/figs/shear/temporal_{name}.gif", writer=PillowWriter(fps=6))
    plt.close(fig)
    print(f"wrote temporal_{name}.gif (Re={Re:.0e} Sc={Sc}, full 0-199)", flush=True)
print("wrote drift_curves.png + trajectories_compare.png", flush=True)
