"""Spherical shallow-water vorticity on a 3D rotating GLOBE -> GIF + a 4-view filmstrip.
Maps (phi, theta) -> unit-sphere (x,y,z); paints vorticity as facecolors; rotates while time advances.
  python scripts/viz_sw_sphere.py [sample]
"""
import os, argparse, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter
ap = argparse.ArgumentParser(); ap.add_argument("traj", type=int, nargs="?", default=0); s = ap.parse_args().traj
d = np.load(os.path.expanduser("~/scratch/sw_data/shallow_water.npz"))
v = d["outputs"][s].astype(np.float32)[:, ::3, ::3]            # (T, NP, NT) downsample 256->86 for speed
a, b = d["params"][s]; T, NP, NT = v.shape
phi = np.linspace(0, 2*np.pi, NP); theta = np.linspace(0.02, np.pi-0.02, NT)
PHI, TH = np.meshgrid(phi, theta, indexing="ij")
X = np.sin(TH)*np.cos(PHI); Y = np.sin(TH)*np.sin(PHI); Z = np.cos(TH)
vmax = float(np.percentile(np.abs(v), 99)); norm = plt.Normalize(-vmax, vmax)
os.makedirs("results/figs/sw", exist_ok=True)

# rotating globe GIF (time advances + view rotates)
fig = plt.figure(figsize=(5, 5)); ax = fig.add_subplot(111, projection="3d")
def upd(i):
    ax.clear(); ax.set_axis_off(); ax.set_box_aspect((1, 1, 1))
    ax.plot_surface(X, Y, Z, facecolors=cm.RdBu_r(norm(v[i])), rstride=1, cstride=1, antialiased=False, shade=False)
    ax.view_init(elev=22, azim=i*4)
    ax.set_title(f"SW vorticity on sphere  (sample {s}, hour {i*5})", fontsize=10)
FuncAnimation(fig, upd, frames=T, blit=False).save(f"results/figs/sw/sw_sphere_traj{s}.gif", writer=PillowWriter(fps=8))
plt.close(fig)

# static 4-view filmstrip (different times) for inline viewing
fig = plt.figure(figsize=(16, 4)); idx = np.linspace(0, T-1, 4).astype(int)
for j, i in enumerate(idx):
    ax = fig.add_subplot(1, 4, j+1, projection="3d"); ax.set_axis_off(); ax.set_box_aspect((1, 1, 1))
    ax.plot_surface(X, Y, Z, facecolors=cm.RdBu_r(norm(v[i])), rstride=1, cstride=1, antialiased=False, shade=False)
    ax.view_init(elev=22, azim=40); ax.set_title(f"hour {i*5}", fontsize=11)
fig.suptitle(f"SW vorticity on the sphere — sample {s} (alpha={a:.2f} beta={b:.2f})", fontsize=13)
fig.tight_layout(); fig.savefig(f"results/figs/sw/sw_sphere_traj{s}_strip.png", dpi=100, bbox_inches="tight")
print(f"wrote sw_sphere_traj{s}.gif ({T} frames) + sw_sphere_traj{s}_strip.png", flush=True)
