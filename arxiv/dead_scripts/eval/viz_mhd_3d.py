"""True 3D volumetric isosurface gif of an MHD trajectory (NOT slices) — to see the turbulent dynamics.
Marching-cubes isosurface of a chosen field, rotating, over time. Headless (Agg + imageio).
  python scripts/eval/viz_mhd_3d.py --file MHD_Ma_2_Ms_7 --field density --traj 0
"""
import os, sys, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import imageio.v2 as imageio

ROOT = os.path.expanduser("~/scratch/the_well/datasets/MHD_64/data")
KEY = {"density": "t0_fields/density", "velocity": "t1_fields/velocity", "magnetic": "t1_fields/magnetic_field"}
ap = argparse.ArgumentParser()
ap.add_argument("--file", default="MHD_Ma_2_Ms_7")
ap.add_argument("--field", default="density", choices=list(KEY))
ap.add_argument("--traj", type=int, default=0)
ap.add_argument("--nframes", type=int, default=24)
ap.add_argument("--pct", type=float, default=85, help="isosurface level = this percentile of each volume")
args = ap.parse_args()

fp = glob.glob(f"{ROOT}/*/{args.file}.hdf5")[0]
with h5py.File(fp, "r") as h:
    T = h[KEY[args.field]].shape[1]
    ts = np.linspace(0, T - 1, args.nframes).astype(int)
    vols = []
    for t in ts:
        a = np.asarray(h[KEY[args.field]][args.traj, t], dtype=np.float32)
        if a.ndim == 4: a = np.sqrt((a ** 2).sum(-1))             # vector field -> magnitude
        vols.append(a)
print(f"{args.file} {args.field} traj {args.traj}: {len(vols)} frames, vol {vols[0].shape}", flush=True)

outdir = "results/figs/mhd"; os.makedirs(outdir, exist_ok=True)
frames = []
for k, (t, vol) in enumerate(zip(ts, vols)):
    lvl = float(np.percentile(vol, args.pct))
    try:
        verts, faces, _, _ = measure.marching_cubes(vol, level=lvl)
    except (ValueError, RuntimeError):
        continue
    fig = plt.figure(figsize=(5, 5)); ax = fig.add_subplot(111, projection="3d")
    tri = verts[faces]
    zc = tri[:, :, 2].mean(1)                                     # color by depth for 3D legibility
    mesh = Poly3DCollection(tri, alpha=0.55, linewidths=0)
    mesh.set_facecolor(cm.plasma((zc - zc.min()) / (np.ptp(zc) + 1e-9)))
    ax.add_collection3d(mesh)
    n = vol.shape[0]; ax.set_xlim(0, n); ax.set_ylim(0, n); ax.set_zlim(0, n)
    ax.set_axis_off(); ax.view_init(elev=22, azim=k * 4)         # slow rotation
    ax.set_title(f"{args.file}\n{args.field} iso(p{int(args.pct)})  t={t}/{T-1}", fontsize=10)
    f = f"{outdir}/_frame_{k:03d}.png"; fig.savefig(f, dpi=110, bbox_inches="tight"); plt.close(fig); frames.append(f)

gif = f"{outdir}/mhd_3d_{args.field}_{args.file}.gif"
imageio.mimsave(gif, [imageio.imread(f) for f in frames], fps=8, loop=0)
for f in frames: os.remove(f)
print(f"wrote {gif}  ({len(frames)} frames)", flush=True)
