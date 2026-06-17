"""Visualize FlowBench FPO (flow past object) -> GIF. Shows the unsteady wake past a
complex geometry evolving in time, with the object overlaid. 3 channels (u,v,p),
242 frames, 256x1024. + a vorticity view (the wake/shedding)."""
import os, glob, re, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import imageio.v2 as imageio
FB = "/gpfs/radev/home/ss5235/scratch/flowbench/FPO_NS_2D_1024x256/harmonics"
os.makedirs("results/plots", exist_ok=True)


def main():
    # pick a high-Re case (more vortex shedding = more interesting)
    files = sorted(glob.glob(FB + "/*/Re_*.npz"), key=lambda f: -int(re.search(r'Re_(\d+)', f).group(1)))
    f = files[0]; case = os.path.dirname(f); Re = int(re.search(r'Re_(\d+)', f).group(1))
    d = np.load(f)['data'].astype(np.float32)          # (242,256,1024,3)
    g = np.load(case + "/input_geometry.npz")
    mask = g['mask'].astype(bool)                       # (256,1024) True = FLUID (object is ~mask, ~1%)
    print(f"case {os.path.basename(case)} Re={Re}  data {d.shape}")
    ds = 2                                              # spatial downsample for gif size
    SOLID = ~mask[::ds, ::ds]                           # True = object (to hide)
    names = ["u (vel-x)", "v (vel-y)", "pressure"]

    frames = []
    for ti in range(0, 242, 3):
        fr = d[ti, ::ds, ::ds]                          # (128,512,3)
        # vorticity from u,v
        u, v = fr[..., 0], fr[..., 1]
        vort = np.gradient(v, axis=1) - np.gradient(u, axis=0)
        fig, ax = plt.subplots(4, 1, figsize=(11, 7))
        panels = [fr[..., 0], fr[..., 1], fr[..., 2], vort]
        titles = names + ["vorticity (wake)"]
        for k, (a, raw, t) in enumerate(zip(ax, panels, titles)):
            vmax = np.percentile(np.abs(raw[~SOLID]), 98) + 1e-6
            a.imshow(np.ma.masked_where(SOLID, raw), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            a.contour(SOLID, levels=[0.5], colors="k", linewidths=0.8)   # object outline
            a.set_title(t, fontsize=9); a.axis("off")
        fig.suptitle(f"FlowBench FPO — flow past object, Re={Re}, frame {ti}/242", fontsize=11)
        fig.tight_layout()
        fig.canvas.draw(); frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave("results/plots/flowbench_fpo.gif", frames, fps=12)
    print("saved results/plots/flowbench_fpo.gif", len(frames), "frames")

    # diversity: vorticity of 6 different geometries (frame 120)
    div = sorted(glob.glob(FB + "/*/Re_*.npz"))[::max(1, len(files) // 6)][:6]
    fig, ax = plt.subplots(6, 1, figsize=(11, 9))
    for a, ff in zip(ax, div):
        dd = np.load(ff)['data'][120, ::ds, ::ds].astype(np.float32)
        solid = ~np.load(os.path.dirname(ff) + "/input_geometry.npz")['mask'][::ds, ::ds].astype(bool)
        u, v = dd[..., 0], dd[..., 1]; vort = np.gradient(v, axis=1) - np.gradient(u, axis=0)
        vmax = np.percentile(np.abs(vort[~solid]), 98) + 1e-6
        a.imshow(np.ma.masked_where(solid, vort), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        a.contour(solid, levels=[0.5], colors="k", linewidths=0.8)
        rev = int(re.search(r'Re_(\d+)', ff).group(1)); shp = os.path.basename(os.path.dirname(ff))
        a.set_title(f"Re={rev}  case {shp}", fontsize=8)
        a.axis("off")
    fig.suptitle("FPO geometry diversity — vorticity past 6 different shapes (frame 120)", fontsize=11)
    fig.tight_layout(); fig.savefig("results/plots/flowbench_diversity.png", dpi=100)
    print("saved results/plots/flowbench_diversity.png")


if __name__ == "__main__":
    main()
