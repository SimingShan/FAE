"""Visualize the Well shear_flow data we train on -> GIFs in results/plots/ for inspection.
Faithful to the pipeline: native 256x512 -> bilinear 224x224, the 4 channels [tracer,pressure,
vx,vy]. (Model additionally z-normalizes per channel — that only rescales the colorbar, not the
structure.) Δt=0.1, 200 frames/trajectory."""
import os, glob, h5py, numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import imageio.v2 as imageio
ROOT = os.environ["THE_WELL_DATA_DIR"] + "/shear_flow/data/train"
SIDE = 224; os.makedirs("results/plots", exist_ok=True)
NAMES = ["tracer", "pressure", "vel_x", "vel_y"]


def load_traj(f, traj=0):
    with h5py.File(f, "r") as h:
        tr = h["t0_fields/tracer"][traj]; pr = h["t0_fields/pressure"][traj]
        ve = h["t1_fields/velocity"][traj]
        Re = float(h["scalars/Reynolds"][()]); Sc = float(h["scalars/Schmidt"][()])
    fields = np.stack([tr, pr, ve[..., 0], ve[..., 1]], 1)              # (200,4,256,512)
    t = F.interpolate(torch.from_numpy(fields).float(), size=(SIDE, SIDE),
                      mode="bilinear", align_corners=False).numpy()      # (200,4,224,224)
    return t, Re, Sc


def fig_to_img(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def main():
    files = sorted(glob.glob(ROOT + "/*.hdf5"))
    print(f"{len(files)} (Re,Sc) files")

    # GIF 1: one trajectory, all 4 channels, full evolution
    fl, Re, Sc = load_traj(files[len(files) // 2], 0)
    frames = []
    for ti in range(0, 200, 2):
        fig, ax = plt.subplots(1, 4, figsize=(12, 3.3))
        for c in range(4):
            v = fl[:, c]; ax[c].imshow(fl[ti, c], cmap="RdBu_r", vmin=v.min(), vmax=v.max())
            ax[c].set_title(NAMES[c], fontsize=9); ax[c].axis("off")
        fig.suptitle(f"shear_flow Re={Re:.0f} Sc={Sc:.2f} | 224x224, frame {ti}/200 (Δt=0.1)", fontsize=10)
        fig.tight_layout(); frames.append(fig_to_img(fig)); plt.close(fig)
    imageio.mimsave("results/plots/well_traj_4ch.gif", frames, fps=12)
    print("saved results/plots/well_traj_4ch.gif", len(frames), "frames")

    # GIF 2: tracer diversity across 6 (Re,Sc)
    sel = files[::max(1, len(files) // 6)][:6]
    trajs = [load_traj(f, 0) for f in sel]
    frames2 = []
    for ti in range(0, 200, 3):
        fig, ax = plt.subplots(2, 3, figsize=(10.5, 6))
        for k, (fl, Re, Sc) in enumerate(trajs):
            a = ax.ravel()[k]; v = fl[:, 0]
            a.imshow(fl[ti, 0], cmap="RdBu_r", vmin=v.min(), vmax=v.max())
            a.set_title(f"Re={Re:.0f} Sc={Sc:.2f}", fontsize=9); a.axis("off")
        fig.suptitle(f"tracer across (Re,Sc) — frame {ti}/200", fontsize=11)
        fig.tight_layout(); frames2.append(fig_to_img(fig)); plt.close(fig)
    imageio.mimsave("results/plots/well_diversity.gif", frames2, fps=10)
    print("saved results/plots/well_diversity.gif", len(frames2), "frames")


if __name__ == "__main__":
    main()
