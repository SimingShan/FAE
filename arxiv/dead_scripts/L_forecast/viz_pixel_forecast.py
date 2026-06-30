"""Visualize pixel-space DeepONet forecasting: true vs DIRECT vs STEPWISE field, + divergence check.
Mirrors viz_forecast.py (the FAE viz) so the two are directly comparable. Channel 0 (smoke)."""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.deeponet import PixelDeepONet
from src.data.ns import NSDataset
from src.data.well2d import make_coords_2d

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIDE, DT_DIV = 64, 8.0
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="results/checkpoints/ns/forecast/pixel_don_s0.pt")
ap.add_argument("--clips", type=int, nargs="+", default=[5, 60, 150])
ap.add_argument("--ch", type=int, default=0)
ap.add_argument("--out", default="results/figs/pixel_don_forecast.png")
args = ap.parse_args()

ck = torch.load(args.ckpt, map_location=DEVICE); a = ck["args"]
C = 3
coords = make_coords_2d(n_side=SIDE, device=DEVICE)
model = PixelDeepONet(in_ch=C, side=SIDE, p=a["p"]).to(DEVICE)
with torch.no_grad():
    model(torch.zeros(1, C, SIDE, SIDE, device=DEVICE), coords, torch.ones(1, device=DEVICE))   # init lazy
model.load_state_dict(ck["model"]); model.eval()
R = a["rollout"]
tr = NSDataset("train", side=SIDE, mode="clip", clip_len=R + 1, frame_stride=a["frame_stride"], n_traj=a["n_traj"])
va = NSDataset("valid", side=SIDE, mode="clip", clip_len=R + 1, frame_stride=a["frame_stride"], n_traj=a["n_traj"], stats=tr.stats)


@torch.no_grad()
def field(f, dt):                                              # f (1,C,H,W), dt scalar -> (1,C,H,W)
    out = model(f, coords, torch.full((1,), dt, device=DEVICE))
    return out.permute(0, 2, 1).reshape(1, C, SIDE, SIDE)


def relL2(p, t): return np.linalg.norm(p - t) / (np.linalg.norm(t) + 1e-8)

ch, clips = args.ch, args.clips
fig, axs = plt.subplots(len(clips) * 3, R + 1, figsize=((R + 1) * 1.9, len(clips) * 3 * 1.9))
for r, ci in enumerate(clips):
    clip = va[ci][0].unsqueeze(0).to(DEVICE)
    f0 = clip[:, :, 0]; fr = f0
    trues = [clip[0, ch, k].cpu().numpy() for k in range(R + 1)]
    direct = [f0[0, ch].cpu().numpy()]; stepw = [f0[0, ch].cpu().numpy()]
    for k in range(R):
        direct.append(field(f0, (k + 1) / DT_DIV)[0, ch].cpu().numpy())
        fr = field(fr, 1.0 / DT_DIV); stepw.append(fr[0, ch].cpu().numpy())
    ddir = [relL2(direct[k], direct[1]) for k in range(1, R + 1)]
    print(f"clip {ci}: divergence-from-t+1  true={np.round([relL2(trues[k],trues[1]) for k in range(1,R+1)],3).tolist()}  "
          f"direct={np.round(ddir,3).tolist()}", flush=True)
    vmax = float(np.abs(np.array(trues)).max()) or 1.0
    for k in range(R + 1):
        for rr, (lab, imgs) in enumerate([("true", trues), ("pixel-DON direct", direct), ("pixel-DON stepwise", stepw)]):
            ax = axs[r * 3 + rr, k]
            ax.imshow(imgs[k], cmap="RdBu_r", vmin=-vmax, vmax=vmax); ax.set_xticks([]); ax.set_yticks([])
            if k == 0:
                ax.set_ylabel(lab, fontsize=10)
        if r == 0:
            axs[0, k].set_title("t0" if k == 0 else f"t+{k}", fontsize=12)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
plt.tight_layout(); plt.savefig(args.out, dpi=130, bbox_inches="tight")
print(f"wrote {args.out}", flush=True)
