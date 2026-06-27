"""Visualize FAE latent forecasting: true field vs the operator's DIRECT prediction, + abs error,
across horizons t0..t+R, for a few NS validation clips. Channel 0 (smoke)."""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.train_forecast import load_fae, SIDE, DT_DIV
from src.data.well2d import make_coords_2d, fields_to_tokens
from src.data.ns import NSDataset
from src.latent_op import SetOperator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="results/checkpoints/ns/forecast/fae_forecast_s0.pt")
ap.add_argument("--clips", type=int, nargs="+", default=[5, 60, 150])
ap.add_argument("--ch", type=int, default=0)
ap.add_argument("--out", default="results/figs/fae_forecast.png")
args = ap.parse_args()

ck = torch.load(args.ckpt, map_location=DEVICE); a = ck["args"]
fae, C = load_fae(a.get("ae_ckpt") or a["fae_ckpt"])
op = SetOperator(fae.emb_dim, depth=a["depth"], heads=a["heads"]).to(DEVICE)
op.load_state_dict(ck["op"]); op.eval()
coords = make_coords_2d(n_side=SIDE, device=DEVICE)
idx = torch.arange(SIDE * SIDE, device=DEVICE)
R = a["rollout"]
tr = NSDataset("train", side=SIDE, mode="clip", clip_len=R + 1, frame_stride=a["frame_stride"], n_traj=a["n_traj"])
va = NSDataset("valid", side=SIDE, mode="clip", clip_len=R + 1, frame_stride=a["frame_stride"], n_traj=a["n_traj"], stats=tr.stats)


@torch.no_grad()
def enc(f): return fae.encode_tokens(fields_to_tokens(f, idx), coords[idx])
@torch.no_grad()
def dec(z): return fae.decoder(z, coords).permute(0, 2, 1).reshape(z.size(0), C, SIDE, SIDE)


def relL2(p, t): return np.linalg.norm(p - t) / (np.linalg.norm(t) + 1e-8)

ch = args.ch; clips = args.clips
fig, axs = plt.subplots(len(clips) * 3, R + 1, figsize=((R + 1) * 1.9, len(clips) * 3 * 1.9))
for r, ci in enumerate(clips):
    clip = va[ci][0].unsqueeze(0).to(DEVICE)                    # (1,C,T,H,W)
    f0 = clip[:, :, 0]; z0 = enc(f0); zr = z0
    trues = [clip[0, ch, k].cpu().numpy() for k in range(R + 1)]
    direct = [f0[0, ch].cpu().numpy()]; stepw = [f0[0, ch].cpu().numpy()]
    for k in range(R):
        direct.append(dec(op(z0, torch.full((1,), (k + 1) / DT_DIV, device=DEVICE)))[0, ch].cpu().numpy())
        zr = op(zr, torch.full((1,), 1.0 / DT_DIV, device=DEVICE)); stepw.append(dec(zr)[0, ch].cpu().numpy())
    # ---- DIAGNOSTIC: how much do TRUE / DIRECT / STEPWISE move away from the t+1 frame? ----
    dtrue = [relL2(trues[k], trues[1]) for k in range(1, R + 1)]
    ddir = [relL2(direct[k], direct[1]) for k in range(1, R + 1)]
    dstep = [relL2(stepw[k], stepw[1]) for k in range(1, R + 1)]
    print(f"clip {ci}: divergence-from-t+1  true={np.round(dtrue,3).tolist()}  "
          f"direct={np.round(ddir,3).tolist()}  stepwise={np.round(dstep,3).tolist()}", flush=True)
    vmax = float(np.abs(np.array(trues)).max()) or 1.0
    rows = [("true", trues), ("FAE direct", direct), ("FAE stepwise", stepw)]
    for k in range(R + 1):
        for rr, (lab, imgs) in enumerate(rows):
            ax = axs[r * 3 + rr, k]
            ax.imshow(imgs[k], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if k == 0:
                ax.set_ylabel(lab, fontsize=11)
        if r == 0:
            axs[0, k].set_title("t0" if k == 0 else f"t+{k}", fontsize=12)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
plt.tight_layout(); plt.savefig(args.out, dpi=130, bbox_inches="tight")
print(f"wrote {args.out}  (clips {clips}, channel {ch}, true/direct/stepwise t0..t+{R})", flush=True)
