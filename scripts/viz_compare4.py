"""Qualitative 4-way forecasting comparison on the SAME NS clips: true vs FAE / FNO / pixel-DeepONet /
L-DeepONet direct predictions, t0..t+R. Channel 0 (smoke). relL2 annotated under each prediction."""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.train_forecast import load_fae, load_grid, SIDE, DT_DIV
from src.latent_op import SetOperator
from src.deeponet import DeepONetOperator, PixelDeepONet
from src.fno import FNO2d
from src.data.well2d import make_coords_2d, fields_to_tokens
from src.data.ns import NSDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CK = "results/checkpoints/ns/forecast"
ap = argparse.ArgumentParser()
ap.add_argument("--clips", type=int, nargs="+", default=[5, 60])
ap.add_argument("--ch", type=int, default=0)
ap.add_argument("--out", default="results/figs/forecast_compare4.png")
args = ap.parse_args()
C = 3
coords = make_coords_2d(n_side=SIDE, device=DEVICE)
idx = torch.arange(SIDE * SIDE, device=DEVICE)


def grid(out):                                                # (B,N,C) -> (B,C,SIDE,SIDE)
    return out.permute(0, 2, 1).reshape(-1, C, SIDE, SIDE)

# ---- FAE ----
ckf = torch.load(f"{CK}/fae_forecast_s0.pt", map_location=DEVICE); af = ckf["args"]
fae, _ = load_fae(af.get("ae_ckpt") or af["fae_ckpt"])
faeop = SetOperator(fae.emb_dim, depth=af["depth"], heads=af["heads"]).to(DEVICE); faeop.load_state_dict(ckf["op"]); faeop.eval()
def pred_fae(f0, dt):
    z = fae.encode_tokens(fields_to_tokens(f0, idx), coords[idx])
    return fae.decoder(faeop(z, dt), coords).permute(0, 2, 1).reshape(-1, C, SIDE, SIDE)

# ---- FNO ----
ckn = torch.load(f"{CK}/fno_s0.pt", map_location=DEVICE); an = ckn["args"]
fno = FNO2d(C, C, width=an["width"], modes=an["modes"], n_layers=an["layers"]).to(DEVICE); fno.load_state_dict(ckn["model"]); fno.eval()
def pred_fno(f0, dt): return fno(f0, dt)

# ---- pixel-DeepONet ----
ckp = torch.load(f"{CK}/pixel_don_s0.pt", map_location=DEVICE); ap_ = ckp["args"]
pix = PixelDeepONet(C, SIDE, p=ap_["p"]).to(DEVICE)
with torch.no_grad(): pix(torch.zeros(1, C, SIDE, SIDE, device=DEVICE), coords, torch.ones(1, device=DEVICE))
pix.load_state_dict(ckp["model"]); pix.eval()
def pred_pix(f0, dt): return grid(pix(f0, coords, dt))

# ---- L-DeepONet ----
ckl = torch.load(f"{CK}/ldon_forecast_s0.pt", map_location=DEVICE); al = ckl["args"]
cae, _ = load_grid(al["ae_ckpt"])
ldonop = DeepONetOperator(cae.latent, p=al["p"]).to(DEVICE)
with torch.no_grad(): ldonop(torch.zeros(1, cae.latent, device=DEVICE), torch.ones(1, device=DEVICE))
ldonop.load_state_dict(ckl["op"]); ldonop.eval()
def pred_ldon(f0, dt): return cae.decode(ldonop(cae.encode(f0), dt))

methods = [("FAE", pred_fae), ("FNO", pred_fno), ("pixel-DON", pred_pix), ("L-DeepONet", pred_ldon)]
R = af["rollout"]; ch = args.ch
def relL2(p, t): return np.linalg.norm(p - t) / (np.linalg.norm(t) + 1e-8)

nrows = len(args.clips) * (len(methods) + 1)
fig, axs = plt.subplots(nrows, R + 1, figsize=((R + 1) * 1.8, nrows * 1.8))
tr = NSDataset("train", side=SIDE, mode="clip", clip_len=R + 1, frame_stride=af["frame_stride"], n_traj=af["n_traj"])
va = NSDataset("valid", side=SIDE, mode="clip", clip_len=R + 1, frame_stride=af["frame_stride"], n_traj=af["n_traj"], stats=tr.stats)

base = 0
for ci in args.clips:
    clip = va[ci][0].unsqueeze(0).to(DEVICE)
    f0 = clip[:, :, 0]
    trues = [clip[0, ch, k].cpu().numpy() for k in range(R + 1)]
    vmax = float(np.abs(np.array(trues)).max()) or 1.0
    rowdata = [("true", trues)]
    for name, fn in methods:
        with torch.no_grad():
            seq = [f0[0, ch].cpu().numpy()] + [fn(f0, torch.full((1,), (k + 1) / DT_DIV, device=DEVICE))[0, ch].cpu().numpy() for k in range(R)]
        rowdata.append((name, seq))
    for rr, (name, seq) in enumerate(rowdata):
        for k in range(R + 1):
            ax = axs[base + rr, k]
            ax.imshow(seq[k], cmap="RdBu_r", vmin=-vmax, vmax=vmax); ax.set_xticks([]); ax.set_yticks([])
            if k == 0: ax.set_ylabel(name, fontsize=10)
            if base == 0 and rr == 0: ax.set_title("t0" if k == 0 else f"t+{k}", fontsize=12)
            if name != "true" and k > 0:
                ax.set_xlabel(f"{relL2(seq[k], trues[k]):.2f}", fontsize=8)
    base += len(rowdata)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
plt.tight_layout(); plt.savefig(args.out, dpi=130, bbox_inches="tight")
print(f"wrote {args.out}", flush=True)
