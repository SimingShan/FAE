"""Unified reconstruction comparison (like results/figures/reconstruction.png) for shear / typhoon.
Rows FAE / MAE / JEPA; cols: input | what-it-sees | reconstruction | |error|+relL2.
  FAE  : K scattered sensors -> coordinate decode (native sparse->dense).
  MAE  : 75% masked -> decoder (paste visible+predicted).
  JEPA : NO pixel decoder -> fit a shared per-patch LINEAR readout (RAE-lite) to visualize recoverable content.
  python scripts/viz_recon_compare.py --dataset typhoon
"""
import os, sys, glob, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.plotstyle import apply, COLORS
apply()
from src.models.fae import FAE
from src.data.well2d import make_coords_2d, fields_to_tokens
from scripts.train_baseline import build_model
from scripts.probe_all import get_data, _frame0
from torch.utils.data import DataLoader

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["shear", "typhoon"], required=True)
ap.add_argument("--sensors", type=int, default=1024)
ap.add_argument("--ch", type=int, default=0)
args = ap.parse_args()
SIDE = {"shear": 128, "typhoon": 128}[args.dataset]; NPIX = SIDE * SIDE; CH = args.ch
ckp = lambda m: sorted(glob.glob(f"results/checkpoints/{args.dataset}/{m}/*_s*.pt"))[0]

ds = get_data(args.dataset, "valid", SIDE)
img = _frame0(ds[len(ds) // 3][0].unsqueeze(0)).to(DEV)            # one held-out field (1,C,H,W)
C = img.shape[1]; vmax = float(img[0, CH].abs().max()) or 1.0; vmin = -vmax

# ---- FAE: native sparse -> dense ----
af = torch.load(ckp("fae"), map_location=DEV)["train_args"]
fae = FAE(emb_dim=af["emb_dim"], num_iter=af.get("num_iter", 4), depth_per_iter=af.get("depth_per_iter", 4),
          num_latents=af["num_latents"], n_freq=32, max_freq=32, coord_dim=2, in_chans=C).to(DEV)
fae.load_state_dict(torch.load(ckp("fae"), map_location=DEV)["model"]); fae.eval()
coords = make_coords_2d(n_side=SIDE, device=DEV)
g = torch.Generator(device=DEV).manual_seed(0); sidx = torch.randperm(NPIX, generator=g, device=DEV)[:args.sensors]
with torch.no_grad():
    pred, _ = fae(fields_to_tokens(img, sidx), coords[sidx], coords)       # (1, NPIX, C)
fae_rec = pred.reshape(1, SIDE, SIDE, C).permute(0, 3, 1, 2)[0, CH].cpu()
fae_see = torch.full((NPIX,), float("nan")); fae_see[sidx.cpu()] = img[0, CH].reshape(-1)[sidx.cpu()].cpu()
fae_see = fae_see.reshape(SIDE, SIDE)

# ---- MAE: native masked recon ----
am = torch.load(ckp("mae"), map_location=DEV)["train_args"]
mae = build_model("mae", resolution=am["resolution"], in_chans=C, embed_dim=am.get("embed_dim"),
                  depth=am.get("depth"), patch_size=am.get("patch_size")).to(DEV)
mae.load_state_dict(torch.load(ckp("mae"), map_location=DEV)["model"]); mae.eval()
with torch.no_grad():
    _, pdt, mask = mae(img, 0.75); op = mae.patchify(img); mk = mask.unsqueeze(-1)
    mae_rec = mae.unpatchify(op * (1 - mk) + pdt * mk)[0, CH].cpu()
    mae_see = mae.unpatchify(op * (1 - mk))[0, CH].cpu()
    mvis = mae.unpatchify(mk.expand(-1, -1, op.shape[-1]).float())[0, CH].cpu()
mae_see[mvis > 0] = float("nan")

# ---- JEPA: no decoder -> fit shared per-patch linear readout (RAE-lite) ----
aj = torch.load(ckp("jepa"), map_location=DEV)["train_args"]; P = aj.get("patch_size"); G = SIDE // P
jepa = build_model("ijepa", resolution=aj["resolution"], in_chans=C, embed_dim=aj.get("embed_dim"),
                   depth=aj.get("depth"), patch_size=P).to(DEV)
jepa.load_state_dict(torch.load(ckp("jepa"), map_location=DEV)["model"]); jepa.eval()
def patchify(im):                                                  # (B,C,H,W)->(B,G*G,P*P*C)
    B = im.size(0); x = im.reshape(B, C, G, P, G, P)
    return x.permute(0, 2, 4, 3, 5, 1).reshape(B, G * G, P * P * C)
def unpatchify(t):                                                 # (B,G*G,P*P*C)->(B,C,H,W)
    B = t.size(0); x = t.reshape(B, G, G, P, P, C)
    return x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, SIDE, SIDE)
with torch.no_grad():
    F_, T_ = [], []
    for x, _ in DataLoader(ds, batch_size=32):
        xf = _frame0(x).to(DEV)
        F_.append(jepa.target(xf).reshape(-1, jepa.target(xf).shape[-1])); T_.append(patchify(xf).reshape(-1, P * P * C))
        if sum(t.shape[0] for t in F_) > 4000: break
    Fm = torch.cat(F_); Tm = torch.cat(T_)
    W = torch.linalg.lstsq(Fm, Tm).solution                        # (D, P*P*C) shared readout
    jrec = unpatchify((jepa.target(img) @ W))[0, CH].cpu()

rows = [("FAE", fae_see, fae_rec, f"{args.sensors} sensors"),
        ("MAE", mae_see, mae_rec, "75% masked"),
        ("JEPA", None, jrec, "full img (latent)")]
fig, ax = plt.subplots(3, 4, figsize=(13, 9.6))
inp = img[0, CH].cpu()
for r, (name, see, rec, seetxt) in enumerate(rows):
    ax[r, 0].imshow(inp, vmin=vmin, vmax=vmax, cmap="RdBu_r")
    ax[r, 0].set_ylabel(name, fontsize=16, fontweight="bold", color=COLORS[name])
    if see is not None:
        ax[r, 1].imshow(see, vmin=vmin, vmax=vmax, cmap="RdBu_r")
    else:
        ax[r, 1].imshow(inp, vmin=vmin, vmax=vmax, cmap="RdBu_r")
        ax[r, 1].text(.5, .5, "no decoder\n(latent only)", ha="center", va="center", transform=ax[r, 1].transAxes,
                      fontsize=13, color="white", fontweight="bold")
    ax[r, 1].set_title(f"sees: {seetxt}" if r == 0 else seetxt, fontsize=13)
    ax[r, 2].imshow(rec, vmin=vmin, vmax=vmax, cmap="RdBu_r")
    err = (rec - inp).abs(); relL2 = float(err.norm() / (inp.norm() + 1e-8))
    print(f"{name:5s} recon relL2 = {relL2:.3f}", flush=True)
    ax[r, 3].imshow(err, vmin=0, vmax=(vmax - vmin) * 0.5, cmap="magma")
    ax[r, 3].set_title((("abs error\n" if r == 0 else "")) + f"relL2={relL2:.3f}", fontsize=13)
    for c in range(4):
        ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
ax[0, 0].set_title("input", fontsize=14); ax[0, 2].set_title("reconstruction", fontsize=14)
ax[2, 2].set_xlabel("linear readout (RAE-lite)", fontsize=11)
fig.suptitle(f"{args.dataset} held-out sample — native recon (FAE, MAE) vs latent-only (JEPA needs a decoder)", fontsize=15)
out = f"results/figs/{args.dataset}/reconstruction.png"; os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(out, dpi=150)
print(f"wrote {out}", flush=True)
