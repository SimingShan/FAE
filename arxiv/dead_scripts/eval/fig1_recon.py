"""Paper fig 1 — reconstruction: FAE (sparse sensors -> coordinate decode) vs MAE (75% masked -> decode).
Rows FAE/MAE; cols: input | what-it-sees | reconstruction | |error|+relL2.  plotstyle, per-dataset cmap.
  python scripts/figs/fig1_recon.py --dataset typhoon
"""
import os, sys, glob, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.plotstyle import apply, COLORS
apply()
from src.models.fae import FAE
from src.data.well2d import make_coords_2d_hw, fields_to_tokens
from scripts.train.train_baseline import build_model
from scripts.eval.probe_all import get_data, _frame0, fae_hw

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CMAP = {"shear": "coolwarm", "ns": "coolwarm", "typhoon": "Greys_r", "sw": "coolwarm"}
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["shear", "typhoon", "ns", "sw"], required=True)
ap.add_argument("--sensors", type=int, default=256)
ap.add_argument("--ch", type=int, default=0)
ap.add_argument("--idx", type=int, default=None, help="dataset frame index (default len//3); pick a developed frame")
args = ap.parse_args()
SIDE = 128; CH = args.ch; cmap = CMAP[args.dataset]
ckp = lambda m: sorted(glob.glob(f"results/checkpoints/{args.dataset}/{m}/*_s*.pt"))[0]


def norm_vis(x):                                                  # standardize for cool-warm; raw for IR
    return (x - x.mean()) / (x.std() + 1e-8) if cmap != "Greys_r" else x


# ---- FAE (its own, possibly rect, resolution) ----
fck = ckp("fae"); H, W = fae_hw(fck, SIDE)
df = get_data(args.dataset, "valid", list((H, W)) if (H, W) != (SIDE, SIDE) else SIDE)
FIDX = args.idx if args.idx is not None else len(df) // 3
img_f = _frame0(df[FIDX][0].unsqueeze(0)).to(DEV); C = img_f.shape[1]
a = torch.load(fck, map_location=DEV)["train_args"]
fae = FAE(emb_dim=a["emb_dim"], num_iter=a.get("num_iter", 4), depth_per_iter=a.get("depth_per_iter", 5),
          num_latents=a["num_latents"], num_cross_heads=a.get("num_cross_heads", 4), num_self_heads=a.get("num_self_heads", 8),
          n_freq=a.get("n_freq", 32), max_freq=a.get("max_freq", 32), coord_dim=2, in_chans=C).to(DEV)
fae.load_state_dict(torch.load(fck, map_location=DEV)["model"]); fae.eval()
coords = make_coords_2d_hw(H, W, device=DEV); g = torch.Generator(device=DEV).manual_seed(0)
sidx = torch.randperm(H * W, generator=g, device=DEV)[:args.sensors]
with torch.no_grad():
    pred, _ = fae(fields_to_tokens(img_f, sidx), coords[sidx], coords)
fae_rec = pred.reshape(1, H, W, C).permute(0, 3, 1, 2)[0, CH].cpu().numpy()
fae_in = img_f[0, CH].cpu().numpy()
fae_see = np.full(H * W, np.nan); fae_see[sidx.cpu()] = fae_in.reshape(-1)[sidx.cpu()]; fae_see = fae_see.reshape(H, W)

# ---- MAE (native res: shear 128x256, else square) ----
am = torch.load(ckp("mae"), map_location=DEV)["train_args"]
mres = (am["res_h"], am["res_w"]) if am.get("res_h") else am["resolution"]
mH, mW = (am["res_h"], am["res_w"]) if am.get("res_h") else (SIDE, SIDE)
dm = get_data(args.dataset, "valid", list((mH, mW)) if (mH, mW) != (SIDE, SIDE) else SIDE)
img_m = _frame0(dm[min(FIDX, len(dm) - 1)][0].unsqueeze(0)).to(DEV)
mae = build_model("mae", resolution=mres, in_chans=C, embed_dim=am.get("embed_dim"),
                  depth=am.get("depth"), patch_size=am.get("patch_size")).to(DEV)
mae.load_state_dict(torch.load(ckp("mae"), map_location=DEV)["model"]); mae.eval()
with torch.no_grad():
    _, pdt, mask = mae(img_m, 0.75); op = mae.patchify(img_m); mk = mask.unsqueeze(-1)
    mae_rec = mae.unpatchify(op * (1 - mk) + pdt * mk)[0, CH].cpu().numpy()
    mae_see = mae.unpatchify(op * (1 - mk))[0, CH].cpu().numpy()
    mvis = mae.unpatchify(mk.expand(-1, -1, op.shape[-1]).float())[0, CH].cpu().numpy()
mae_in = img_m[0, CH].cpu().numpy(); mae_see[mvis > 0] = np.nan


def relL2(p, t): return float(np.linalg.norm(p - t) / (np.linalg.norm(t) + 1e-8))


rows = [("FAE", fae_in, fae_see, fae_rec, f"{args.sensors} sensors"),
        ("MAE", mae_in, mae_see, mae_rec, "75% masked")]
fig, ax = plt.subplots(2, 4, figsize=(11, 5.4))
for r, (name, inp, see, rec, seetxt) in enumerate(rows):
    vis = norm_vis(inp)
    if cmap == "Greys_r":
        lo, hi = np.percentile(vis, 1), np.percentile(vis, 99)
    else:
        hi = float(np.percentile(np.abs(vis), 99)) or 1.0; lo = -hi
    kw = dict(cmap=cmap, vmin=lo, vmax=hi, aspect="equal")
    ax[r, 0].imshow(vis, **kw); ax[r, 0].set_ylabel(name, fontsize=16, fontweight="bold", color=COLORS[name])
    sv = norm_vis(np.where(np.isnan(see), inp.mean(), see)); sv[np.isnan(see)] = np.nan
    ax[r, 1].imshow(sv, **kw)
    ax[r, 2].imshow(norm_vis(rec), **kw)
    err = np.abs(rec - inp); rl = relL2(rec, inp)
    ax[r, 3].imshow(err, cmap="magma", vmin=0, vmax=(hi - lo) * 0.5, aspect="equal")
    ax[r, 3].set_title((("|error|\n" if r == 0 else "")) + f"relL2={rl:.3f}", fontsize=13)
    for c in range(4):
        ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    print(f"{name} relL2={rl:.3f}", flush=True)
for c, t in [(0, "input"), (1, "what it sees"), (2, "reconstruction")]:
    ax[0, c].set_title(t, fontsize=14)
fig.suptitle(f"{args.dataset} — coordinate-native (FAE) vs masked-grid (MAE) reconstruction", fontsize=14)
out = f"results/figs/{args.dataset}/fig1_recon.png"; os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out, dpi=200); print(f"wrote {out}", flush=True)
