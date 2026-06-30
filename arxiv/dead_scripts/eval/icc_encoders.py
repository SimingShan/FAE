"""fig4 (cross-encoder) — observation-invariance ICC vs #sensors: FAE vs MAE vs JEPA. Parallel to fig3.
Encode each field with TWO sensor draws; ICC = between-field var / (between + within-draw var). Per-dim mean.
FAE native-sparse; MAE/JEPA linear-interp (same as fig3). Fast (~128 fields, GPU).
  python scripts/figs/icc_encoders.py --dataset ns
"""
import os, sys, glob, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
from torch.utils.data import DataLoader
import matplotlib; matplotlib.use("Agg")
from src.plotstyle import apply, COLORS, panels
apply()
from src.data.well2d import make_coords_2d_hw, fields_to_tokens
from src.sparse import linear_weights, apply_linear
from src.encoders import load_fae, load_vit
from scripts.eval.probe_all import get_data, _frame0, fae_hw

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["shear", "typhoon", "ns"], required=True)
ap.add_argument("--nfields", type=int, default=128)
args = ap.parse_args()
ds = args.dataset; SIDE = 128


def pick(m, pats):
    for p in pats:
        c = sorted(glob.glob(f"results/checkpoints/{ds}/{m}/{p}"))
        if c:
            return c[0]
    return None


# ---- fields (held-out) at FAE resolution; gives in_chans (FAE ckpt stores None) ----
fck = pick("fae", ["*128*_s0.pt", "*224*_s0.pt", "*_s0.pt"])
H, W = fae_hw(fck, SIDE); NPIX = H * W
split = "valid" if ds == "shear" else "test"
dset = get_data(ds, split, list((H, W)) if (H, W) != (SIDE, SIDE) else SIDE)
X = []
for x, _ in DataLoader(dset, batch_size=64):
    X.append(_frame0(x))
    if sum(t.shape[0] for t in X) >= args.nfields:
        break
X = torch.cat(X)[:args.nfields].to(DEV)
fae, _ = load_fae(fck, DEV)                                       # default (recon_both/128) cell
coords = make_coords_2d_hw(H, W, device=DEV)
vits = [(nm,) + load_vit(pick(nm.lower(), ["*_s0.pt"]), DEV) for nm in ["MAE", "JEPA"] if pick(nm.lower(), ["*_s0.pt"])]


def icc(z1, z2):
    Z = np.stack([z1, z2]); within = Z.var(0).mean(0); between = Z.mean(0).var(0)
    return round(float(np.mean(between / (between + within + 1e-9))), 4)


SENS = [k for k in [64, 256, 1024, 4096, 16384, 32768] if k <= NPIX]
g = torch.Generator(device=DEV).manual_seed(0)
res = {"FAE": [], "MAE": [], "JEPA": []}

for K in SENS:
    i1 = torch.randperm(NPIX, generator=g, device=DEV)[:K]; i2 = torch.randperm(NPIX, generator=g, device=DEV)[:K]
    bs = max(8, 65536 // K); full = K >= NPIX
    with torch.no_grad():
        def fae_enc(idx): return np.concatenate([fae.encode_tokens(fields_to_tokens(X[j:j+bs], idx), coords[idx]).mean(1).cpu().numpy() for j in range(0, len(X), bs)])
        res["FAE"].append(icc(fae_enc(i1), fae_enc(i2)))
        w1, w2 = (None, None) if full else (linear_weights(i1, H, W, DEV), linear_weights(i2, H, W, DEV))
        for nm, m, meth in vits:
            def venc(idx, w):
                Z = []
                for j in range(0, len(X), bs):
                    xb = X[j:j+bs]
                    f = xb if full else apply_linear(xb.reshape(xb.shape[0], xb.shape[1], -1)[:, :, idx], w).reshape(xb.shape)
                    tok = m.forward_encoder(f, 0.0)[0][:, 1:] if meth == "mae" else m.target(f)
                    Z.append(tok.mean(1).cpu().numpy())
                return np.concatenate(Z)
            res[nm].append(icc(venc(i1, w1), venc(i2, w2)))
    print(f"K={K:6d}  FAE={res['FAE'][-1]:.3f}  MAE={res['MAE'][-1] if res['MAE'] else float('nan'):.3f}  JEPA={res['JEPA'][-1] if res['JEPA'] else float('nan'):.3f}", flush=True)

fig, ax = panels(1, side=5.5)
for nm in ["FAE", "MAE", "JEPA"]:
    if res[nm]:
        ax.plot(SENS, res[nm], "-o", color=COLORS[nm], label=nm)
ax.set_xscale("log", base=2); ax.set_xticks(SENS); ax.set_xticklabels([str(s) for s in SENS])
ax.set_xlabel("# sensors"); ax.set_ylabel("observation-invariance ICC"); ax.set_ylim(0, 1.02)
ax.set_title(f"{ds}: invariance vs sensors (FAE / MAE / JEPA)"); ax.legend(loc="lower right")
out = f"results/figs/{ds}/fig4_icc_encoders.png"; os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=200); print(f"wrote {out}", flush=True)
