"""Sensor sweep -> results/sweeps/sensor_sweep_<ds>.json  (feeds figs 3 + 4).  GPU.
  probe R2 @ 2^x sensors: FAE (native sparse) vs MAE/JEPA (griddata-interp -> grid).
  ICC   @ 2^x sensors: the 4 FAE ablation cells (Senseiver/+temporal/+dual-view/full).
  python scripts/figs/sweep_sensors.py --dataset typhoon
"""
import os, sys, json, glob, argparse, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import numpy as np, torch
from torch.utils.data import DataLoader
from src.data.well2d import make_coords_2d_hw, fields_to_tokens
from src.eval import _ridge
from src.sparse import linear_weights, apply_linear
from src.encoders import load_fae, load_vit
from scripts.eval.probe_all import get_data, _frame0, TARGETS

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3]
# SENS defined below once NPIX (full grid) is known
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["shear", "typhoon", "ns"], required=True)
ap.add_argument("--nmax", type=int, default=600)
args = ap.parse_args()
ds = args.dataset; SIDE = 128; tgt = TARGETS[ds][0]
fit, test = ("train", "valid") if ds == "shear" else ("valid", "test")   # shear valid too small for internal split
CK = lambda tag_glob: (sorted(glob.glob(f"results/checkpoints/{ds}/fae/{tag_glob}")) or [None])[0]
CELLS = {"Senseiver": "*no_dual_no_temp*", "+temporal": "*",  # +temporal = the recon_both default
         "+dual-view": "*dual_no_temp*", "full FAE": "*dual_temp*"}


@torch.no_grad()
def fields_labels(dset, hw, nmax):
    H, W = hw; X, Y = [], []
    for x, y in DataLoader(dset, batch_size=64):
        X.append(_frame0(x)); Y.append(y)
        if sum(t.shape[0] for t in X) >= nmax: break
    return torch.cat(X)[:nmax].to(DEV), torch.cat(Y)[:nmax].numpy()


# ===== probe R2 vs sensors: FAE (sparse) + MAE/JEPA (interp) =====
g = torch.Generator(device=DEV).manual_seed(0)
fae_ck = CK("*_s0.pt"); fm, (H, W) = load_fae([c for c in [CK("*recon_both*"), CK("*128*"), CK("*224*"), fae_ck] if c][0], DEV)
_sz = list((H, W)) if (H, W) != (SIDE, SIDE) else SIDE
_fit = get_data(ds, fit, _sz); Xf, Yf = fields_labels(_fit, (H, W), args.nmax)
Xft, Yft = fields_labels(get_data(ds, test, _sz, stats=_fit.stats), (H, W), args.nmax)   # SHARE fit stats (else test mis-normalized -> endpoint != dense)
coords = make_coords_2d_hw(H, W, device=DEV); NPIX = H * W
mae_ck = (sorted(glob.glob(f"results/checkpoints/{ds}/mae/*_s0.pt")) or [None])[0]
jepa_ck = (sorted(glob.glob(f"results/checkpoints/{ds}/jepa/*_s0.pt")) or [None])[0]
PERM = torch.randperm(NPIX, generator=g, device=DEV)           # nested; SAME sensors+grid for FAE and ViT (fair); ViT reuses Xf/Xft
SENS = [k for k in [64, 256, 1024, 4096, 16384, 32768] if k <= NPIX]   # up to FULL grid -> endpoint == dense probe (fig5)


@torch.no_grad()
def fae_probe_at(K):
    idx = PERM[:K]                                              # nested: first K of one fixed ordering (same for fit+test)
    bs = max(8, 65536 // K)                                     # shrink batch at high K (avoid OOM: K=16384 -> bs=8)
    def emb(X):
        Z = [fm.encode_tokens(fields_to_tokens(X[i:i+bs], idx), coords[idx]).mean(1).cpu().numpy() for i in range(0, len(X), bs)]
        return np.concatenate(Z)
    return _ridge(emb(Xf), Yf[:, 0], emb(Xft), Yft[:, 0], ALPHAS)[0]


@torch.no_grad()
def vit_probe_at(ck, K):
    m, meth = load_vit(ck, DEV)
    idx = PERM[:K]; full = K >= NPIX                            # full grid -> field itself (== dense probe / fig5 endpoint)
    vw = None if full else linear_weights(idx, H, W, DEV)      # LINEAR interp weights built ONCE, applied on GPU to all fields
    bs = max(8, 65536 // K)
    def emb(X):
        Z = []
        for i in range(0, len(X), bs):
            xb = X[i:i+bs]                                       # (B,C,H,W) already on DEV
            if full:
                f = xb
            else:
                vals = xb.reshape(xb.shape[0], xb.shape[1], -1)[:, :, idx]   # (B,C,K) sensor values (same K as FAE)
                f = apply_linear(vals, vw).reshape(xb.shape)    # linear interp -> dense grid (GPU)
            tok = m.forward_encoder(f, 0.0)[0][:, 1:] if meth == "mae" else m.target(f)
            Z.append(tok.mean(1).cpu().numpy())
        return np.concatenate(Z)
    return _ridge(emb(Xf), Yf[:, 0], emb(Xft), Yft[:, 0], ALPHAS)[0]


probe = {"FAE": [], "MAE": [], "JEPA": []}
for K in SENS:
    probe["FAE"].append(round(fae_probe_at(K), 4))
    if mae_ck: probe["MAE"].append(round(vit_probe_at(mae_ck, K), 4))
    if jepa_ck: probe["JEPA"].append(round(vit_probe_at(jepa_ck, K), 4))
    print(f"K={K:5d}  FAE={probe['FAE'][-1]:+.3f}  MAE={probe['MAE'][-1] if mae_ck else float('nan'):+.3f}", flush=True)


# ===== ICC vs sensors: 4 FAE cells (encode each field with 2 sensor draws) =====
def icc_cells(K):
    out = {}
    for cell, tg in CELLS.items():
        ck = CK(tg + "_s0.pt" if "*" in tg and not tg.endswith("*") else tg)
        ck = (sorted(glob.glob(f"results/checkpoints/{ds}/fae/{tg if tg!='*' else '*recon_both*'}*_s0.pt")) or [CK("*_s0.pt")])[0]
        if ck is None: continue
        m, (h, w) = load_fae(ck, DEV); cc = make_coords_2d_hw(h, w, device=DEV); npx = h * w
        Xc, _ = fields_labels(get_data(ds, test, list((h, w)) if (h, w) != (SIDE, SIDE) else SIDE), (h, w), min(96, args.nmax))
        with torch.no_grad():
            i1 = torch.randperm(npx, generator=g, device=DEV)[:K]; i2 = torch.randperm(npx, generator=g, device=DEV)[:K]
            bs = max(4, 65536 // K)                                # batch encode (avoid OOM at high K, e.g. shear 32768)
            enc = lambda ii: np.concatenate([m.encode_tokens(fields_to_tokens(Xc[j:j+bs], ii), cc[ii]).mean(1).cpu().numpy()
                                             for j in range(0, len(Xc), bs)])
            z1, z2 = enc(i1), enc(i2)
        Z = np.stack([z1, z2])                                    # (2, N, D)
        within = Z.var(0).mean(0); between = Z.mean(0).var(0)     # per-dim
        out[cell] = round(float(np.mean(between / (between + within + 1e-9))), 4)
    return out


icc = {c: [] for c in CELLS}
for K in SENS:
    r = icc_cells(K)
    for c in CELLS:
        if c in r: icc[c].append(r[c])
    print(f"ICC K={K:5d}  " + "  ".join(f"{c}={r.get(c,float('nan')):.3f}" for c in CELLS), flush=True)

os.makedirs("results/sweeps", exist_ok=True)
out = f"results/sweeps/sensor_sweep_{ds}.json"
json.dump({"dataset": ds, "target": tgt, "sensors": SENS, "probe": probe,
           "icc": {c: v for c, v in icc.items() if v}}, open(out, "w"), indent=2)
print(f"wrote {out}", flush=True)
