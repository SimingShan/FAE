"""Ablation 2x2 evaluated on the lenses the full-grid probe HIDES: probe-R2 and invariance-ICC
vs sensor budget, for the four cells (Senseiver / +temporal / +dual-view / full FAE), 3 seeds.
Hypothesis: temporal drives ACCURACY (probe), dual-view drives INVARIANCE (ICC) + sparse robustness.
Data loaded ONCE; GPU.
"""
import os, sys, json
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
ROOT = "/gpfs/radev/scratch/lu_lu/ss5235/WFAE"; sys.path.insert(0, ROOT)
from src.plotstyle import apply, panels, NPG, COLORS
from src.config import load_config
from src.data.ns import NSDataset
from src.data.well2d import make_coords_2d, fields_to_tokens
from src.models.fae import FAE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
apply()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
from src.config import ckpt_file
OUT = os.path.join(ROOT, "results/figures")
cfg = load_config("configs/ns/fae/default.yaml")
SIDE = cfg.resolution; NPIX = SIDE * SIDE; alphas = list(cfg.ridge_alphas)
coords = make_coords_2d(n_side=SIDE, device=DEV)
SEEDS = [0, 1, 2]; BUD = [64, 128, 256, 512, 1024, 2048, 4096]; K = 8
CELLS = {"Senseiver": "FAE_no_dual_no_temp", "+temporal": "FAE_no_dual_temp",
         "+dual-view": "FAE_dual_no_temp", "full FAE": "fae_ns128"}   # tags under ns/fae/
CCOL = {"Senseiver": NPG[5], "+temporal": NPG[3], "+dual-view": NPG[2], "full FAE": COLORS["FAE"]}

# ---- probe data loaded ONCE ----
va = NSDataset("valid", side=SIDE, mode="clip", clip_len=2, frame_stride=cfg.frame_stride, n_traj=cfg.eval_n_traj)
te = NSDataset("test", side=SIDE, mode="clip", clip_len=2, frame_stride=cfg.frame_stride, n_traj=cfg.eval_n_traj, stats=va.stats)
def pack(ds):
    X = torch.stack([ds[i][0][:, 0] for i in range(len(ds))])              # (N,C,H,W) frame0
    Y = np.array([float(ds[i][1].reshape(-1)[0]) for i in range(len(ds))])
    return X, Y
vaX, vaY = pack(va); teX, teY = pack(te)
def chfeat(X): return torch.cat([X.mean((2, 3)), X.std((2, 3))], 1).numpy()
def ridge(Ztr, ytr, Zte, yte):
    sc = StandardScaler().fit(Ztr); m, s = ytr.mean(), ytr.std() + 1e-8
    reg = RidgeCV(alphas=alphas).fit(sc.transform(Ztr), (ytr - m) / s)
    return float(r2_score((yte - m) / s, reg.predict(sc.transform(Zte))))
floor = ridge(chfeat(vaX), vaY, chfeat(teX), teY)
print(f"floor R2={floor:.3f}  | n_valid={len(vaY)} n_test={len(teY)}", flush=True)
Xicc = teX[:96].to(DEV)

@torch.no_grad()
def embed(model, X, idx, bs=64):
    Z = []
    for i in range(0, len(X), bs):
        tok = model.encode_tokens(fields_to_tokens(X[i:i + bs].to(DEV), idx), coords[idx])
        Z.append(tok.mean(1).cpu().numpy())
    return np.concatenate(Z)
def icc(Z):                                                                # Z (K,N,D)
    between = Z.mean(0).var(0, unbiased=False).sum().item()
    within = Z.var(0, unbiased=False).mean(0).sum().item()
    return between / (between + within + 1e-12)

res = {c: {b: {"r2": [], "icc": []} for b in BUD} for c in CELLS}
for c, tag in CELLS.items():
    for s in SEEDS:
        ck = torch.load(ckpt_file("fae", tag, s), map_location=DEV); a = ck["train_args"]
        m = FAE(emb_dim=a["emb_dim"], num_iter=a.get("num_iter", 4), num_latents=a["num_latents"],
                in_chans=cfg.in_chans, coord_dim=2).to(DEV); m.load_state_dict(ck["model"]); m.eval()
        for b in BUD:
            g = torch.Generator(device=DEV).manual_seed(0)
            idx = torch.randperm(NPIX, generator=g, device=DEV)[:b]
            r2 = ridge(embed(m, vaX, idx), vaY, embed(m, teX, idx), teY)
            with torch.no_grad():
                Z = torch.stack([m.encode_tokens(fields_to_tokens(Xicc, ii), coords[ii]).mean(1)
                                 for ii in (torch.randperm(NPIX, device=DEV)[:b] for _ in range(K))])
            ic = icc(Z)
            res[c][b]["r2"].append(r2); res[c][b]["icc"].append(ic)
            print(f"{c:11s} s{s} b={b:5d}  R2={r2:.3f}  ICC={ic:.3f}", flush=True)
json.dump({"res": res, "floor": floor}, open(os.path.join(OUT, "ablation_eval.json"), "w"))

fig, (a0, a1) = panels(2)
for c in CELLS:
    r2m = [np.mean(res[c][b]["r2"]) for b in BUD]; r2s = [np.std(res[c][b]["r2"]) for b in BUD]
    icm = [np.mean(res[c][b]["icc"]) for b in BUD]
    a0.errorbar(BUD, r2m, yerr=r2s, fmt="-o", color=CCOL[c], capsize=3, label=c)
    a1.plot(BUD, icm, "-o", color=CCOL[c], label=c)
a0.axhline(floor, color=COLORS["floor"], ls=":", label=f"floor {floor:.2f}")
for a, yl, t in [(a0, "buoyancy probe $R^2$", "Probe accuracy vs budget"),
                 (a1, "invariance ICC", "Invariance vs budget")]:
    a.set_xscale("log", base=2); a.set_xlabel("# sensors"); a.set_ylabel(yl); a.set_title(t); a.legend()
a1.set_ylim(0, 1.03)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "ablation_eval.png")); print("wrote ablation_eval.png")
for c in CELLS:
    print(f"{c:11s}  R2@64={np.mean(res[c][64]['r2']):.3f}  R2@full4096={np.mean(res[c][4096]['r2']):.3f}  "
          f"ICC@64={np.mean(res[c][64]['icc']):.3f}  ICC@1024={np.mean(res[c][1024]['icc']):.3f}")
