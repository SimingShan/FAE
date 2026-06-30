"""Pooling ablation for the FAE set-latents: mean-pool vs CONCAT-TOP-PR.
Rank the 128 latent slots by TRAIN variance, keep the top-k (k swept, chosen on fit-CV), flatten -> probe.
The probe stays LINEAR in the frozen features (StandardScaler + RidgeCV); only WHICH function of the
latents (global average vs a flattened subset) the linear map sees changes. No leakage: the slot ranking
and k are fixed on the fit split and applied unchanged to test. MAE/JEPA keep their natural GAP (concat
over spatially-indexed patches is not analogous), so this table is FAE-only.
  python scripts/eval/probe_pooling.py --dataset shear
"""
import os, sys, glob, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from src.data.well2d import make_coords_2d_hw, fields_to_tokens
from src.encoders import load_fae, fae_hw
from scripts.eval.probe_all import get_data, _frame0, TARGETS

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ALPHAS = np.logspace(-2, 5, 15)          # extended ceiling — wide concat inputs need strong regularization
KS = [1, 4, 8, 16, 32]
SIDE = 128

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["shear", "typhoon", "ns"], required=True)
args = ap.parse_args()
ds = args.dataset
fit, test = ("train", "valid") if ds == "shear" else ("valid", "test")   # shear valid too small for internal split
targets = TARGETS[ds]


@torch.no_grad()
def embed_full(ck, dset, hw):            # -> (N, 128, D) full set-latents, Y  (NOT pooled)
    m, _ = load_fae(ck, DEV)
    H, W = hw; coords = make_coords_2d_hw(H, W, device=DEV); idx = torch.arange(H * W, device=DEV)
    Z, Y = [], []
    for x, y in DataLoader(dset, batch_size=64):
        tok = m.encode_tokens(fields_to_tokens(_frame0(x).to(DEV), idx), coords[idx])   # (B, 128, D)
        Z.append(tok.cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(Z), np.concatenate(Y)


def ridge(Ztr, ytr, Zte, yte):           # standardized y, StandardScaler X, RidgeCV(r2) -> (fit_cv_r2, test_r2)
    sc = StandardScaler().fit(Ztr); Ztr, Zte = sc.transform(Ztr), sc.transform(Zte)
    ym, ys = ytr.mean(), ytr.std() + 1e-8
    r = RidgeCV(alphas=ALPHAS, scoring="r2").fit(Ztr, (ytr - ym) / ys)
    return float(r.best_score_), float(r2_score((yte - ym) / ys, r.predict(Zte)))


print(f"=== {ds} pooling ablation  fit={fit}->test={test}  targets={targets}  k in {KS} ===", flush=True)
print(f"  {'cell':30s} {'target':9s} {'mean':>7s}   {'concat-top-PR (k*)':>20s}")
for ck in sorted(glob.glob(f"results/checkpoints/{ds}/fae/*_s0.pt")):
    hw = fae_hw(ck, SIDE); sz = list(hw) if hw != (SIDE, SIDE) else SIDE
    fit_ds = get_data(ds, fit, sz)
    Ttr, Ytr = embed_full(ck, fit_ds, hw)
    Tte, Yte = embed_full(ck, get_data(ds, test, sz, stats=fit_ds.stats), hw)
    for j, tname in enumerate(targets):
        ytr, yte = Ytr[:, j], Yte[:, j]
        _, mean_r2 = ridge(Ttr.mean(1), ytr, Tte.mean(1), yte)
        order = np.argsort(Ttr.var(0).mean(-1))[::-1]      # rank slots by TRAIN variance (fixed)
        best = (-9.0, None, None)                           # (fit_cv, k, test)
        for k in KS:
            top = order[:k]
            cv, te = ridge(Ttr[:, top, :].reshape(len(Ttr), -1), ytr, Tte[:, top, :].reshape(len(Tte), -1), yte)
            if cv > best[0]:
                best = (cv, k, te)
        flag = "  <-- concat wins" if best[2] > mean_r2 + 0.005 else ""
        print(f"  {os.path.basename(ck):30s} {tname:9s} {mean_r2:+7.3f}   {best[2]:+7.3f} (k={best[1]:>2}){flag}", flush=True)
