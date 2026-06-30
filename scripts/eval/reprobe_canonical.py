"""THE canonical test-set probe — the ONLY probe to report (never in-log).

Full-grid encode (mean-pool) -> RidgeCV, TRAIN -> TEST (in-distribution seed split, exactly fig5's protocol).
Per physical label R2 (log-transforms raw Sc/Pr to logSc/logPr). Works for any dataset.
Encoders labelled by ROLE: Senseiver = recon (static), FAE = recon_both (Senseiver+temporal).

  python scripts/eval/reprobe_canonical.py shear
"""
import os, sys, glob, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.data.preprocessed import PDEDataset
from src.data.well2d import make_coords_2d_hw, fields_to_tokens
from src.encoders import load_fae, load_vit
from src.eval import _ridge
from benchmarks import build_model

DEV = "cuda"; ALPHAS = np.logspace(-3, 4, 8); ds = sys.argv[1]
meta = json.load(open(f"data/{ds}/meta.json")); H, W, C = meta["H"], meta["W"], meta["C"]
RAW = meta["label_names"]
LOG = [i for i, L in enumerate(RAW) if L in ("Sc", "Schmidt", "Pr", "Prandtl")]
LABELS = ["log" + L if i in LOG else L for i, L in enumerate(RAW)]
coords = make_coords_2d_hw(H, W, device=DEV); IDX = torch.arange(H * W, device=DEV)


def logify(Y):
    Y = Y.copy()
    for i in LOG: Y[:, i] = np.log10(Y[:, i])
    return Y


@torch.no_grad()
def emb_fae(ck, split):
    m, _ = load_fae(ck, DEV); m.eval(); Z, Y = [], []
    for x, y in DataLoader(PDEDataset(ds, split, mode="single", start_stride=4), batch_size=16):
        Z.append(m.encode_tokens(fields_to_tokens(x.to(DEV), IDX), coords[IDX]).mean(1).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(Z), logify(np.concatenate(Y))


@torch.no_grad()
def emb_vit(ck, split):
    m, _ = load_vit(ck, DEV); m.eval(); Z, Y = [], []
    for x, y in DataLoader(PDEDataset(ds, split, mode="single", start_stride=4), batch_size=32):
        Z.append(m.encode(x.to(DEV)).float().cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(Z), logify(np.concatenate(Y))


def emb_floor(split, dim=256, seed=0):
    rng = np.random.default_rng(seed); Z, Y, P = [], [], None
    for x, y in DataLoader(PDEDataset(ds, split, mode="single", start_stride=4), batch_size=32):
        f = F.adaptive_avg_pool2d(x, 12).reshape(x.size(0), -1).numpy()
        if P is None: P = rng.standard_normal((f.shape[1], dim)) / np.sqrt(f.shape[1])
        Z.append(np.tanh(f @ P)); Y.append(y.numpy())
    return np.concatenate(Z), logify(np.concatenate(Y))


def row(name, embfn, ck=None):
    Ztr, Ytr = (embfn(ck, "train") if ck else embfn("train")); Zte, Yte = (embfn(ck, "test") if ck else embfn("test"))
    r2 = [_ridge(Ztr, Ytr[:, j], Zte, Yte[:, j], ALPHAS)[0] for j in range(len(LABELS))]
    print(f"  {name:22s} " + "  ".join(f"{L}={r:+.3f}" for L, r in zip(LABELS, r2)), flush=True)


print(f"=== {ds} CANONICAL test-set probe (full-grid, train->test, RidgeCV) — the ONLY probe to report ===", flush=True)
row("floor (rand-proj)", emb_floor)
for sub in ["mae", "jepa"]:                                        # baselines (any saved seed-0)
    for ck in sorted(glob.glob(f"results/checkpoints/{ds}/{sub}/*_s0.pt")):
        row(sub, emb_vit, ck)
for ck in sorted(glob.glob(f"results/checkpoints/{ds}/fae/*_s0.pt")):   # role by tag: recon_both=FAE, recon=Senseiver
    tag = os.path.basename(ck).replace(".pt", "")
    role = "FAE (recon_both)" if "recon_both" in tag else ("Senseiver (recon)" if "recon" in tag else tag)
    row(role, emb_fae, ck)
print("done.", flush=True)
