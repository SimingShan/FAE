"""Canonical FULL-GRID probe (train->test, RidgeCV) of the restored-NS candidates vs MAE vs floor.
The FAIR metric (vs the in-log 1024-sensor self-probe). Compares to the old headline FAE 0.928 > MAE 0.905."""
import os, sys, glob, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from src.data.preprocessed import PDEDataset
from src.data.well2d import make_coords_2d_hw, fields_to_tokens
from src.encoders import load_fae
from src.eval import _ridge, _pr
from benchmarks import build_model

DEV = "cuda"; ALPHAS = np.logspace(-3, 4, 8); ds = "ns"
meta = json.load(open(f"data/{ds}/meta.json")); H, W, C = meta["H"], meta["W"], meta["C"]
coords = make_coords_2d_hw(H, W, device=DEV); IDX = torch.arange(H * W, device=DEV)


@torch.no_grad()
def emb_fae(ck, split):
    m, _ = load_fae(ck, DEV); m.eval(); Z, Y = [], []
    for x, y in DataLoader(PDEDataset(ds, split, mode="single", start_stride=4), batch_size=16):
        Z.append(m.encode_tokens(fields_to_tokens(x.to(DEV), IDX), coords[IDX]).mean(1).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(Z), np.concatenate(Y)


@torch.no_grad()
def emb_mae(ck, split):
    a = torch.load(ck, map_location=DEV)["train_args"]
    m = build_model("mae", H, C, False, embed_dim=a["embed_dim"], depth=a["depth"], patch_size=a["patch_size"], num_heads=a.get("num_heads", 8)).to(DEV)
    m.load_state_dict(torch.load(ck, map_location=DEV)["model"]); m.eval(); Z, Y = [], []
    for x, y in DataLoader(PDEDataset(ds, split, mode="single", start_stride=4), batch_size=64):
        Z.append(m.encode(x.to(DEV)).float().cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(Z), np.concatenate(Y)


def emb_floor(split, dim=256, seed=0):
    rng = np.random.default_rng(seed); Z, Y, P = [], [], None
    for x, y in DataLoader(PDEDataset(ds, split, mode="single", start_stride=4), batch_size=64):
        f = F.adaptive_avg_pool2d(x, 16).reshape(x.size(0), -1).numpy()
        if P is None: P = rng.standard_normal((f.shape[1], dim)) / np.sqrt(f.shape[1])
        Z.append(np.tanh(f @ P)); Y.append(y.numpy())
    return np.concatenate(Z), np.concatenate(Y)


def score(Ztr, Ytr, Zte, Yte):
    return _ridge(Ztr, Ytr[:, 0], Zte, Yte[:, 0], ALPHAS)[0], _pr(Ztr)


print("=== NS CANONICAL full-grid probe (train->test, RidgeCV) — buoyancy R2 ===", flush=True)
print("    (old headline: FAE-twoview 0.928 > MAE 0.905, 3-seed valid->test)", flush=True)
Ftr, Ytr = emb_floor("train"); Fte, Yte = emb_floor("test")
r, _ = score(Ftr, Ytr, Fte, Yte); print(f"  {'floor (rand-proj)':24s} {r:+.4f}", flush=True)
mck = "results/checkpoints/ns/mae/repro_mae_s0.pt"
if os.path.exists(mck):
    Ztr, Ytr = emb_mae(mck, "train"); Zte, Yte = emb_mae(mck, "test"); r, p = score(Ztr, Ytr, Zte, Yte)
    print(f"  {'mae (repro)':24s} {r:+.4f}  PR={p:.1f}", flush=True)
cks = sorted(glob.glob("results/checkpoints/ns/fae/repro_recon_both_s0.pt") +     # default nl128 dt1 ref (canonical 0.931)
             glob.glob("results/checkpoints/ns/fae/dt1g_*_s0.pt"))                # dt1 num_latents x mcnt sweep
for ck in cks:
    tag = os.path.basename(ck).replace("_s0.pt", "")
    Ztr, Ytr = emb_fae(ck, "train"); Zte, Yte = emb_fae(ck, "test"); r, p = score(Ztr, Ytr, Zte, Yte)
    print(f"  {tag:24s} {r:+.4f}  PR={p:.1f}", flush=True)
print("done.", flush=True)
