"""Cross-sparsity probe: separate TRAINING-sparsity effect from EVAL-sparsity (OOD) effect.

Probe each FAE checkpoint at several EVAL sensor counts (same random sensors across models for fairness).
If a sparse-trained model is good at its native eval count but bad at full-grid -> eval-OOD (latent fine).
If it's bad everywhere -> below the recoverability threshold (latent genuinely failed to learn).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
from torch.utils.data import DataLoader
from src.data.preprocessed import PDEDataset
from src.data.well2d import make_coords_2d_hw, fields_to_tokens
from src.encoders import load_fae
from src.eval import _ridge

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DS = "ns"
EVAL_N = [64, 256, 1024, 4096, 16384]                          # 16384 = full grid (128^2)
TAGS = ["recon_m64", "recon_m1024", "recon_m4096", "recon"]    # fixed-64, fixed-1024, fixed-4096, mixed
ALPHAS = np.logspace(-3, 4, 8)

meta = json.load(open(f"data/{DS}/meta.json")); H, W = meta["H"], meta["W"]
coords_all = make_coords_2d_hw(H, W, device=DEV)
g = torch.Generator().manual_seed(0)
eval_idx = {n: (torch.arange(H * W) if n >= H * W else torch.randperm(H * W, generator=g)[:n]).to(DEV) for n in EVAL_N}


@torch.no_grad()
def embed(m, split, idx):
    Z, Y = [], []
    for x, y in DataLoader(PDEDataset(DS, split, mode="single", start_stride=8), batch_size=16):
        tok = m.encode_tokens(fields_to_tokens(x.to(DEV), idx), coords_all[idx])
        Z.append(tok.mean(1).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(Z), np.concatenate(Y)


print(f"=== {DS} cross-sparsity probe (buoyancy R2, train->test)  rows=train-sparsity, cols=eval-sparsity ===", flush=True)
hdr = "train\\eval"
print(f"{hdr:14s} " + "  ".join(f"{n:>6d}" for n in EVAL_N), flush=True)
for tag in TAGS:
    ck = f"results/checkpoints/{DS}/fae/{tag}_s0.pt"
    if not os.path.exists(ck):
        print(f"{tag:14s} (not saved yet)", flush=True); continue
    m, _ = load_fae(ck, DEV); m.eval()
    row = []
    for n in EVAL_N:
        Ztr, Ytr = embed(m, "train", eval_idx[n]); Zte, Yte = embed(m, "test", eval_idx[n])
        r2 = _ridge(Ztr, Ytr[:, 0], Zte, Yte[:, 0], ALPHAS)[0]
        row.append(r2)
    print(f"{tag:14s} " + "  ".join(f"{r:+.3f}" for r in row), flush=True)
