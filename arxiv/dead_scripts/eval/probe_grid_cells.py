"""Held-out-CELL 2D-regression probe for coarse parameter GRIDS (shear logRe×Sc; active_matter α×ζ).
Instead of holding out a parameter VALUE (zero-variance trap) or seeds (in-distribution), hold out
INTERIOR grid CELLS (combos): each held-out cell's coordinates still appear in OTHER train cells, so it's
clean 2D INTERPOLATION on the parameter plane. Reports per-axis R2 on held-out cells + cell-classification
accuracy (secondary, in-distribution). Pools train+test splits (SSL is label-free -> no leakage)."""
import os, sys, json, glob, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from src.data.preprocessed import PDEDataset
from src.data.well2d import make_coords_2d_hw, fields_to_tokens
from src.encoders import load_fae
from src.eval import _ridge
from benchmarks import build_model

DEV = "cuda"; ALPHAS = np.logspace(-3, 4, 8)
ds = sys.argv[1] if len(sys.argv) > 1 else "shear"
meta = json.load(open(f"data/{ds}/meta.json")); H, W, C = meta["H"], meta["W"], meta["C"]
LABELS_RAW = meta["label_names"]
LOG_COLS = [i for i, L in enumerate(LABELS_RAW) if L in ("Sc", "Schmidt", "Pr", "Prandtl")]   # raw diffusivity -> log (match fig5 logSc)
LABELS = ["log" + L if i in LOG_COLS else L for i, L in enumerate(LABELS_RAW)]
coords = make_coords_2d_hw(H, W, device=DEV); IDX = torch.arange(H * W, device=DEV)


def pool():
    return ConcatDataset([PDEDataset(ds, "train", mode="single", start_stride=8),
                          PDEDataset(ds, "test", mode="single", start_stride=8)])


@torch.no_grad()
def emb_fae(ck):
    m, _ = load_fae(ck, DEV); m.eval(); Z, Y = [], []
    for x, y in DataLoader(pool(), batch_size=8):
        Z.append(m.encode_tokens(fields_to_tokens(x.to(DEV), IDX), coords[IDX]).mean(1).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(Z), np.concatenate(Y)


@torch.no_grad()
def emb_mae(ck):
    a = torch.load(ck, map_location=DEV)["train_args"]; IMG = (H, W) if H != W else H
    m = build_model("mae", IMG, C, False, embed_dim=a["embed_dim"], depth=a["depth"], patch_size=a["patch_size"], num_heads=a.get("num_heads", 8)).to(DEV)
    m.load_state_dict(torch.load(ck, map_location=DEV)["model"]); m.eval(); Z, Y = [], []
    for x, y in DataLoader(pool(), batch_size=32):
        Z.append(m.encode(x.to(DEV)).float().cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(Z), np.concatenate(Y)


@torch.no_grad()
def emb_jepa(ck):
    a = torch.load(ck, map_location=DEV)["train_args"]; IMG = (H, W) if H != W else H
    m = build_model("ijepa", IMG, C, embed_dim=a.get("embed_dim"), depth=a.get("depth"), patch_size=a.get("patch_size")).to(DEV)
    m.load_state_dict(torch.load(ck, map_location=DEV)["model"]); m.eval(); Z, Y = [], []
    for x, y in DataLoader(pool(), batch_size=32):
        Z.append(m.encode(x.to(DEV)).float().cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(Z), np.concatenate(Y)


def emb_floor(dim=256, seed=0):
    rng = np.random.default_rng(seed); Z, Y, P = [], [], None
    for x, y in DataLoader(pool(), batch_size=32):
        f = F.adaptive_avg_pool2d(x, 12).reshape(x.size(0), -1).numpy()
        if P is None: P = rng.standard_normal((f.shape[1], dim)) / np.sqrt(f.shape[1])
        Z.append(np.tanh(f @ P)); Y.append(y.numpy())
    return np.concatenate(Z), np.concatenate(Y)


def held_out_cells(Y, frac=0.25, seed=0):
    Yr = np.round(Y, 5); cells = np.unique(Yr, axis=0)
    a_vals, b_vals = np.unique(cells[:, 0]), np.unique(cells[:, 1])
    a_int, b_int = set(a_vals[1:-1].tolist()), set(b_vals[1:-1].tolist())    # interior values (each axis still in train)
    interior = [tuple(c) for c in cells if c[0] in a_int and c[1] in b_int]
    rng = np.random.default_rng(seed); rng.shuffle(interior)
    k = max(1, int(round(frac * len(interior)))); test_cells = set(interior[:k])
    is_te = np.array([tuple(r) in test_cells for r in Yr])
    return is_te, len(cells), len(interior), k


def probe(name, Z, Y):
    if LOG_COLS:                                              # log-transform raw Sc/Pr to match fig5 (logSc)
        Y = Y.copy()
        for i in LOG_COLS: Y[:, i] = np.log10(Y[:, i])
    is_te, ncell, nint, k = held_out_cells(Y)
    if is_te.sum() == 0 or (~is_te).sum() == 0:
        print(f"  {name:20s} (split degenerate)", flush=True); return
    Ztr, Zte, Ytr, Yte = Z[~is_te], Z[is_te], Y[~is_te], Y[is_te]
    r2 = [_ridge(Ztr, Ytr[:, j], Zte, Yte[:, j], ALPHAS)[0] for j in range(Y.shape[1])]
    print(f"  {name:20s} " + "  ".join(f"{L}={r:+.3f}" for L, r in zip(LABELS, r2)) +
          f"   (held out {k}/{nint} interior cells of {ncell} total)", flush=True)


print(f"=== {ds} held-out-CELL 2D-regression probe (interior-cell interpolation) ===", flush=True)
Zf, Yf = emb_floor(); probe("floor (rand-proj)", Zf, Yf)
mck = f"results/checkpoints/{ds}/mae/shr_mae_s0.pt"
if os.path.exists(mck): probe("mae", *emb_mae(mck))
jck = f"results/checkpoints/{ds}/jepa/shr_jepa_s0.pt"
if os.path.exists(jck): probe("jepa", *emb_jepa(jck))
for ck in sorted(glob.glob(f"results/checkpoints/{ds}/fae/shr_*_s0.pt")):
    probe(os.path.basename(ck).replace("_s0.pt", ""), *emb_fae(ck))
print("done.", flush=True)
