"""Representation-QUALITY battery (beyond a single probe) on the restored-NS candidates:
  (1) effective rank (PR + entropy)  -> non-collapse
  (2) off-diagonal correlation magnitude -> non-redundant
  (3) decodability of a BATTERY of physical diagnostics -> rank spent on USEFUL structure.
All fair across encoders (FAE full-grid / MAE full-image), train->test ridge, same diagnostics."""
import os, sys, glob, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
from torch.utils.data import DataLoader
from src.data.preprocessed import PDEDataset
from src.data.well2d import make_coords_2d_hw, fields_to_tokens
from src.encoders import load_fae
from src.eval import _ridge
from benchmarks import build_model

DEV = "cuda"; ALPHAS = np.logspace(-3, 4, 8); ds = "ns"
meta = json.load(open(f"data/{ds}/meta.json")); H, W, C = meta["H"], meta["W"], meta["C"]
coords = make_coords_2d_hw(H, W, device=DEV); IDX = torch.arange(H * W, device=DEV)
DIAGS = ["buoyancy", "KE", "enstrophy", "smoke_mass", "plume_height", "max_speed"]


def diagnostics(x, y_buoy):                                    # x (B,3,H,W) raw-ish (normalized) -> (B,6)
    sm, vx, vy = x[:, 0], x[:, 1], x[:, 2]
    KE = (vx ** 2 + vy ** 2).mean((1, 2))
    w = (vy[:, 1:-1, 2:] - vy[:, 1:-1, :-2]) - (vx[:, 2:, 1:-1] - vx[:, :-2, 1:-1])   # central-diff vorticity
    ens = (w ** 2).mean((1, 2))
    mass = sm.mean((1, 2))
    yg = torch.linspace(0, 1, sm.shape[1], device=sm.device).view(1, -1, 1)
    com = (yg * sm).sum((1, 2)) / (sm.sum((1, 2)).abs() + 1e-4)
    vmax = (vx ** 2 + vy ** 2).sqrt().amax((-1, -2)) if False else (vx ** 2 + vy ** 2).sqrt().flatten(1).amax(1)
    return torch.stack([y_buoy, KE, ens, mass, com, vmax], 1)


@torch.no_grad()
def collect(enc_fn, split):
    Z, Y = [], []
    for x, y in DataLoader(PDEDataset(ds, split, mode="single", start_stride=4), batch_size=32):
        Z.append(enc_fn(x.to(DEV)).float().cpu().numpy())
        Y.append(diagnostics(x.to(DEV), y[:, 0].to(DEV)).cpu().numpy())
    return np.concatenate(Z), np.concatenate(Y)


def eff_rank(Z):                                              # PR + entropy effective rank
    Zc = Z - Z.mean(0); ev = np.clip(np.linalg.eigvalsh(np.cov(Zc.T)), 0, None)
    pr = ev.sum() ** 2 / max((ev ** 2).sum(), 1e-30)
    p = ev / max(ev.sum(), 1e-30); p = p[p > 0]
    erank = float(np.exp(-(p * np.log(p)).sum()))
    return float(pr), erank


def off_diag(Z):                                             # mean |corr| off-diagonal (redundancy)
    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-8); Ccorr = (Zs.T @ Zs) / len(Zs)
    n = Ccorr.shape[0]; off = Ccorr[~np.eye(n, dtype=bool)]
    return float(np.abs(off).mean())


def run(name, enc_fn):
    Ztr, Ytr = collect(enc_fn, "train"); Zte, Yte = collect(enc_fn, "test")
    pr, er = eff_rank(Ztr); od = off_diag(Ztr)
    r2 = [_ridge(Ztr, Ytr[:, j], Zte, Yte[:, j], ALPHAS)[0] for j in range(len(DIAGS))]
    useful = sum(r > 0.5 for r in r2)
    print(f"\n{name}  (dim={Ztr.shape[1]})", flush=True)
    print(f"  PR={pr:.1f}  eff_rank={er:.1f}  off_diag|corr|={od:.4f}  useful(R2>0.5)={useful}/{len(DIAGS)}", flush=True)
    print("  " + "  ".join(f"{d}={r:+.3f}" for d, r in zip(DIAGS, r2)), flush=True)


print("=== NS representation-quality battery (full-grid, train->test) ===", flush=True)
mck = "results/checkpoints/ns/mae/repro_mae_s0.pt"
a = torch.load(mck, map_location=DEV)["train_args"]
mae = build_model("mae", H, C, False, embed_dim=a["embed_dim"], depth=a["depth"], patch_size=a["patch_size"], num_heads=a.get("num_heads", 8)).to(DEV)
mae.load_state_dict(torch.load(mck, map_location=DEV)["model"]); mae.eval()
run("MAE (repro)", lambda x: mae.encode(x))
for tag in ["repro_recon_both", "repro_recon", "recon_both_linpred"]:
    ck = f"results/checkpoints/ns/fae/{tag}_s0.pt"
    if not os.path.exists(ck): continue
    m, _ = load_fae(ck, DEV); m.eval()
    run(f"FAE {tag}", lambda x, m=m: m.encode_tokens(fields_to_tokens(x, IDX), coords[IDX]).mean(1))
print("\ndone.", flush=True)
