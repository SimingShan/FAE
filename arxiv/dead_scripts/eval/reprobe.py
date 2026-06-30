"""Canonical FULL-GRID probe on the 27 campaign checkpoints — the FAIR FAE-vs-MAE comparison.
(The in-log self-probe undersold FAE: it used 1024 random sensors for FAE vs the full image for MAE.)
  FAE: full-grid encode (ALL pixels as sensors) -> 128 latents -> mean-pool.
  MAE: encode(full image).
  Ridge: StandardScaler + RidgeCV, train -> test, standardized labels (src.eval._ridge). Floor = random proj.
"""
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

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ALPHAS = np.logspace(-3, 4, 8)
LABELS = {"ns": ["buoyancy"], "shear": ["logRe", "Sc"], "sw": ["alpha", "beta"]}


@torch.no_grad()
def embed_fae(ck, ds, split, H, W, bs):
    m, _ = load_fae(ck, DEV); m.eval()
    coords = make_coords_2d_hw(H, W, device=DEV); idx = torch.arange(H * W, device=DEV)
    Z, Y = [], []
    for x, y in DataLoader(PDEDataset(ds, split, mode="single", start_stride=8), batch_size=bs):
        tok = m.encode_tokens(fields_to_tokens(x.to(DEV), idx), coords[idx])      # (B,128,D) full grid
        Z.append(tok.mean(1).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(Z), np.concatenate(Y)


@torch.no_grad()
def embed_mae(ck, ds, split, IMG, C):
    a = torch.load(ck, map_location=DEV)["train_args"]
    m = build_model("mae", IMG, C, False, embed_dim=a["embed_dim"], depth=a["depth"],
                    patch_size=a["patch_size"], num_heads=a.get("num_heads", 8)).to(DEV)
    m.load_state_dict(torch.load(ck, map_location=DEV)["model"]); m.eval()
    Z, Y = [], []
    for x, y in DataLoader(PDEDataset(ds, split, mode="single", start_stride=8), batch_size=64):
        Z.append(m.encode(x.to(DEV)).float().cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(Z), np.concatenate(Y)


@torch.no_grad()
def embed_dino(ck, ds, split, IMG, C):
    a = torch.load(ck, map_location=DEV)["train_args"]
    m = build_model("dino", IMG, C, False, embed_dim=a["embed_dim"], depth=a["depth"],
                    patch_size=a["patch_size"], num_heads=a.get("num_heads", 8)).to(DEV)
    m.load_state_dict(torch.load(ck, map_location=DEV)["model"], strict=False); m.eval()   # backbone only (head dims may differ)
    Z, Y = [], []
    for x, y in DataLoader(PDEDataset(ds, split, mode="single", start_stride=8), batch_size=64):
        Z.append(m.encode(x.to(DEV)).float().cpu().numpy()); Y.append(y.numpy())     # teacher backbone, mean-pool
    return np.concatenate(Z), np.concatenate(Y)


def embed_floor(ds, split, dim=256, seed=0):                                     # random-proj of 16x16-pooled field
    rng = np.random.default_rng(seed); Z, Y, P = [], [], None
    for x, y in DataLoader(PDEDataset(ds, split, mode="single", start_stride=8), batch_size=64):
        f = F.adaptive_avg_pool2d(x, 16).reshape(x.size(0), -1).numpy()
        if P is None: P = rng.standard_normal((f.shape[1], dim)) / np.sqrt(f.shape[1])
        Z.append(np.tanh(f @ P)); Y.append(y.numpy())
    return np.concatenate(Z), np.concatenate(Y)


def probe(Ztr, Ytr, Zte, Yte, names):
    return [_ridge(Ztr, Ytr[:, j], Zte, Yte[:, j], ALPHAS)[0] for j in range(len(names))]


for ds in (sys.argv[1:] or ["ns", "shear", "sw"]):
    meta = json.load(open(f"data/{ds}/meta.json")); H, W, C = meta["H"], meta["W"], meta["C"]
    IMG = (H, W) if H != W else H; bs = 8 if ds == "shear" else 16; names = LABELS[ds]
    print(f"\n==== {ds}  canonical FULL-GRID probe (train->test, RidgeCV)  {names} ====", flush=True)
    Ftr, Ytr = embed_floor(ds, "train"); Fte, Yte = embed_floor(ds, "test")
    print(f"  {'floor(rand-proj)':22s} " + "  ".join(f"{n}={r:+.3f}" for n, r in zip(names, probe(Ftr, Ytr, Fte, Yte, names))), flush=True)
    mck = f"results/checkpoints/{ds}/mae/mae_s0.pt"
    if os.path.exists(mck):
        Ztr, Ytr = embed_mae(mck, ds, "train", IMG, C); Zte, Yte = embed_mae(mck, ds, "test", IMG, C)
        print(f"  {'mae':22s} " + "  ".join(f"{n}={r:+.3f}" for n, r in zip(names, probe(Ztr, Ytr, Zte, Yte, names))) + f"  PR={_pr(Ztr):.1f}", flush=True)
    dtag = {"ns": "dino_ns128", "shear": "dino_shear", "sw": "dino_sw128"}[ds]
    dck = f"results/checkpoints/{ds}/dino/{dtag}_s0.pt"
    if os.path.exists(dck):
        try:
            Ztr, Ytr = embed_dino(dck, ds, "train", IMG, C); Zte, Yte = embed_dino(dck, ds, "test", IMG, C)
            print(f"  {'dino':22s} " + "  ".join(f"{n}={r:+.3f}" for n, r in zip(names, probe(Ztr, Ytr, Zte, Yte, names))) + f"  PR={_pr(Ztr):.1f}", flush=True)
        except Exception as e:
            print(f"  {'dino':22s} SKIPPED ({type(e).__name__})", flush=True)         # non-fatal -> never blocks the FAE table
    CAMPAIGN = ["recon", "twoview_present", "recon_both_dt1", "recon_both_dt5", "recon_both_dt10",
                "twoview_dt1", "twoview_dt5", "twoview_dt10",
                "recon_local", "twoview_local",                                       # +local-neighbourhood ablation
                "recon_m64", "recon_m1024", "recon_m4096",                            # +fixed-sparsity recon (train at one sensor count; 4096 = 25% of 128^2)
                "recon_nbhd", "recon_nbhd_r4", "recon_nbhd_r16",                     # +point-to-neighborhood decoder target
                "recon_nbhd_m64", "recon_nbhd_m1024",                                 # +2x2: fixed input x neighborhood target
                "recon_q4096",                                                        # +global random query, n_query=4096 (vs 1024)
                "recon_nbhd_m1024_e400", "recon_nbhd_m64_e400", "recon_nbhd_e400",     # +neighborhood 400-epoch (convergence: nbhd needs more epochs)
                "recon_q4096_m1024",                                                  # +fixed-1024-input -> 4096 global query (sparse-in/dense-out)
                "recon_qfull"]                                                        # +query ALL 16384 points every step (deterministic full-field output)
    for tag in CAMPAIGN:
        ck = f"results/checkpoints/{ds}/fae/{tag}_s0.pt"
        if not os.path.exists(ck): continue
        Ztr, Ytr = embed_fae(ck, ds, "train", H, W, bs); Zte, Yte = embed_fae(ck, ds, "test", H, W, bs)
        print(f"  {tag:22s} " + "  ".join(f"{n}={r:+.3f}" for n, r in zip(names, probe(Ztr, Ytr, Zte, Yte, names))) + f"  PR={_pr(Ztr):.1f}", flush=True)
