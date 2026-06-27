"""Unified linear probe (valid->test, RidgeCV, standardized targets) for shear / typhoon / ns.
Floor FIRST (crude channel stats), then each frozen encoder. Probes physical PROPERTIES:
  shear: logRe, logSc   |   typhoon: wind, pressure   |   ns: buoyancy
Same mean+std pooling + RidgeCV + participation-ratio (collapse guard) for FAE and the ViTs.

  python scripts/probe_all.py --dataset typhoon
"""
import os, sys, glob, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from src.data.well2d import make_coords_2d, make_coords_2d_hw, fields_to_tokens
from src.models.fae import FAE
from src.eval import _ridge as eval_ridge, _pr as eval_pr, _pool as eval_pool   # CANONICAL NS probe — reuse, don't reimplement

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3]
TARGETS = {"shear": ["logRe", "logSc"], "typhoon": ["wind", "pressure"], "ns": ["buoyancy"]}


def get_data(ds, split, side, stats=None):
    if ds == "typhoon":
        from src.data.typhoon import TyphoonDataset
        return TyphoonDataset(split, side=side, mode="single", stats=stats)
    if ds == "shear":
        from src.data.well2d import ShearFlowClipDataset
        parity = {"valid_a": 0, "valid_b": 1}.get(split)              # shear has no test dir -> disjoint valid halves
        return ShearFlowClipDataset("valid" if parity is not None else split, n_seed=8, side=side,
                                    clip_len=2, frame_stride=4, stats=stats, traj_parity=parity)
    from src.data.ns import NSDataset
    return NSDataset(split, side=side, mode="clip", clip_len=2, frame_stride=4, n_traj=16, stats=stats)


def _frame0(x):                                                   # BATCHED (B,C,H,W) or (B,C,T,H,W) -> (B,C,H,W)
    return x if x.dim() == 4 else x[:, :, 0]


def pr(Z):                                                        # CANONICAL: src.eval._pr (eigvalsh of cov)
    return eval_pr(Z)


def ridge_multi(Ztr, Ytr, Zte, Yte):                             # CANONICAL: src.eval._ridge per target (StandardScaler + standardized y)
    return [eval_ridge(Ztr, Ytr[:, k], Zte, Yte[:, k], ALPHAS)[0] for k in range(Ytr.shape[1])]


@torch.no_grad()
def embed_floor(ds):
    X, Y = [], []
    for x, y in DataLoader(ds, batch_size=256):
        f0 = _frame0(x)
        X.append(torch.cat([f0.mean((2, 3)), f0.std((2, 3))], -1).numpy()); Y.append(y.numpy())   # canonical _chstats
    return np.concatenate(X), np.concatenate(Y)                  # (N, 2C), (N, P)


def fae_hw(ck, base_side):                                       # (H, W) the FAE was trained at (rect-aware)
    a = torch.load(ck, map_location="cpu")["train_args"]
    return (a["res_h"], a["res_w"]) if a.get("res_h") else (base_side, base_side)


@torch.no_grad()
def embed_fae(ck, ds, hw):
    a = torch.load(ck, map_location=DEVICE)["train_args"]         # rebuild EXACT arch from ckpt (no hardcoded heads/freqs)
    inc = a.get("in_chans") or (1 if a.get("dataset") == "typhoon" else 3 if a.get("dataset") in ("ns", "flowbench") else 4)
    m = FAE(emb_dim=a["emb_dim"], num_iter=a.get("num_iter", 4), depth_per_iter=a.get("depth_per_iter", 5),
            num_latents=a["num_latents"], num_cross_heads=a.get("num_cross_heads", 4), num_self_heads=a.get("num_self_heads", 8),
            n_freq=a.get("n_freq", 32), max_freq=a.get("max_freq", 32), coord_dim=2, in_chans=inc).to(DEVICE)
    m.load_state_dict(torch.load(ck, map_location=DEVICE)["model"]); m.eval()
    H, W = hw; coords = make_coords_2d_hw(H, W, device=DEVICE); idx = torch.arange(H * W, device=DEVICE)
    Z, Y = [], []
    for x, y in DataLoader(ds, batch_size=64):
        f0 = _frame0(x).to(DEVICE)
        tok = m.encode_tokens(fields_to_tokens(f0, idx), coords[idx])
        Z.append(tok.mean(1).cpu().numpy()); Y.append(y.numpy())                     # canonical mean pool
    return np.concatenate(Z), np.concatenate(Y)


@torch.no_grad()
def embed_vit(ck, ds):
    from scripts.train_baseline import build_model
    a = torch.load(ck, map_location=DEVICE)["train_args"]; method = a["method"]
    m = build_model("mae" if method == "mae" else "ijepa", resolution=a["resolution"], in_chans=a["in_chans"],
                    embed_dim=a.get("embed_dim"), depth=a.get("depth"), patch_size=a.get("patch_size")).to(DEVICE)
    m.load_state_dict(torch.load(ck, map_location=DEVICE)["model"]); m.eval()
    Z, Y = [], []
    for x, y in DataLoader(ds, batch_size=128):
        f0 = _frame0(x).to(DEVICE)
        tok = m.forward_encoder(f0, 0.0)[0][:, 1:] if method == "mae" else m.target(f0)
        Z.append(tok.mean(1).cpu().numpy()); Y.append(y.numpy())                     # canonical mean pool
    return np.concatenate(Z), np.concatenate(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["shear", "typhoon", "ns"], required=True)
    ap.add_argument("--side", type=int, default=None)
    args = ap.parse_args()
    side = args.side or {"shear": 128, "typhoon": 128, "ns": 128}[args.dataset]
    tgt = TARGETS[args.dataset]
    # shear has no test dir -> split VALID into trajectory-disjoint halves (both held out from the encoder,
    # which pretrained on TRAIN). Avoids probing on encoder-pretraining data. typhoon/ns -> valid->test.
    fit_split, test_split = ("valid_a", "valid_b") if args.dataset == "shear" else ("valid", "test")
    va = get_data(args.dataset, fit_split, side)                 # square (ViTs + floor)
    te = get_data(args.dataset, test_split, side, stats=va.stats)
    ckdir = f"results/checkpoints/{args.dataset}"
    fcks = sorted(glob.glob(f"{ckdir}/fae/*_s*.pt"))
    hw = fae_hw(fcks[0], side) if fcks else (side, side)         # FAE may be rect (shear 128x256, native aspect)
    if hw != (side, side):                                       # load FAE data at ITS resolution (matches training)
        va_f = get_data(args.dataset, fit_split, list(hw)); te_f = get_data(args.dataset, test_split, list(hw), stats=va_f.stats)
    else:
        va_f, te_f = va, te
    print(f"=== {args.dataset} probe ({fit_split}->{test_split}, RidgeCV)  targets={tgt}  ViT={side} FAE={hw[0]}x{hw[1]}  "
          f"n_probe-train={len(va)} n_test={len(te)} ===", flush=True)
    rows = []

    Xtr, Ytr = embed_floor(va); Xte, Yte = embed_floor(te)
    rows.append(("FLOOR (crude stats)", ridge_multi(Xtr, Ytr, Xte, Yte), None))

    for method in ["fae", "mae", "jepa"]:
        for ck in sorted(glob.glob(f"{ckdir}/{method}/*_s*.pt")):
            try:
                Ztr, Ytr = (embed_fae(ck, va_f, fae_hw(ck, side)) if method == "fae" else embed_vit(ck, va))
                Zte, Yte = (embed_fae(ck, te_f, fae_hw(ck, side)) if method == "fae" else embed_vit(ck, te))
                rows.append((f"{method.upper()} {os.path.basename(ck)}", ridge_multi(Ztr, Ytr, Zte, Yte), pr(Ztr)))
            except Exception as e:
                rows.append((f"{method.upper()} {os.path.basename(ck)} FAILED", [float('nan')] * len(tgt), str(e)[:50]))

    hdr = "  ".join(f"R2_{t:>8s}" for t in tgt)
    print(f"\n  {'encoder':34s} {hdr}   PR")
    print("  " + "-" * (40 + 14 * len(tgt)))
    for nm, r2, p in rows:
        vals = "  ".join(f"{v:>+9.3f}" for v in r2)
        print(f"  {nm:34s} {vals}   {p if isinstance(p,str) else (f'{p:.1f}' if p else '-')}", flush=True)


if __name__ == "__main__":
    main()
