"""Sparsity sweep: probe R2 vs number of input sensors. FAE ingests the K scattered sensors NATIVELY;
the ViT baselines (MAE/JEPA) must INTERPOLATE the K sensors onto the grid first (scipy griddata) — the
architecture-axis baseline. Shows where FAE becomes necessary. Line plot R2 vs K (primary target).
  python scripts/sparse_sweep.py --dataset typhoon
"""
import os, sys, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from torch.utils.data import DataLoader, Subset
from scipy.interpolate import griddata
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.models.fae import FAE
from src.data.well2d import make_coords_2d, fields_to_tokens
from scripts.probe_all import get_data, _frame0, ridge_multi, TARGETS, embed_floor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COL = {"FAE": "#E64B35", "MAE": "#4DBBD5", "JEPA": "#00A087", "floor": "#888888"}


def load_fae(ck, side):
    a = torch.load(ck, map_location=DEVICE)["train_args"]
    inc = a.get("in_chans") or (1 if a.get("dataset") == "typhoon" else 4 if a.get("dataset") == "shear" else 3)
    m = FAE(emb_dim=a["emb_dim"], num_iter=a.get("num_iter", 4), depth_per_iter=a.get("depth_per_iter", 4),
            num_latents=a["num_latents"], n_freq=32, max_freq=32, coord_dim=2, in_chans=inc).to(DEVICE)
    m.load_state_dict(torch.load(ck, map_location=DEVICE)["model"]); m.eval(); return m, inc


def load_vit(ck):
    from scripts.train_baseline import build_model
    a = torch.load(ck, map_location=DEVICE)["train_args"]; method = a["method"]
    m = build_model("mae" if method == "mae" else "ijepa", resolution=a["resolution"], in_chans=a["in_chans"],
                    embed_dim=a.get("embed_dim"), depth=a.get("depth"), patch_size=a.get("patch_size")).to(DEVICE)
    m.load_state_dict(torch.load(ck, map_location=DEVICE)["model"]); m.eval(); return m, method


@torch.no_grad()
def embed_fae(m, coords, side, ds, K, g):
    NPIX = side * side; Z, Y = [], []
    for x, y in DataLoader(ds, batch_size=64):
        f0 = _frame0(x).to(DEVICE)
        idx = torch.arange(NPIX, device=DEVICE) if K >= NPIX else torch.randperm(NPIX, generator=g, device=DEVICE)[:K]
        tok = m.encode_tokens(fields_to_tokens(f0, idx), coords[idx])
        Z.append(tok.mean(1).cpu().numpy()); Y.append(y.numpy())                     # canonical mean pool
    return np.concatenate(Z), np.concatenate(Y)


def interp(f0, idx, side):                                          # (C,H,W) np, idx (K,) -> (C,H,W) griddata
    if len(idx) >= side * side:
        return f0
    C = f0.shape[0]; flat = f0.reshape(C, -1)
    ys, xs = np.divmod(idx, side); gy, gx = np.mgrid[0:side, 0:side]
    pts = np.stack([ys, xs], 1)
    out = np.zeros_like(f0)
    for c in range(C):
        out[c] = griddata(pts, flat[c, idx], (gy, gx), method="linear", fill_value=float(flat[c, idx].mean()))
    return out


@torch.no_grad()
def embed_vit(m, method, side, ds, K, rng):
    NPIX = side * side; Z, Y = [], []
    for x, y in DataLoader(ds, batch_size=64):
        f0 = _frame0(x).numpy()                                    # (B,C,H,W)
        idx = np.arange(NPIX) if K >= NPIX else rng.permutation(NPIX)[:K]
        xi = np.stack([interp(f0[b], idx, side) for b in range(f0.shape[0])])    # interpolate each
        xt = torch.from_numpy(xi).float().to(DEVICE)
        tok = m.forward_encoder(xt, 0.0)[0][:, 1:] if method == "mae" else m.target(xt)
        Z.append(tok.mean(1).cpu().numpy()); Y.append(y.numpy())                     # canonical mean pool
    return np.concatenate(Z), np.concatenate(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["shear", "typhoon"], required=True)
    ap.add_argument("--sensors", type=int, nargs="+", default=[16, 64, 256, 1024, None])
    ap.add_argument("--nmax", type=int, default=600, help="subsample fields (griddata is slow)")
    args = ap.parse_args()
    side = {"shear": 128, "typhoon": 128}[args.dataset]
    tgt = TARGETS[args.dataset][0]                                  # primary target (logRe / wind)
    fit_sp, test_sp = ("valid_a", "valid_b") if args.dataset == "shear" else ("valid", "test")
    va = get_data(args.dataset, fit_sp, side); te = get_data(args.dataset, test_sp, side, stats=va.stats)
    rng = np.random.default_rng(0)
    va = Subset(va, rng.permutation(len(va))[:args.nmax]); te = Subset(te, rng.permutation(len(te))[:args.nmax])
    coords = make_coords_2d(n_side=side, device=DEVICE)
    sens = [s if s else side * side for s in args.sensors]
    # rect/non-square FAE (shear 128x256) not yet supported here: the griddata->grid baseline assumes a
    # square grid. Fail LOUDLY rather than silently mis-eval (TODO: non-square sweep design).
    _fck = (sorted(glob.glob(f"results/checkpoints/{args.dataset}/fae/*_s*.pt")) or [None])[0]
    if _fck and torch.load(_fck, map_location="cpu")["train_args"].get("res_h"):
        raise SystemExit("sparse_sweep: rect FAE (res_h set) not yet supported — needs non-square grid design.")
    print(f"=== {args.dataset} sparsity sweep: R²({tgt}) vs K {sens}  nfit={len(va)} ntest={len(te)} ===", flush=True)

    cks = {meth: (sorted(glob.glob(f"results/checkpoints/{args.dataset}/{meth}/*_s*.pt")) or [None])[0]
           for meth in ["fae", "mae", "jepa"]}
    curves = {}
    for meth, ck in cks.items():
        if ck is None: continue
        r2s = []
        for K in sens:
            g = torch.Generator(device=DEVICE).manual_seed(0)
            if meth == "fae":
                m, _ = load_fae(ck, side); Ztr, Ytr = embed_fae(m, coords, side, va, K, g); Zte, Yte = embed_fae(m, coords, side, te, K, g)
            else:
                m, mm = load_vit(ck); Ztr, Ytr = embed_vit(m, mm, side, va, K, np.random.default_rng(0)); Zte, Yte = embed_vit(m, mm, side, te, K, np.random.default_rng(0))
            r2 = ridge_multi(Ztr, Ytr[:, :1], Zte, Yte[:, :1])[0]
            r2s.append(r2); print(f"  {meth:4s} K={K:>5}  R²({tgt})={r2:+.3f}", flush=True)
        curves[meth.upper()] = r2s
    # floor (dense crude stats) as a flat reference
    Xtr, Ytr = embed_floor(va); Xte, Yte = embed_floor(te)
    floor = ridge_multi(Xtr, Ytr[:, :1], Xte, Yte[:, :1])[0]

    from src.plotstyle import apply, COLORS as PC, panels
    apply(); fig, ax = panels(1, side=5.5)
    for meth, r2s in curves.items():
        ax.plot(sens, r2s, "-o", color=PC.get(meth), label=meth)
    ax.axhline(floor, color=PC["floor"], ls="--", label=f"floor {floor:+.2f}")
    ax.set_xscale("log"); ax.set_xlabel("# input sensors (K)"); ax.set_ylabel(f"probe R²({tgt})")
    ax.set_title(f"{args.dataset}: probe vs sensors"); ax.legend()
    out = f"results/figs/{args.dataset}/sparsity_sweep.png"; os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out); print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
