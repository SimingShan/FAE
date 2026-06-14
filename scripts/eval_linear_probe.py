"""The evaluation pipeline: frozen-encoder LINEAR probe of physical parameters,
with the TRIVIAL baseline as the floor. (Other evaluations are archived under
arxiv/ — out of scope for the current AE/MAE/JEPA/FAE comparison.)

For shear_flow (the discriminating benchmark): probe (logRe, logSc) from each
method's frozen representation; report R^2 and MSE on standardized labels (same
standard as the paper's Table 1 — trivial predictor -> MSE 1.0), plus the
participation ratio (collapse guard). Always reports the random/channel-mean
trivial baselines: a method only "counts" if it clearly beats them.

  python scripts/eval_linear_probe.py --method fae   --ckpt results/checkpoints/g1/fae_vicreg_shear_v1.pt
  python scripts/eval_linear_probe.py --method mae   --ckpt <mae.pt>
  python scripts/eval_linear_probe.py --method ijepa --ckpt <ijepa.pt>
  python scripts/eval_linear_probe.py --method trivial      # baselines only
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

from src.data.well2d import ShearFlowSnapshotDataset, make_coords_2d, fields_to_tokens
from src.metrics import lin_probe, r2_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NPIX = 128 * 128
PARAMS = ["logRe", "logSc"]


def participation_ratio(Z):
    Z = Z - Z.mean(0)
    e = np.clip(np.linalg.eigvalsh(Z.T @ Z / max(len(Z) - 1, 1)), 0, None)
    return float(e.sum() ** 2 / max((e ** 2).sum(), 1e-30))


def probe_report(name, Ztr, Ytr, Zva, Yva, with_pr=True):
    """Linear ridge probe per param; print R^2 + MSE (standardized labels)."""
    cols, pr = [], (f"PR={participation_ratio(Ztr):.1f}" if with_pr else "")
    for j, nm in enumerate(PARAMS):
        ytr, yva = Ytr[:, j], Yva[:, j]
        ym, ys = ytr.mean(), ytr.std() + 1e-8
        ytr_n, yva_n = (ytr - ym) / ys, (yva - ym) / ys
        r2 = lin_probe(Ztr, ytr_n, Zva, yva_n)
        mse = 1.0 - r2 * (yva_n.var())            # MSE on standardized labels
        cols.append(f"{nm} R2={r2:+.3f} MSE={max(mse,0):.3f}")
    print(f"  {name:28s} {'  '.join(cols)}   {pr}")


# ----- per-method frozen embedding -----
@torch.no_grad()
def embed_fae(ckpt, ds, n_sensors=1024):
    from src.models import FAE
    ck = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    m = FAE(**ck["config"]).to(DEVICE).eval(); m.load_state_dict(ck["model"])
    coords = make_coords_2d(device=DEVICE)
    g = torch.Generator(device=DEVICE).manual_seed(0)
    idx = torch.randperm(NPIX, generator=g, device=DEVICE)[:n_sensors]
    Z, Y = [], []
    for f, y in DataLoader(ds, batch_size=64):
        tok = m.encoder(fields_to_tokens(f.to(DEVICE), idx), coords[idx])
        Z.append(m.represent(tok).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(Z), np.concatenate(Y)


@torch.no_grad()
def embed_image_model(model, ds):
    """For MAE/AE/I-JEPA: model.encode(imgs) -> (B, D)."""
    Z, Y = [], []
    for f, y in DataLoader(ds, batch_size=64):
        Z.append(model.encode(f.to(DEVICE)).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(Z), np.concatenate(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["fae", "mae", "ae", "ijepa", "trivial"], required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n_seed", type=int, default=24)
    args = ap.parse_args()

    tr = ShearFlowSnapshotDataset("train", n_seed=args.n_seed, side=128)
    va = ShearFlowSnapshotDataset("valid", n_seed=8, side=128, stats=tr.stats)
    Ytr = np.stack([tr.logRe, tr.logSc], 1); Yva = np.stack([va.logRe, va.logSc], 1)
    Ftr = tr.fields.reshape(len(tr), -1); Fva = va.fields.reshape(len(va), -1)
    print(f"=== shear_flow linear probe  train {len(tr)} / valid {len(va)} ===")

    # --- TRIVIAL baselines (the floor — a method must beat these) ---
    print("--- trivial baselines (floor) ---")
    probe_report("channel means (4-d)",
                  tr.fields.reshape(len(tr), 4, -1).mean(-1),
                  Ytr, va.fields.reshape(len(va), 4, -1).mean(-1), Yva, with_pr=False)
    rng = np.random.default_rng(0); W = rng.standard_normal((Ftr.shape[1], 320)).astype(np.float32) / np.sqrt(Ftr.shape[1])
    probe_report("random projection -> 320d", Ftr @ W, Ytr, Fva @ W, Yva, with_pr=False)
    pca = PCA(n_components=50).fit(Ftr)
    probe_report("field PCA-50", pca.transform(Ftr), Ytr, pca.transform(Fva), Yva, with_pr=False)

    if args.method == "trivial":
        return
    assert args.ckpt, "--ckpt required"
    print(f"--- {args.method} (frozen) ---")
    if args.method == "fae":
        Ztr, Ytr2 = embed_fae(args.ckpt, tr); Zva, Yva2 = embed_fae(args.ckpt, va)
    else:
        ck = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
        if args.method in ("mae", "ae"):
            from benchmarks.mae.mae import mae_physics
            model = mae_physics().to(DEVICE).eval()
        else:
            from benchmarks.jepa.ijepa2d import ijepa2d_physics
            model = ijepa2d_physics().to(DEVICE).eval()
        model.load_state_dict(ck["model"] if "model" in ck else ck)
        Ztr, Ytr2 = embed_image_model(model, tr); Zva, Yva2 = embed_image_model(model, va)
    probe_report(args.method, Ztr, Ytr2, Zva, Yva2)


if __name__ == "__main__":
    main()
