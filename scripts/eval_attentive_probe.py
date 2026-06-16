"""Attentive-probe eval — Qu et al. protocol, for CROSS-PAPER parity only.

Frozen encoder -> THEIR AttentiveClassifier (learned query cross-attends the
encoder's token grid) -> [log10(Re), raw Sc]. MSE on standardized labels = the
same metric as their Table 1 (JEPA 0.38 / VideoMAE 0.67 / DISCO 0.13 / MPP 0.59).

This is the *same-standard* readout; the internal method comparison stays a plain
linear probe (eval_linear_probe.py) by design. Tokens are extracted once (frozen)
then the attentive head is trained ~100 epochs.

  python scripts/eval_attentive_probe.py --method fae --temporal \
      --ckpt results/checkpoints/g1/fae_vicreg_shear_align224t.pt
"""
import sys, os, argparse, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "external/physical-representation-learning"))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from physics_jepa.attentive_pooler import AttentiveClassifier
from src.metrics import r2_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PARAMS = ["logRe", "Sc"]


@torch.no_grad()
def extract_tokens(method, ckpt, temporal, res, n_seed, n_sensors=1024):
    """Frozen encoder -> token grids (N, num_tokens, D) for train and valid."""
    from src.data.well2d import (ShearFlowSnapshotDataset, ShearFlowWindowDataset,
                                  make_coords_2d, make_coords_3d, fields_to_tokens)
    if temporal:
        tr = ShearFlowWindowDataset("train", n_seed=n_seed, n_frames=16, side=res)
        va = ShearFlowWindowDataset("valid", n_seed=8, n_frames=16, side=res, stats=tr.stats)
        coords = make_coords_3d(16, res, DEVICE); NPIX = 16 * res * res
    else:
        tr = ShearFlowSnapshotDataset("train", n_seed=n_seed, frame_stride=12, side=res)
        va = ShearFlowSnapshotDataset("valid", n_seed=8, frame_stride=12, side=res, stats=tr.stats)
        coords = make_coords_2d(res, DEVICE); NPIX = res * res

    ck = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    if method == "fae":
        from src.models import FAE
        m = FAE(**ck["config"]).to(DEVICE).eval(); m.load_state_dict(ck["model"])
        g = torch.Generator(device=DEVICE).manual_seed(0)
        idx = torch.randperm(NPIX, generator=g, device=DEVICE)[:n_sensors]
        enc = lambda f: m.encoder(fields_to_tokens(f.to(DEVICE), idx), coords[idx])  # (B,128,320)
    elif method in ("mae", "ae"):
        from benchmarks.mae.mae import mae_physics
        m = mae_physics(img_size=res).to(DEVICE).eval()
        m.load_state_dict(ck["model"] if "model" in ck else ck)
        enc = lambda f: m.forward_encoder(f.to(DEVICE), 0.0)[0][:, 1:, :]            # (B,P,256)
    else:
        from benchmarks.jepa.ijepa2d import ijepa2d_physics
        m = ijepa2d_physics(img_size=res).to(DEVICE).eval()
        m.load_state_dict(ck["model"] if "model" in ck else ck)
        enc = lambda f: m.target(f.to(DEVICE))                                       # (B,P,256)

    def run(ds):
        Z, Y = [], []
        for f, y in DataLoader(ds, batch_size=48):
            Z.append(enc(f).float().cpu()); Y.append(y)
        return torch.cat(Z), torch.cat(Y)
    return run(tr), run(va)


def train_attentive(Ztr, Ytr, Zva, Yva, epochs, lr, batch, heads):
    D = Ztr.shape[-1]
    clf = AttentiveClassifier(embed_dim=D, num_heads=heads, depth=1, num_classes=2).to(DEVICE)
    ym, ys = Ytr.mean(0), Ytr.std(0) + 1e-8
    Ytr_n = ((Ytr - ym) / ys).to(DEVICE)
    Yva_n = (Yva - ym) / ys
    Zva_d = Zva.to(DEVICE)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    N = len(Ztr); best = None
    for ep in range(epochs):
        clf.train(); perm = torch.randperm(N)
        for i0 in range(0, N, batch):
            ix = perm[i0:i0 + batch]
            pred = clf(Ztr[ix].to(DEVICE))
            loss = F.mse_loss(pred, Ytr_n[ix])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        clf.eval()
        with torch.no_grad():
            pv = clf(Zva_d).cpu()
        r2 = [r2_score(pv[:, j].numpy(), Yva_n[:, j].numpy()) for j in range(2)]
        mse = [float(((pv[:, j] - Yva_n[:, j]) ** 2).mean()) for j in range(2)]
        if best is None or sum(mse) / 2 < best["avg_mse"]:
            best = {"avg_mse": sum(mse) / 2, "r2": r2, "mse": mse, "ep": ep + 1}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["fae", "mae", "ae", "ijepa"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--temporal", action="store_true", help="FAE 16-frame windows")
    ap.add_argument("--n_seed", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    args = ap.parse_args()
    res = int(torch.load(args.ckpt, map_location="cpu", weights_only=False)
              .get("train_args", {}).get("resolution", 224))

    print(f"=== attentive probe [{args.method}{'/temporal' if args.temporal else ''}] "
          f"res={res} ===", flush=True)
    t0 = time.time()
    (Ztr, Ytr), (Zva, Yva) = extract_tokens(args.method, args.ckpt, args.temporal, res, args.n_seed)
    print(f"  tokens train{tuple(Ztr.shape)} valid{tuple(Zva.shape)}  ({time.time()-t0:.0f}s)", flush=True)
    b = train_attentive(Ztr, Ytr, Zva, Yva, args.epochs, args.lr, args.batch, args.heads)
    print(f"  best @ep{b['ep']}:  "
          f"logRe R2={b['r2'][0]:+.3f} MSE={b['mse'][0]:.3f}   "
          f"Sc R2={b['r2'][1]:+.3f} MSE={b['mse'][1]:.3f}   "
          f"avg-MSE={b['avg_mse']:.3f}  (paper: JEPA 0.38 / VideoMAE 0.67 / DISCO 0.13)", flush=True)


if __name__ == "__main__":
    main()
