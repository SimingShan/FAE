"""FAE+VICReg on The Well turbulent_radiative_layer_2D (2D, 4-channel).

Self-supervised pretraining (two independent sensor views + VICReg) on
single 2D snapshots, then a FROZEN-encoder ridge probe of log10(t_cool) —
the same evaluation protocol as the JEPA baseline (which on this dataset,
6-epoch pretrain + frozen probe, gets R^2 ~ 0.71). Directly comparable.

Run:  CUDA_VISIBLE_DEVICES=1 python scripts/train_fae_trl2d.py
"""
from __future__ import annotations
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models import FAE
from src.data.well2d import TRL2DSnapshotDataset, make_coords_2d, fields_to_tokens
from src.metrics import lin_probe, r2_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NPIX = 128 * 128


def make_projector(in_dim, spec="8192-8192-8192"):
    f = [in_dim] + [int(x) for x in spec.split("-")]
    layers = []
    for i in range(len(f) - 2):
        layers += [nn.Linear(f[i], f[i+1]), nn.BatchNorm1d(f[i+1]), nn.ReLU(True)]
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


def off_diag(x):
    n, m = x.shape; assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg(projector, rA, rB, B):
    xz, yz = projector(rA), projector(rB)
    l_sim = F.mse_loss(xz, yz)
    xz, yz = xz - xz.mean(0), yz - yz.mean(0)
    l_std = (F.relu(1 - torch.sqrt(xz.var(0) + 1e-4)).mean() / 2
              + F.relu(1 - torch.sqrt(yz.var(0) + 1e-4)).mean() / 2)
    d = xz.shape[1]
    l_cov = (off_diag((xz.T @ xz) / (B - 1)).pow_(2).sum() / d
              + off_diag((yz.T @ yz) / (B - 1)).pow_(2).sum() / d)
    return l_sim, l_std, l_cov


@torch.no_grad()
def embed(model, ds, coords, idx, batch=64):
    """Frozen pooled embeddings at a FIXED sensor subset `idx` + labels."""
    model.eval()
    c_in = coords[idx]
    Z, Y = [], []
    loader = DataLoader(ds, batch_size=batch)
    for fields, y in loader:
        fields = fields.to(DEVICE)
        vals = fields_to_tokens(fields, idx)
        tok = model.encoder(vals, c_in)
        Z.append(model.represent(tok).cpu().numpy())
        Y.append(y.numpy())
    return np.concatenate(Z), np.concatenate(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--frame_stride", type=int, default=4)
    ap.add_argument("--mcnt", type=int, nargs="+", default=[256, 512, 1024, 2048])
    ap.add_argument("--n_query", type=int, default=1024)
    ap.add_argument("--sim", type=float, default=25.0)
    ap.add_argument("--std", type=float, default=25.0)
    ap.add_argument("--cov", type=float, default=1.0)
    ap.add_argument("--out", default="results/checkpoints/g1/fae_vicreg_trl2d.pt")
    args = ap.parse_args()

    print("=== FAE+VICReg on trl_2D (2D, 4-chan) ===", flush=True)
    tr = TRL2DSnapshotDataset("train", frame_stride=args.frame_stride)
    va = TRL2DSnapshotDataset("valid", frame_stride=args.frame_stride, stats=tr.stats)
    print(f"  train {len(tr)}  valid {len(va)} snapshots", flush=True)
    loader = DataLoader(tr, batch_size=args.batch, shuffle=True, drop_last=True,
                         num_workers=4, pin_memory=True)
    coords = make_coords_2d(device=DEVICE)

    model = FAE(emb_dim=320, num_iter=4, depth_per_iter=4, num_latents=128,
                  num_cross_heads=4, num_self_heads=8, n_freq=16, max_freq=32,
                  coord_dim=2, in_chans=4).to(DEVICE)
    proj = make_projector(320).to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  FAE params: {n_par/1e6:.2f}M", flush=True)

    params = list(model.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); proj.train()
        agg = {"rec": 0, "sim": 0, "std": 0, "cov": 0, "n": 0}
        for fields, _y in loader:
            fields = fields.to(DEVICE, non_blocking=True)
            B = fields.size(0)
            nA = int(np.random.choice(args.mcnt)); nB = int(np.random.choice(args.mcnt))
            iA = torch.randperm(NPIX, device=DEVICE)[:nA]
            iB = torch.randperm(NPIX, device=DEVICE)[:nB]
            iq = torch.randperm(NPIX, device=DEVICE)[:args.n_query]
            target = fields_to_tokens(fields, iq)                # (B, Nq, 4)
            pA, tA = model(fields_to_tokens(fields, iA), coords[iA], coords[iq])
            pB, tB = model(fields_to_tokens(fields, iB), coords[iB], coords[iq])
            l_rec = 0.5 * (F.mse_loss(pA, target) + F.mse_loss(pB, target))
            l_sim, l_std, l_cov = vicreg(proj, model.represent(tA), model.represent(tB), B)
            loss = l_rec + args.sim * l_sim + args.std * l_std + args.cov * l_cov
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            for k, v in [("rec", l_rec), ("sim", l_sim), ("std", l_std), ("cov", l_cov)]:
                agg[k] += float(v) * B
            agg["n"] += B
        sched.step()
        n = agg["n"]
        print(f"ep {ep+1:3d}/{args.epochs}  rec={agg['rec']/n:.4e} sim={agg['sim']/n:.4e} "
              f"std={agg['std']/n:.4e} cov={agg['cov']/n:.4e}  ({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"method": "fae_vicreg_trl2d", "model": model.state_dict(),
                  "config": dict(emb_dim=320, num_iter=4, depth_per_iter=4, num_latents=128,
                                   num_cross_heads=4, num_self_heads=8, n_freq=16, max_freq=32,
                                   coord_dim=2, in_chans=4),
                  "stats": tr.stats}, args.out)
    print(f"saved {args.out}", flush=True)

    # ---- frozen-encoder probe of log10(t_cool) ----
    print("\n=== frozen FAE probe (log10 t_cool) ===", flush=True)
    g = torch.Generator(device=DEVICE).manual_seed(0)
    probe_idx = torch.randperm(NPIX, generator=g, device=DEVICE)[:1024]  # fixed for train+val
    Ztr, Ytr = embed(model, tr, coords, probe_idx)
    Zva, Yva = embed(model, va, coords, probe_idx)
    # standardize labels by train stats (match JEPA's normalized-MSE -> R^2)
    ym, ys = Ytr.mean(), Ytr.std() + 1e-8
    r2 = lin_probe(Ztr, (Ytr - ym) / ys, Zva, (Yva - ym) / ys)
    print(f"  FAE+VICReg  linear-probe R^2(log10 t_cool) = {r2:.3f}", flush=True)
    print(f"  [JEPA baseline on same dataset: R^2 ~ 0.71]", flush=True)


if __name__ == "__main__":
    main()
