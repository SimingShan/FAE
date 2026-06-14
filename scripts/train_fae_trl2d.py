"""FAE+VICReg on The Well turbulent_radiative_layer_2D — snapshot (2D) or
spatiotemporal (3D) — with collapse monitoring (participation ratio).

Self-supervised pretrain (two independent coordinate-set views + VICReg) then a
FROZEN-encoder ridge probe of log10(t_cool). Reports the representation's
participation ratio (PR) so a collapsed "detector" latent (PR ~ 1-2) is not
mistaken for a genuine win: a probe number only counts if PR is healthy (>~10).

Snapshot:       coord_dim=2 over (x, y),     in_chans=4
Spatiotemporal: coord_dim=3 over (x, y, t),  in_chans=4   (--temporal)
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
from src.data.well2d import (TRL2DSnapshotDataset, TRL2DWindowDataset,
                               make_coords_2d, make_coords_3d, fields_to_tokens)
from src.metrics import lin_probe

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def participation_ratio(Z):
    Z = Z - Z.mean(0)
    e = np.linalg.eigvalsh(Z.T @ Z / max(len(Z) - 1, 1))
    e = np.clip(e, 0, None)
    return float(e.sum() ** 2 / max((e ** 2).sum(), 1e-30))


@torch.no_grad()
def embed(model, ds, coords, idx, batch=64):
    model.eval()
    c_in = coords[idx]
    Z, Y = [], []
    for fields, y in DataLoader(ds, batch_size=batch):
        tok = model.encoder(fields_to_tokens(fields.to(DEVICE), idx), c_in)
        Z.append(model.represent(tok).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(Z), np.concatenate(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--frame_stride", type=int, default=4)
    ap.add_argument("--n_query", type=int, default=1024)
    ap.add_argument("--lam_rec", type=float, default=1.0)
    ap.add_argument("--sim", type=float, default=25.0)
    ap.add_argument("--std", type=float, default=25.0)
    ap.add_argument("--cov", type=float, default=1.0)
    ap.add_argument("--n_freq", type=int, default=16)
    ap.add_argument("--proj_dim", type=int, default=8192)
    ap.add_argument("--readout_queries", type=int, default=0)
    ap.add_argument("--temporal", action="store_true", help="coord_dim=3 (x,y,t) windows")
    ap.add_argument("--n_frames", type=int, default=8)
    ap.add_argument("--win_stride", type=int, default=8)
    ap.add_argument("--mcnt", type=int, nargs="+", default=None)
    ap.add_argument("--tag", default="snap")
    args = ap.parse_args()

    cd = 3 if args.temporal else 2
    print(f"=== FAE+VICReg trl_2D [{args.tag}]  coord_dim={cd} in_chans=4 "
          f"batch={args.batch} sim/std/cov={args.sim}/{args.std}/{args.cov} "
          f"n_freq={args.n_freq} readout={args.readout_queries} ===", flush=True)

    if args.temporal:
        tr = TRL2DWindowDataset("train", n_frames=args.n_frames, win_stride=args.win_stride)
        va = TRL2DWindowDataset("valid", n_frames=args.n_frames, win_stride=args.win_stride, stats=tr.stats)
        coords = make_coords_3d(args.n_frames, device=DEVICE)
        mcnt = args.mcnt or [512, 1024, 2048, 4096]
    else:
        tr = TRL2DSnapshotDataset("train", frame_stride=args.frame_stride)
        va = TRL2DSnapshotDataset("valid", frame_stride=args.frame_stride, stats=tr.stats)
        coords = make_coords_2d(device=DEVICE)
        mcnt = args.mcnt or [256, 512, 1024, 2048]
    NPIX = coords.shape[0]
    print(f"  train {len(tr)}  valid {len(va)}  | grid pts {NPIX}  mcnt {mcnt}", flush=True)
    loader = DataLoader(tr, batch_size=args.batch, shuffle=True, drop_last=True,
                         num_workers=4, pin_memory=True)

    model = FAE(emb_dim=320, num_iter=4, depth_per_iter=4, num_latents=128,
                  num_cross_heads=4, num_self_heads=8, n_freq=args.n_freq, max_freq=32,
                  coord_dim=cd, in_chans=4, readout_queries=args.readout_queries).to(DEVICE)
    rep_dim = 320 * args.readout_queries if args.readout_queries > 0 else 320
    proj = make_projector(rep_dim, f"{args.proj_dim}-{args.proj_dim}-{args.proj_dim}").to(DEVICE)
    params = list(model.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"  FAE params {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    g0 = torch.Generator(device=DEVICE).manual_seed(0)
    probe_idx = torch.randperm(NPIX, generator=g0, device=DEVICE)[:min(1024, NPIX)]

    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); proj.train()
        agg = {"rec": 0, "sim": 0, "std": 0, "cov": 0, "n": 0}
        for fields, _y in loader:
            fields = fields.to(DEVICE, non_blocking=True)
            B = fields.size(0)
            nA, nB = int(np.random.choice(mcnt)), int(np.random.choice(mcnt))
            iA = torch.randperm(NPIX, device=DEVICE)[:nA]
            iB = torch.randperm(NPIX, device=DEVICE)[:nB]
            iq = torch.randperm(NPIX, device=DEVICE)[:args.n_query]
            target = fields_to_tokens(fields, iq)
            pA, tA = model(fields_to_tokens(fields, iA), coords[iA], coords[iq])
            pB, tB = model(fields_to_tokens(fields, iB), coords[iB], coords[iq])
            l_rec = 0.5 * (F.mse_loss(pA, target) + F.mse_loss(pB, target))
            l_sim, l_std, l_cov = vicreg(proj, model.represent(tA), model.represent(tB), B)
            loss = args.lam_rec * l_rec + args.sim * l_sim + args.std * l_std + args.cov * l_cov
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            for k, v in [("rec", l_rec), ("sim", l_sim), ("std", l_std), ("cov", l_cov)]:
                agg[k] += float(v) * B
            agg["n"] += B
        sched.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            Ztr, _ = embed(model, tr, coords, probe_idx)
            pr = participation_ratio(Ztr)
            n = agg["n"]
            print(f"ep {ep+1:3d}/{args.epochs}  rec={agg['rec']/n:.3e} sim={agg['sim']/n:.3e} "
                  f"std={agg['std']/n:.3e} cov={agg['cov']/n:.3e}  PR={pr:.1f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    out = f"results/checkpoints/g1/fae_vicreg_trl2d_{args.tag}.pt"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"model": model.state_dict(),
                  "config": dict(emb_dim=320, num_iter=4, depth_per_iter=4, num_latents=128,
                                   num_cross_heads=4, num_self_heads=8, n_freq=args.n_freq, max_freq=32,
                                   coord_dim=cd, in_chans=4, readout_queries=args.readout_queries),
                  "stats": tr.stats, "args": vars(args)}, out)

    # ---- frozen probe + PR ----
    Ztr, Ytr = embed(model, tr, coords, probe_idx)
    Zva, Yva = embed(model, va, coords, probe_idx)
    pr_tr = participation_ratio(Ztr)
    ym, ys = Ytr.mean(), Ytr.std() + 1e-8
    r2 = lin_probe(Ztr, (Ytr - ym) / ys, Zva, (Yva - ym) / ys)
    verdict = "GENUINE" if pr_tr > 8 else "COLLAPSED (probe invalid)"
    print(f"\n=== [{args.tag}] PR={pr_tr:.2f}  linear-probe R^2(log10 t_cool)={r2:.3f}  "
          f"-> {verdict} ===", flush=True)
    print(f"  saved {out}", flush=True)


if __name__ == "__main__":
    main()
