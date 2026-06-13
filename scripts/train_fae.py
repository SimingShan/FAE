"""Unified G1 multi-PDE training entry point (deterministic methods).

Trains one of {fae_recon, fae_vicreg, mlp, cnn, mae} on the combined G1
dataset (~7M params each, asserted parity by construction).

The core method is ``fae_vicreg``: per batch, two independent random sensor
subsets A and B of the same field (multi-count: |A|, |B| drawn from
mcnt_choices) are encoded by the shared FAE encoder; the loss is

  L = lam_rec * L_rec(A) + L_rec(B)) / 2
      + sim_coeff * ||proj(pool_A) - proj(pool_B)||^2     (view alignment)
      + std_coeff * VICReg variance hinge
      + cov_coeff * VICReg covariance penalty

through an 8192-8192-8192 projector (training-only, discarded afterwards).

Usage:
  python scripts/train_fae.py --method fae_vicreg --out results/checkpoints/g1/fae_vicreg.pt
  (pass a single value to --mcnt_choices for fixed-N ablations)
"""
from __future__ import annotations
import argparse, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models import FAE, MLPSparseAE, CNN1DAE, MAE1DAE
from src.data.g1 import G1FrameDataset, make_coords_1d

FAE_CONFIG = dict(emb_dim=320, num_iter=4, depth_per_iter=4, num_latents=128,
                    num_cross_heads=4, num_self_heads=8,
                    n_freq=32, max_freq=32, coord_dim=1)


def make_projector(in_dim, mlp_spec="8192-8192-8192"):
    full = f"{in_dim}-{mlp_spec}"
    layers = []
    f = list(map(int, full.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers), f[-1]


def off_diagonal(x):
    n, m = x.shape; assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_terms(projector, rA, rB, B):
    """Projected VICReg terms on the two views' representations rA, rB."""
    xz = projector(rA)
    yz = projector(rB)
    l_repr = F.mse_loss(xz, yz)
    xz = xz - xz.mean(0); yz = yz - yz.mean(0)
    std_x = torch.sqrt(xz.var(0) + 1e-4)
    std_y = torch.sqrt(yz.var(0) + 1e-4)
    l_std = (torch.mean(F.relu(1 - std_x)) / 2
              + torch.mean(F.relu(1 - std_y)) / 2)
    cov_x = (xz.T @ xz) / (B - 1)
    cov_y = (yz.T @ yz) / (B - 1)
    n_proj_dim = xz.shape[1]
    l_cov = (off_diagonal(cov_x).pow_(2).sum().div(n_proj_dim)
              + off_diagonal(cov_y).pow_(2).sum().div(n_proj_dim))
    return l_repr, l_std, l_cov


def train_one(method, out_path, epochs=20, batch=32, lr=5e-4,
                warmup_epochs=2, gpu=0, workers=4,
                time_subsample=10, n_query=512,
                mcnt_choices=(64, 128, 256, 512, 1024),
                sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0, lam_rec=1.0,
                decoder_kind="senseiver", decoder_num_blocks=2,
                decoder_mlp_mult=2, readout_queries=0):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    print(f"=== {method} on G1 multi-PDE (device={device}) ===", flush=True)

    print("loading G1 frames...", flush=True)
    ds = G1FrameDataset(time_subsample=time_subsample)
    print(f"  total frames: {len(ds)}", flush=True)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers,
                         pin_memory=True, drop_last=True,
                         persistent_workers=(workers > 0))

    X = 1024
    full_coords = make_coords_1d(device, N=X)

    fae_cfg = dict(FAE_CONFIG, decoder_kind=decoder_kind,
                    decoder_num_blocks=decoder_num_blocks,
                    decoder_mlp_mult=decoder_mlp_mult,
                    readout_queries=readout_queries)
    rep_dim = 320 * readout_queries if readout_queries > 0 else 320
    projector = None
    if method == "fae_recon":
        model = FAE(**fae_cfg).to(device)
        recipe = "sparse_recon"
    elif method == "fae_vicreg":
        model = FAE(**fae_cfg).to(device)
        projector, _ = make_projector(rep_dim, "8192-8192-8192")
        projector = projector.to(device)
        recipe = "sparse_vicreg"
    elif method == "mlp":
        model = MLPSparseAE(coord_dim=1, latent_dim=320,
                              enc_emb=640, dec_emb=640).to(device)
        recipe = "sparse_recon"
    elif method == "cnn":
        model = CNN1DAE().to(device)
        recipe = "dense_recon"
    elif method == "mae":
        model = MAE1DAE().to(device)
        recipe = "mae_dense"
    else:
        raise ValueError(f"unknown method={method}")

    n_par = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_par/1e6:.3f}M  recipe: {recipe}", flush=True)
    if projector is not None:
        n_proj_par = sum(p.numel() for p in projector.parameters())
        print(f"  projector params: {n_proj_par/1e6:.3f}M  (training-only)", flush=True)

    params = list(model.parameters())
    if projector is not None:
        params += list(projector.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = []
    t0 = time.time()
    for ep in range(epochs):
        if ep < warmup_epochs:
            for g in opt.param_groups:
                g["lr"] = lr * (ep + 1) / warmup_epochs
        model.train()
        if projector is not None:
            projector.train()
        agg = {"rec": 0.0, "n": 0}
        if recipe == "sparse_vicreg":
            agg.update({"repr": 0.0, "std": 0.0, "cov": 0.0})

        for x_flat, _cls, _coeff in loader:
            x_flat = x_flat.to(device, non_blocking=True).float()        # (B, X)
            B = x_flat.size(0)

            if recipe == "sparse_recon":
                n_A = int(mcnt_choices[np.random.randint(len(mcnt_choices))])
                q_idx = torch.randperm(X, device=device)[:n_query]
                q_coords = full_coords[q_idx]
                target = x_flat[:, q_idx].unsqueeze(-1)
                idx = torch.randperm(X, device=device)[:n_A].sort().values
                u_in = x_flat[:, idx].unsqueeze(-1)
                pred, _ = model(u_in, full_coords[idx], q_coords)
                loss = ((pred - target) ** 2).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                agg["rec"] += float(loss) * B; agg["n"] += B

            elif recipe == "sparse_vicreg":
                n_A = int(mcnt_choices[np.random.randint(len(mcnt_choices))])
                n_B = int(mcnt_choices[np.random.randint(len(mcnt_choices))])
                q_idx = torch.randperm(X, device=device)[:n_query]
                q_coords = full_coords[q_idx]
                target = x_flat[:, q_idx].unsqueeze(-1)
                idx_A = torch.randperm(X, device=device)[:n_A].sort().values
                idx_B = torch.randperm(X, device=device)[:n_B].sort().values
                u_A = x_flat[:, idx_A].unsqueeze(-1)
                u_B = x_flat[:, idx_B].unsqueeze(-1)
                pA, tA = model(u_A, full_coords[idx_A], q_coords)
                pB, tB = model(u_B, full_coords[idx_B], q_coords)
                l_rec = 0.5 * (((pA - target) ** 2).mean() + ((pB - target) ** 2).mean())
                l_repr, l_std, l_cov = vicreg_terms(
                    projector, model.represent(tA), model.represent(tB), B)
                loss = (lam_rec * l_rec + sim_coeff * l_repr
                          + std_coeff * l_std + cov_coeff * l_cov)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                agg["rec"]  += float(l_rec)  * B
                agg["repr"] += float(l_repr) * B
                agg["std"]  += float(l_std)  * B
                agg["cov"]  += float(l_cov)  * B
                agg["n"] += B

            elif recipe == "dense_recon":
                x_in = x_flat.unsqueeze(1)                                 # (B, 1, X)
                pred, _ = model(x_in)
                loss = ((pred - x_in) ** 2).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                agg["rec"] += float(loss) * B; agg["n"] += B

            elif recipe == "mae_dense":
                x_in = x_flat.unsqueeze(1)                                 # (B, 1, X)
                pred, mask = model(x_in)
                patches_gt = model.patchify(x_in)                          # (B, P, ps)
                patches_pred = model.patchify(pred)
                err = ((patches_pred - patches_gt) ** 2).mean(dim=-1)      # (B, P)
                loss = (err * mask).sum() / max(mask.sum(), 1.0)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                agg["rec"] += float(loss) * B; agg["n"] += B

        sched.step()
        n = max(agg["n"], 1)
        line = f"ep {ep+1:3d}/{epochs}  rec={agg['rec']/n:.4e}"
        for k in ("repr", "std", "cov"):
            if k in agg: line += f"  {k}={agg[k]/n:.4e}"
        line += f"  ({time.time()-t0:.0f}s)"
        print(line, flush=True)
        history.append({"epoch": ep + 1,
                          "rec": agg["rec"]/n,
                          **({k: agg[k]/n for k in ("repr","std","cov") if k in agg}),
                          "lr": opt.param_groups[0]["lr"],
                          "elapsed": time.time() - t0})

    save = {"method": method, "history": history, "n_par": n_par}
    save["model"] = model.state_dict()
    if projector is not None:
        save["projector"] = projector.state_dict()
    if method.startswith("fae"):
        save["config"] = fae_cfg
    torch.save(save, out_path)
    print(f"\ndone in {time.time()-t0:.0f}s  →  {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["fae_recon", "fae_vicreg",
                                              "mlp", "cnn", "mae"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup_epochs", type=int, default=2)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--time_subsample", type=int, default=10)
    ap.add_argument("--sim_coeff", type=float, default=25.0,
                     help="VICReg view-alignment weight (fae_vicreg)")
    ap.add_argument("--std_coeff", type=float, default=25.0)
    ap.add_argument("--cov_coeff", type=float, default=1.0)
    ap.add_argument("--mcnt_choices", type=int, nargs="+",
                     default=[64, 128, 256, 512, 1024],
                     help="sensor counts to sample from per batch; pass a single value for fixed-N")
    ap.add_argument("--decoder_kind", choices=["senseiver", "cvit"],
                     default="senseiver",
                     help="FAE decoder: senseiver (1 cross-attn block, default) or cvit (N blocks)")
    ap.add_argument("--decoder_num_blocks", type=int, default=2,
                     help="number of CViT decoder blocks (ignored if senseiver)")
    ap.add_argument("--decoder_mlp_mult", type=int, default=2,
                     help="MLP expansion in CViT decoder (ignored if senseiver)")
    ap.add_argument("--readout_queries", type=int, default=0,
                     help="0 = mean-pool representation; K>0 = learned K-query "
                          "readout (representation = flattened K*320)")
    args = ap.parse_args()
    train_one(method=args.method, out_path=args.out, epochs=args.epochs,
                batch=args.batch, lr=args.lr, warmup_epochs=args.warmup_epochs,
                gpu=args.gpu, workers=args.workers,
                time_subsample=args.time_subsample,
                sim_coeff=args.sim_coeff, std_coeff=args.std_coeff,
                cov_coeff=args.cov_coeff,
                mcnt_choices=tuple(args.mcnt_choices),
                decoder_kind=args.decoder_kind,
                decoder_num_blocks=args.decoder_num_blocks,
                decoder_mlp_mult=args.decoder_mlp_mult,
                readout_queries=args.readout_queries)


if __name__ == "__main__":
    main()
