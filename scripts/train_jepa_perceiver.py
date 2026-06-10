"""I-JEPA recipe with PERCEIVER backbone + SPARSE TARGET ablation.

Differences from official I-JEPA:
  - Backbone: Perceiver (V3) — pools to 128 abstract latent tokens.
  - Target encoder sees ONLY u_B (sparse 64-point subset), NOT the full field.
    There is no mask-on-output: Perceiver pools, so there's nothing to mask.
  - Predictor: 1-block self-attn transformer with mean-Fourier B-coord bias.
  - NO VICReg, NO covariance terms.

Authentic to I-JEPA:
  - F.layer_norm on TARGET features over feature-dim (collapse prevention).
  - smooth_l1_loss (Huber).
  - EMA momentum: per-iteration linear ramp 0.996 → 1.0.
  - No reconstruction loss.

This isolates: does the I-JEPA recipe (EMA + LayerNorm + smooth_l1) work for
PDE snapshots when the target is "informationally weaker" than the context?
"""
import os, sys, time, argparse, math, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models import FAE
from src.models.fae import (
    WiderMLP, SelfAttention, Residual, _fourier_features_linear
)
from src.data.g1 import G1FrameDataset


class JEPAPredictor(nn.Module):
    """1-block self-attn transformer; injects B-coord context as bias."""
    def __init__(self, dim=320, num_heads=4, n_freq=32, max_freq=32,
                  coord_dim=1, mlp_mult=2):
        super().__init__()
        self.n_freq = n_freq; self.max_freq = max_freq
        self.ctx_proj = nn.Linear(2 * coord_dim * n_freq, dim)
        self.attn = Residual(SelfAttention(dim, num_heads, dropout=0.0))
        self.mlp  = Residual(WiderMLP(dim, mlp_mult))
        nn.init.zeros_(self.mlp.module.fc2.weight); nn.init.zeros_(self.mlp.module.fc2.bias)
        nn.init.zeros_(self.attn.module.attn.out_proj.weight)
        nn.init.zeros_(self.attn.module.attn.out_proj.bias)
        nn.init.zeros_(self.ctx_proj.weight); nn.init.zeros_(self.ctx_proj.bias)

    def forward(self, z_A, b_coords):
        cf = _fourier_features_linear(b_coords, self.n_freq, self.max_freq)
        ctx_emb = self.ctx_proj(cf.mean(dim=1))
        z = z_A + ctx_emb.unsqueeze(1)
        z = self.attn(z); z = self.mlp(z)
        return z


def train(out_path, epochs=15, batch=32, lr=5e-4, gpu=0, workers=4,
          n_A=128, n_B=64,
          ema_start=0.996, ema_end=1.0,
          time_subsample=5):
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"=== FAE I-JEPA  (Perceiver + sparse target, no mask)  device={device} ===", flush=True)
    print(f"  N_A={n_A}, N_B={n_B},  EMA τ: {ema_start} → {ema_end} (per-iter)", flush=True)
    print(f"  loss=smooth_l1, target=LayerNorm(z_B), no VICReg", flush=True)

    ds = G1FrameDataset(time_subsample=time_subsample)
    print(f"  snapshots: {len(ds):,}", flush=True)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers,
                         pin_memory=True, drop_last=True,
                         persistent_workers=(workers > 0))
    ipe = len(loader)

    X = 1024
    full_coords = torch.linspace(0, 1 - 1.0/X, X, device=device).unsqueeze(-1)

    cfg = dict(emb_dim=320, num_iter=4, depth_per_iter=4, num_latents=128,
                num_cross_heads=4, num_self_heads=8,
                n_freq=32, max_freq=32, coord_dim=1)
    online = FAE(**cfg).to(device)
    target = FAE(**cfg).to(device)
    target.load_state_dict(online.state_dict())
    for p_ in target.parameters(): p_.requires_grad_(False)

    predictor = JEPAPredictor(dim=320, num_heads=4, n_freq=32, max_freq=32,
                                coord_dim=1, mlp_mult=2).to(device)

    n_par_o = sum(p.numel() for p in online.encoder.parameters())
    n_par_p = sum(p.numel() for p in predictor.parameters())
    print(f"  online encoder params: {n_par_o/1e6:.3f}M", flush=True)
    print(f"  predictor params:      {n_par_p/1e6:.3f}M", flush=True)

    opt = torch.optim.AdamW(
        list(online.encoder.parameters()) + list(predictor.parameters()),
        lr=lr, weight_decay=1e-4)
    total_iters = epochs * ipe
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_iters)

    history = []
    t0_train = time.time()
    it_global = 0
    for ep in range(epochs):
        ls_pred = 0.0
        n_batch = 0
        for u, _cls, _coeff in loader:
            u = u.to(device, non_blocking=True).float()
            B = u.size(0)
            perm = torch.randperm(X, device=device)
            a_idx = perm[:n_A]
            b_idx = perm[n_A : n_A + n_B]

            u_A = u[:, a_idx].unsqueeze(-1)
            u_B = u[:, b_idx].unsqueeze(-1)
            x_A = full_coords[a_idx].unsqueeze(0).expand(B, -1, -1)
            x_B = full_coords[b_idx].unsqueeze(0).expand(B, -1, -1)

            z_A = online.encoder(u_A, x_A)
            with torch.no_grad():
                z_B = target.encoder(u_B, x_B)
                # I-JEPA target post-processing: LayerNorm over feature dim
                z_B = F.layer_norm(z_B, (z_B.size(-1),))
            z_B_pred = predictor(z_A, x_B)

            loss = F.smooth_l1_loss(z_B_pred, z_B)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()

            # EMA: per-iteration ramp
            frac = it_global / max(total_iters - 1, 1)
            tau = ema_start + (ema_end - ema_start) * frac
            with torch.no_grad():
                for p_o, p_t in zip(online.parameters(), target.parameters()):
                    p_t.mul_(tau).add_(p_o.data, alpha=1.0 - tau)

            ls_pred += loss.item(); n_batch += 1; it_global += 1
        ls_pred /= n_batch
        elapsed = int(time.time() - t0_train)
        print(f"ep {ep+1:>3d}/{epochs}  pred={ls_pred:.4e}  τ_end={tau:.4f}  ({elapsed}s)",
              flush=True)
        history.append({"epoch": ep + 1, "pred": ls_pred, "tau": tau, "elapsed": elapsed})

    save = {
        "method": "jepa_perceiver",
        "history": history,
        "n_par_encoder": n_par_o, "n_par_predictor": n_par_p,
        "model": online.state_dict(),
        "target": target.state_dict(),
        "predictor": predictor.state_dict(),
        "config": cfg,
        "jepa_config": {"n_A": n_A, "n_B": n_B,
                          "ema_start": ema_start, "ema_end": ema_end,
                          "loss": "smooth_l1", "target_layernorm": True,
                          "vicreg": False,
                          "time_subsample": time_subsample},
    }
    torch.save(save, out_path)
    print(f"\ndone in {int(time.time() - t0_train)}s  →  {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",   required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch",  type=int, default=32)
    ap.add_argument("--lr",     type=float, default=5e-4)
    ap.add_argument("--gpu",    type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n_A",    type=int, default=128)
    ap.add_argument("--n_B",    type=int, default=64)
    ap.add_argument("--ema_start", type=float, default=0.996)
    ap.add_argument("--ema_end",   type=float, default=1.0)
    ap.add_argument("--time_subsample", type=int, default=5)
    args = ap.parse_args()
    train(args.out, args.epochs, args.batch, args.lr, args.gpu, args.workers,
          args.n_A, args.n_B, args.ema_start, args.ema_end, args.time_subsample)
