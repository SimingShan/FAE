"""V3 + Temporal alignment training recipe.

Architecture (jointly trained):
  - V3 encoder + V3 decoder
  - TemporalPropagator P: tokens_A + Δt → predicted tokens_B

Per batch:
  - Sample t_A ∈ [0, T-11];  Δt ∈ {1..10} random;  t_B = t_A + Δt
  - Encode both frames at sparse N (random from multicount)
  - Reconstruct both frames at random query coords (FAE-style)
  - Predict tokens_B from tokens_A via P (Δt-conditioned)
  - Loss = λ_rec (recA + recB) + λ_pred · ||P(z_A, Δt) − z_B||²
    (symmetric: gradients flow into BOTH branches)
  - Anti-collapse safety: small spatial VICReg (var+cov) on pooled tokens

Trained from scratch on G1 v2.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.models import FAE as V3
from src.models.fae import (CrossAttention, SelfAttention, WiderMLP, Residual,
                              _fourier_features_linear)
from src.data.g1 import load_g1_system, PDE_NAMES


# =========================================================================
# Trajectory-pair dataset
# =========================================================================
class G1PairDataset(Dataset):
    """Random (t_A, t_B = t_A+Δt) pair per item. Δt ∈ [1, dt_max]."""
    def __init__(self, time_subsample=1, dt_max=10):
        blocks = [load_g1_system(n) for n in PDE_NAMES]
        u = np.concatenate([b["u"] for b in blocks], axis=0)  # (N, T, X)
        self.u = u[:, ::time_subsample]
        self.dt_max = dt_max
        self.T = self.u.shape[1]
        # Each trajectory gives many pairs; index = (traj_idx, t_A, Δt)
        # We'll just sample at __getitem__ time
        self.K = self.u.shape[0]

    def __len__(self):
        # roughly K * (T - dt_max) per epoch
        return self.K * 10

    def __getitem__(self, idx):
        rng = np.random.default_rng(idx)
        k = rng.integers(0, self.K)
        dt = int(rng.integers(1, self.dt_max + 1))
        t_A = int(rng.integers(0, self.T - dt))
        u_A = self.u[k, t_A].astype(np.float32)
        u_B = self.u[k, t_A + dt].astype(np.float32)
        return torch.from_numpy(u_A), torch.from_numpy(u_B), dt


# =========================================================================
# Temporal token propagator
# =========================================================================
class TemporalPropagator(nn.Module):
    """1-block transformer over tokens, Δt-conditioned via Fourier features.

    tokens (B, L, D), Δt (B,) integer → predicted tokens (B, L, D)
    """
    def __init__(self, dim=320, num_heads=4, n_freq=8, max_dt=10, mlp_mult=2):
        super().__init__()
        self.n_freq = n_freq
        self.max_dt = float(max_dt)
        self.dt_proj = nn.Linear(2 * n_freq, dim)
        # Pre-LN self-attn + FFN, residual
        self.attn  = Residual(SelfAttention(dim, num_heads, dropout=0.0))
        self.mlp   = Residual(WiderMLP(dim, mlp_mult))
        # Identity-init: zero out the last linear of MLP + attn out_proj
        nn.init.zeros_(self.mlp.module.fc2.weight); nn.init.zeros_(self.mlp.module.fc2.bias)
        nn.init.zeros_(self.attn.module.attn.out_proj.weight)
        nn.init.zeros_(self.attn.module.attn.out_proj.bias)

    def forward(self, tokens, delta_t):
        """tokens: (B, L, D). delta_t: (B,) ints."""
        # Normalize Δt to [0, 1], Fourier features, project to dim, broadcast
        dt_norm = (delta_t.float() / self.max_dt).unsqueeze(-1)        # (B, 1)
        dt_feat = _fourier_features_linear(dt_norm.unsqueeze(-1), self.n_freq, 1.0)
        dt_emb = self.dt_proj(dt_feat.squeeze(1))                       # (B, D)
        tokens = tokens + dt_emb.unsqueeze(1)                            # broadcast over L
        tokens = self.attn(tokens)
        tokens = self.mlp(tokens)
        return tokens


# =========================================================================
# Anti-collapse VICReg (small)
# =========================================================================
def off_diagonal(x):
    n, m = x.shape
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def var_cov_loss(z, gamma=1.0):
    """Variance hinge + covariance off-diag, computed on a pooled latent."""
    z = z - z.mean(0)
    std = torch.sqrt(z.var(0) + 1e-4)
    l_var = torch.mean(F.relu(gamma - std))
    cov = (z.T @ z) / max(z.shape[0] - 1, 1)
    l_cov = off_diagonal(cov).pow_(2).sum() / z.shape[1]
    return l_var, l_cov


# =========================================================================
# Train
# =========================================================================
def make_projector(in_dim, mlp_spec="2048-2048-2048"):
    """Smaller VICReg expander tuned for our batch=32 (matches theory better)."""
    full = f"{in_dim}-{mlp_spec}"
    layers = []
    f = list(map(int, full.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


def train(out_path, epochs=20, batch=32, lr=5e-4, gpu=0, workers=4,
            n_query=512, mcnt_choices=(64, 128, 256, 512, 1024),
            dt_max=10, time_subsample=2,
            lam_rec=1.0, lam_pred=10.0, lam_v=1.0, lam_c=0.1,
            spatial_vicreg=False, no_recon=False,
            lam_inv=25.0, lam_spatial_v=25.0, lam_spatial_c=1.0):
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if no_recon: lam_rec = 0.0
    print(f"=== V3 + Temporal alignment (device={device}) ===", flush=True)
    print(f"  λ_rec={lam_rec}  λ_pred={lam_pred}  λ_v={lam_v}  λ_c={lam_c}  dt_max={dt_max}",
          flush=True)
    if spatial_vicreg:
        print(f"  + spatial VICReg: λ_inv={lam_inv}  λ_spatial_v={lam_spatial_v}  "
              f"λ_spatial_c={lam_spatial_c}", flush=True)

    ds = G1PairDataset(time_subsample=time_subsample, dt_max=dt_max)
    print(f"  dataset: K={ds.K} trajectories, T={ds.T}, ~{len(ds)} pairs/epoch",
          flush=True)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers,
                         pin_memory=True, drop_last=True,
                         persistent_workers=(workers > 0))

    X = 1024
    full_coords = torch.linspace(0, 1 - 1.0/X, X, device=device).unsqueeze(-1)

    model = V3(emb_dim=320, num_iter=4, depth_per_iter=4, num_latents=128,
                  num_cross_heads=4, num_self_heads=8,
                  n_freq=32, max_freq=32, coord_dim=1).to(device)
    prop = TemporalPropagator(dim=320, num_heads=4, n_freq=8,
                                max_dt=dt_max, mlp_mult=2).to(device)
    n_par_m = sum(p.numel() for p in model.parameters())
    n_par_p = sum(p.numel() for p in prop.parameters())
    print(f"  V3 params:        {n_par_m/1e6:.3f}M", flush=True)
    print(f"  Propagator params: {n_par_p/1e6:.3f}M", flush=True)

    projector = None
    if spatial_vicreg:
        projector = make_projector(320, "2048-2048-2048").to(device)
        n_par_proj = sum(p.numel() for p in projector.parameters())
        print(f"  Projector params:  {n_par_proj/1e6:.3f}M", flush=True)

    params = list(model.parameters()) + list(prop.parameters())
    if projector is not None:
        params += list(projector.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history = []
    t0 = time.time()
    for ep in range(epochs):
        # warmup
        if ep < 2:
            for g in opt.param_groups:
                g["lr"] = lr * (ep + 1) / 2
        model.train(); prop.train()
        if projector is not None: projector.train()
        agg = {"rec": 0.0, "pred": 0.0, "var": 0.0, "cov": 0.0,
                 "inv_sp": 0.0, "var_sp": 0.0, "cov_sp": 0.0, "n": 0}
        for u_A, u_B, dt in loader:
            u_A = u_A.to(device, non_blocking=True).float()
            u_B = u_B.to(device, non_blocking=True).float()
            dt = dt.to(device)
            B = u_A.size(0)

            n_in = int(mcnt_choices[np.random.randint(len(mcnt_choices))])
            idx_in = torch.randperm(X, device=device)[:n_in].sort().values
            coords_in = full_coords[idx_in]
            q_idx = torch.randperm(X, device=device)[:n_query]
            q_coords = full_coords[q_idx]
            tgt_A = u_A[:, q_idx].unsqueeze(-1)
            tgt_B = u_B[:, q_idx].unsqueeze(-1)
            u_A_in = u_A[:, idx_in].unsqueeze(-1)
            u_B_in = u_B[:, idx_in].unsqueeze(-1)

            tokens_A = model.encoder(u_A_in, coords_in)
            tokens_B = model.encoder(u_B_in, coords_in)

            l_rec = torch.tensor(0.0, device=device)
            if not no_recon:
                pred_A = model.decoder(tokens_A, q_coords)
                pred_B = model.decoder(tokens_B, q_coords)
                l_rec = 0.5 * (F.mse_loss(pred_A, tgt_A) + F.mse_loss(pred_B, tgt_B))

            tokens_B_pred = prop(tokens_A, dt)
            l_pred = F.mse_loss(tokens_B_pred, tokens_B)

            z_pool_A = tokens_A.mean(dim=1)
            l_var, l_cov = var_cov_loss(z_pool_A, gamma=1.0)

            l_inv_sp = l_var_sp = l_cov_sp = torch.tensor(0.0, device=device)
            if spatial_vicreg:
                # Second sensor subset of u_A (random), spatial pair
                n_in_2 = int(mcnt_choices[np.random.randint(len(mcnt_choices))])
                idx_in_2 = torch.randperm(X, device=device)[:n_in_2].sort().values
                coords_in_2 = full_coords[idx_in_2]
                u_A2_in = u_A[:, idx_in_2].unsqueeze(-1)
                tokens_A2 = model.encoder(u_A2_in, coords_in_2)
                z1 = projector(tokens_A.mean(dim=1))
                z2 = projector(tokens_A2.mean(dim=1))
                l_inv_sp = F.mse_loss(z1, z2)
                # var + cov on projector outputs
                z1c = z1 - z1.mean(0); z2c = z2 - z2.mean(0)
                std1 = torch.sqrt(z1c.var(0) + 1e-4)
                std2 = torch.sqrt(z2c.var(0) + 1e-4)
                l_var_sp = 0.5 * (torch.mean(F.relu(1 - std1))
                                    + torch.mean(F.relu(1 - std2)))
                D = z1.shape[1]
                cov1 = (z1c.T @ z1c) / max(B - 1, 1)
                cov2 = (z2c.T @ z2c) / max(B - 1, 1)
                l_cov_sp = (off_diagonal(cov1).pow_(2).sum() / D
                              + off_diagonal(cov2).pow_(2).sum() / D)

            loss = (lam_rec * l_rec + lam_pred * l_pred
                     + lam_v * l_var + lam_c * l_cov
                     + lam_inv * l_inv_sp + lam_spatial_v * l_var_sp
                     + lam_spatial_c * l_cov_sp)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            agg["rec"]  += float(l_rec)  * B
            agg["pred"] += float(l_pred) * B
            agg["var"]  += float(l_var)  * B
            agg["cov"]  += float(l_cov)  * B
            if spatial_vicreg:
                agg["inv_sp"] += float(l_inv_sp) * B
                agg["var_sp"] += float(l_var_sp) * B
                agg["cov_sp"] += float(l_cov_sp) * B
            agg["n"] += B
        sched.step()
        n = max(agg["n"], 1)
        line = (f"ep {ep+1:3d}/{epochs}  rec={agg['rec']/n:.4e}  "
                  f"pred={agg['pred']/n:.4e}  var={agg['var']/n:.4e}  "
                  f"cov={agg['cov']/n:.4e}")
        if spatial_vicreg:
            line += (f"  invSp={agg['inv_sp']/n:.4e}  varSp={agg['var_sp']/n:.4e}  "
                       f"covSp={agg['cov_sp']/n:.4e}")
        line += f"  ({time.time()-t0:.0f}s)"
        print(line, flush=True)
        history.append({"epoch": ep + 1,
                          "rec": agg["rec"]/n, "pred": agg["pred"]/n,
                          "var": agg["var"]/n, "cov": agg["cov"]/n,
                          "inv_sp": agg["inv_sp"]/n, "var_sp": agg["var_sp"]/n,
                          "cov_sp": agg["cov_sp"]/n,
                          "elapsed": time.time() - t0})

    save = {
        "method": "v3_temporal",
        "history": history,
        "n_par_encoder": n_par_m,
        "n_par_propagator": n_par_p,
        "model": model.state_dict(),
        "propagator": prop.state_dict(),
        "config": dict(emb_dim=320, num_iter=4, depth_per_iter=4,
                          num_latents=128, num_cross_heads=4, num_self_heads=8,
                          n_freq=32, max_freq=32, coord_dim=1),
        "prop_config": dict(dim=320, num_heads=4, n_freq=8, max_dt=dt_max, mlp_mult=2),
    }
    torch.save(save, out_path)
    print(f"\ndone in {time.time()-t0:.0f}s  →  {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dt_max", type=int, default=10)
    ap.add_argument("--time_subsample", type=int, default=2)
    ap.add_argument("--lam_pred", type=float, default=10.0)
    ap.add_argument("--lam_rec", type=float, default=1.0)
    ap.add_argument("--lam_v", type=float, default=1.0)
    ap.add_argument("--lam_c", type=float, default=0.1)
    ap.add_argument("--spatial_vicreg", action="store_true")
    ap.add_argument("--no_recon", action="store_true")
    ap.add_argument("--lam_inv", type=float, default=25.0)
    ap.add_argument("--lam_spatial_v", type=float, default=25.0)
    ap.add_argument("--lam_spatial_c", type=float, default=1.0)
    args = ap.parse_args()
    train(out_path=args.out, epochs=args.epochs, batch=args.batch, lr=args.lr,
            gpu=args.gpu, workers=args.workers, dt_max=args.dt_max,
            time_subsample=args.time_subsample,
            lam_rec=args.lam_rec, lam_pred=args.lam_pred,
            lam_v=args.lam_v, lam_c=args.lam_c,
            spatial_vicreg=args.spatial_vicreg, no_recon=args.no_recon,
            lam_inv=args.lam_inv, lam_spatial_v=args.lam_spatial_v,
            lam_spatial_c=args.lam_spatial_c)


if __name__ == "__main__":
    main()
