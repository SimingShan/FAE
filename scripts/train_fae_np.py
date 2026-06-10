"""Train FAE-NP (functional Neural Process) on G1 multi-PDE.

Per batch step:
  1. Sample sensor count n_sensors from mcnt_choices.
  2. Pick n_sensors random positions; randomly split into context C (n_ctx) and
     sensor-target T_s (rest).
  3. Also sample extra OFF-context query positions T_off (forces inference).
  4. Encode q(z|C) and q(z|C+T_s).
  5. Sample z from rich posterior, decode at (T_s + T_off).
  6. Loss = heteroscedastic Gaussian NLL on the targets + β·KL(q_CT||q_C)
     with per-slot free-bits floor.
  7. β linearly warms 0 → β_target over `beta_warmup_steps`.

Monitored each epoch:
  recon NLL, raw KL sum, max-per-slot KL, fraction of slots above free_bits,
  mean logvar of both posteriors (sanity check posterior collapse / explosion).
"""
from __future__ import annotations
import argparse, sys, os, time, math, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.fae_np import FAENP, gaussian_kl, het_gaussian_nll
from src.data.g1 import G1FrameDataset, make_coords_1d


def train(out_path, epochs=15, batch=32, lr=5e-4, gpu=0, workers=4,
          warmup_epochs=2, time_subsample=10,
          n_query_offcontext=384,
          context_ratio=0.5, jitter_ctx_ratio=True,
          mcnt_choices=(64, 128, 256, 512, 1024),
          beta=1.0, free_bits=0.01,
          recon_only_epochs=3, beta_warmup_epochs=4,
          d_latent=None,
          decoder_num_blocks=2,
          recon_kind="mse"):
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"=== FAE-NP on G1 (device={device}) ===", flush=True)
    print(f"  batch={batch} lr={lr} epochs={epochs} time_subsample={time_subsample}", flush=True)
    print(f"  context_ratio={context_ratio} (jitter={jitter_ctx_ratio})  "
          f"n_query_offcontext={n_query_offcontext}", flush=True)
    print(f"  mcnt_choices={mcnt_choices}", flush=True)
    print(f"  target β={beta}  free_bits={free_bits}/dim  "
          f"recon-only={recon_only_epochs} ep, β-warmup={beta_warmup_epochs} ep",
          flush=True)

    ds = G1FrameDataset(time_subsample=time_subsample)
    print(f"  total frames: {len(ds)}", flush=True)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers,
                         pin_memory=True, drop_last=True,
                         persistent_workers=(workers > 0))

    X = 1024
    full_coords = make_coords_1d(device, N=X)                          # (X, 1)

    d_latent_use = d_latent if d_latent is not None else 256
    model = FAENP(emb_dim=320, num_iter=4, depth_per_iter=4, num_latents=128,
                  num_cross_heads=4, num_self_heads=8,
                  n_freq=32, max_freq=32, coord_dim=1,
                  d_latent=d_latent_use, decoder_num_blocks=decoder_num_blocks,
                  n_context_tokens=64, dec_dim=320).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  FAE-NP params: {n_par/1e6:.3f}M  (d_latent={model.d_latent})", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = []
    t0 = time.time()
    it_global = 0

    for ep in range(epochs):
        if ep < warmup_epochs:
            for g in opt.param_groups:
                g["lr"] = lr * (ep + 1) / warmup_epochs
        model.train()
        agg = {"recon": 0.0, "kl_raw_sum": 0.0, "kl_max": 0.0,
                "kl_active_frac": 0.0,
                "logvar_C": 0.0, "logvar_CT": 0.0,
                "logvar_y": 0.0, "n": 0}

        for batch_data in loader:
            x_flat, _cls, _coeff = batch_data
            x_flat = x_flat.to(device, non_blocking=True).float()         # (B, X)
            B = x_flat.size(0)

            # --- build sensor + target sets ---
            n_sensors = int(mcnt_choices[np.random.randint(len(mcnt_choices))])
            sensor_perm = torch.randperm(X, device=device)[:n_sensors]
            ratio = context_ratio
            if jitter_ctx_ratio:
                ratio = float(np.clip(np.random.uniform(
                    context_ratio - 0.2, context_ratio + 0.2), 0.1, 0.9))
            n_ctx = max(8, min(n_sensors - 8, int(round(n_sensors * ratio))))
            ctx_idx = sensor_perm[:n_ctx]
            tgt_sensor_idx = sensor_perm[n_ctx:]                          # (n_sensors - n_ctx,)

            # extra off-context query positions (positions NOT in sensor_perm)
            mask = torch.ones(X, dtype=torch.bool, device=device)
            mask[sensor_perm] = False
            offset_pool = torch.arange(X, device=device)[mask]
            n_extra = min(n_query_offcontext, int(offset_pool.shape[0]))
            extra_idx = offset_pool[torch.randperm(offset_pool.shape[0], device=device)[:n_extra]]

            tgt_idx_all = torch.cat([tgt_sensor_idx, extra_idx])

            # --- encode context + rich (C ∪ T_sensor) ---
            u_C = x_flat[:, ctx_idx].unsqueeze(-1)
            coords_C = full_coords[ctx_idx]
            mu_C, logvar_C = model.encode_distribution(u_C, coords_C)

            ct_idx = torch.cat([ctx_idx, tgt_sensor_idx])
            u_CT = x_flat[:, ct_idx].unsqueeze(-1)
            coords_CT = full_coords[ct_idx]
            mu_CT, logvar_CT = model.encode_distribution(u_CT, coords_CT)

            # --- sample from rich posterior, decode at targets ---
            z = model.reparam(mu_CT, logvar_CT)
            target_coords = full_coords[tgt_idx_all]
            mu_y, logvar_y = model.decode(z, target_coords)
            y = x_flat[:, tgt_idx_all]

            # --- loss ---
            # MSE recon makes the decoder NEED z's information (heteroscedastic
            # logvar_y can absorb prediction uncertainty and induce posterior
            # collapse — use MSE for v1; flip to het once collapse is controlled).
            # True ELBO scaling: per-sample recon (mean over query points then
            # divided by 1 — we keep per-element to stay scale-comparable with
            # other methods), per-sample KL (sum over latent dims) averaged
            # over batch.  This puts β at the right scale.
            if recon_kind == "het":
                # mean over (B, N_q) — per-element NLL
                recon = het_gaussian_nll(y, mu_y, logvar_y).mean()
            else:
                recon = F.mse_loss(mu_y, y)
            kl_elem = gaussian_kl(mu_CT, logvar_CT, mu_C, logvar_C)       # (B, d_latent)
            # per-dim mean over batch then floor each dim (free-bits)
            kl_per_dim = kl_elem.mean(dim=0)                               # (d_latent,)
            kl_max = float(kl_per_dim.max().item())
            kl_active = float((kl_per_dim > free_bits).float().mean().item())
            kl_raw_sum = float(kl_per_dim.sum().item())
            # per-sample summed KL (with free-bits) divided by latent dim
            # so the KL term is on a similar per-element scale to recon
            kl_floored = kl_per_dim.clamp(min=free_bits).sum() / kl_per_dim.numel()

            # Two-stage β schedule: recon-only for `recon_only_epochs`, then
            # linear ramp 0→β over `beta_warmup_epochs`.
            if ep < recon_only_epochs:
                beta_t = 0.0
            elif ep < recon_only_epochs + beta_warmup_epochs:
                frac = (ep - recon_only_epochs + 0.5) / max(beta_warmup_epochs, 1)
                beta_t = beta * frac
            else:
                beta_t = beta
            loss = recon + beta_t * kl_floored

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            agg["recon"]          += float(recon) * B
            agg["kl_raw_sum"]     += kl_raw_sum   * B
            agg["kl_max"]         += kl_max       * B
            agg["kl_active_frac"] += kl_active    * B
            agg["logvar_C"]       += float(logvar_C.mean())  * B
            agg["logvar_CT"]      += float(logvar_CT.mean()) * B
            agg["logvar_y"]       += float(logvar_y.mean())  * B
            agg["n"] += B
            it_global += 1

        sched.step()
        n = max(agg["n"], 1)
        elapsed = int(time.time() - t0)
        recon_avg = agg["recon"]/n
        kl_avg = agg["kl_raw_sum"]/n
        kl_max_avg = agg["kl_max"]/n
        kl_active_avg = agg["kl_active_frac"]/n
        lvC = agg["logvar_C"]/n; lvCT = agg["logvar_CT"]/n; lvy = agg["logvar_y"]/n
        print(f"ep {ep+1:3d}/{epochs}  recon={recon_avg:+.3e}  KL_sum={kl_avg:.3e}  "
              f"KL_max={kl_max_avg:.3e}  active={kl_active_avg:.2f}  "
              f"logvar(C/CT/y)={lvC:+.2f}/{lvCT:+.2f}/{lvy:+.2f}  "
              f"β_t={beta_t:.3f}  ({elapsed}s)", flush=True)
        history.append({"epoch": ep+1, "recon": recon_avg, "kl_raw_sum": kl_avg,
                          "kl_max": kl_max_avg, "kl_active_frac": kl_active_avg,
                          "logvar_C": lvC, "logvar_CT": lvCT, "logvar_y": lvy,
                          "beta_t": beta_t, "elapsed": elapsed})

        # Intermediate save every epoch (overwrites .latest.pt so we always
        # have a fallback if the run crashes mid-training).
        latest = out_path.replace(".pt", ".latest.pt")
        torch.save({"method": "fae_np", "history": history, "n_par": n_par,
                      "model": model.state_dict(),
                      "config": dict(emb_dim=320, num_iter=4, depth_per_iter=4,
                                        num_latents=128, num_cross_heads=4, num_self_heads=8,
                                        n_freq=32, max_freq=32, coord_dim=1,
                                        d_latent=model.d_latent,
                                        decoder_num_blocks=decoder_num_blocks,
                                        decoder_mlp_mult=2),
                      "np_config": dict(context_ratio=context_ratio,
                                          jitter_ctx_ratio=jitter_ctx_ratio,
                                          n_query_offcontext=n_query_offcontext,
                                          mcnt_choices=list(mcnt_choices),
                                          beta=beta, free_bits=free_bits,
                                          recon_only_epochs=recon_only_epochs,
                                          beta_warmup_epochs=beta_warmup_epochs,
                                          time_subsample=time_subsample,
                                          recon_kind=recon_kind),
                      "epoch": ep + 1}, latest)

    save = {"method": "fae_np", "history": history, "n_par": n_par,
              "model": model.state_dict(),
              "config": dict(emb_dim=320, num_iter=4, depth_per_iter=4,
                                num_latents=128, num_cross_heads=4, num_self_heads=8,
                                n_freq=32, max_freq=32, coord_dim=1,
                                d_latent=model.d_latent,
                                decoder_num_blocks=decoder_num_blocks,
                                decoder_mlp_mult=2),
              "np_config": dict(context_ratio=context_ratio,
                                  jitter_ctx_ratio=jitter_ctx_ratio,
                                  n_query_offcontext=n_query_offcontext,
                                  mcnt_choices=list(mcnt_choices),
                                  beta=beta, free_bits=free_bits,
                                  recon_only_epochs=recon_only_epochs,
                                  beta_warmup_epochs=beta_warmup_epochs,
                                  time_subsample=time_subsample,
                                  recon_kind=recon_kind),
              }
    torch.save(save, out_path)
    print(f"\ndone in {int(time.time() - t0)}s → {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",            required=True)
    ap.add_argument("--epochs",         type=int,   default=15)
    ap.add_argument("--batch",          type=int,   default=32)
    ap.add_argument("--lr",             type=float, default=5e-4)
    ap.add_argument("--gpu",            type=int,   default=0)
    ap.add_argument("--workers",        type=int,   default=4)
    ap.add_argument("--warmup_epochs",  type=int,   default=2)
    ap.add_argument("--time_subsample", type=int,   default=10)
    ap.add_argument("--n_query_offcontext", type=int, default=384)
    ap.add_argument("--context_ratio",  type=float, default=0.5)
    ap.add_argument("--no_jitter_ctx",  action="store_true")
    ap.add_argument("--mcnt_choices",   type=int,   nargs="+",
                     default=[64, 128, 256, 512, 1024])
    ap.add_argument("--beta",           type=float, default=1.0)
    ap.add_argument("--free_bits",      type=float, default=0.1)
    ap.add_argument("--recon_only_epochs", type=int, default=3,
                     help="train pure MSE recon first (β=0) so encoder learns informative z")
    ap.add_argument("--beta_warmup_epochs", type=int, default=4)
    ap.add_argument("--d_latent",       type=int,   default=None)
    ap.add_argument("--decoder_num_blocks", type=int, default=2)
    ap.add_argument("--recon_kind", choices=["mse", "het"], default="mse",
                     help="mse forces decoder to use z; het is the full NP NLL")
    args = ap.parse_args()
    train(args.out, args.epochs, args.batch, args.lr, args.gpu, args.workers,
            args.warmup_epochs, args.time_subsample,
            args.n_query_offcontext, args.context_ratio, not args.no_jitter_ctx,
            tuple(args.mcnt_choices),
            args.beta, args.free_bits,
            args.recon_only_epochs, args.beta_warmup_epochs,
            args.d_latent, args.decoder_num_blocks, args.recon_kind)
