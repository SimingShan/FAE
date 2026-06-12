"""Train FAE-NP (functional Neural Process) on G1 multi-PDE.

Per batch step:
  1. Sample sensor count n_sensors from mcnt_choices.
  2. Pick n_sensors random positions; randomly split into context C (n_ctx) and
     sensor-target T_s (rest).
  3. Also sample extra OFF-context query positions T_off (forces inference).
  4. Encode q(z|C) and q(z|C+T_s).
  5. Sample z from rich posterior, decode at (T_s + T_off).
  6. Loss = recon (MSE or heteroscedastic NLL)
         + beta   * KL(q_CT || q_C)            with per-dim free-bits floor
         + anchor * KL(q_C  || N(0, I))        global prior anchor
         + vicreg_mu_std/cov * VICReg variance/covariance on mu_CT (optional)
  7. beta and anchor share a schedule: recon-only epochs, then linear warmup.

2026-06-10 fixes (after the degenerate-posterior diagnosis):
  - logvar_param="sigmoid" (smooth bounds; the legacy hard clamp was a
    gradient-dead corner the encoder parked in).
  - The anchor term removes the two escape routes of the C/CT-only KL:
    inflating ||mu|| and maxing prior variance. It also makes z ~ N(0, I)
    decoding (unconditional function generation) meaningful.
  - Optional VICReg-on-mu targets the linear-probe gap vs FAE+VICReg.

Monitored each epoch: recon, raw KL sum, max-per-slot KL, fraction of slots
above free_bits, anchor KL, mean |mu_C|, mean logvar of both posteriors,
fraction of logvar within 0.05 of the upper bound (saturation watch).
"""
from __future__ import annotations
import argparse, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.fae_np import FAENP, gaussian_kl, het_gaussian_nll
from src.data.g1 import G1FrameDataset, make_coords_1d


def off_diagonal(x):
    n, m = x.shape; assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def make_projector(in_dim, mlp_spec="8192-8192-8192"):
    """VICReg expander (training-only), as in train_fae.py."""
    import torch.nn as nn
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
          warmup_epochs=2, time_subsample=10,
          n_query_offcontext=384,
          context_ratio=0.5, jitter_ctx_ratio=True,
          mcnt_choices=(64, 128, 256, 512, 1024),
          beta=1e-3, free_bits=0.1,
          anchor=1e-3,
          vicreg_mu_std=0.0, vicreg_mu_cov=0.0,
          align_kl=0.0, proj_sim=0.0, proj_std=0.0, proj_cov=0.0,
          recon_only_epochs=2, beta_warmup_epochs=4,
          d_latent=None,
          decoder_num_blocks=2,
          recon_kind="mse",
          logvar_param="sigmoid",
          det_path=False, det_drop=0.25):
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"=== FAE-NP on G1 (device={device}) ===", flush=True)
    print(f"  batch={batch} lr={lr} epochs={epochs} time_subsample={time_subsample}", flush=True)
    print(f"  context_ratio={context_ratio} (jitter={jitter_ctx_ratio})  "
          f"n_query_offcontext={n_query_offcontext}", flush=True)
    print(f"  mcnt_choices={mcnt_choices}", flush=True)
    print(f"  target β={beta}  anchor={anchor}  free_bits={free_bits}/dim  "
          f"recon-only={recon_only_epochs} ep, warmup={beta_warmup_epochs} ep", flush=True)
    print(f"  logvar_param={logvar_param}  vicreg_mu std/cov={vicreg_mu_std}/{vicreg_mu_cov}",
          flush=True)

    ds = G1FrameDataset(time_subsample=time_subsample)
    print(f"  total frames: {len(ds)}", flush=True)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers,
                         pin_memory=True, drop_last=True,
                         persistent_workers=(workers > 0))

    X = 1024
    full_coords = make_coords_1d(device, N=X)                          # (X, 1)

    d_latent_use = d_latent if d_latent is not None else 256
    config = dict(emb_dim=320, num_iter=4, depth_per_iter=4, num_latents=128,
                    num_cross_heads=4, num_self_heads=8,
                    n_freq=32, max_freq=32, coord_dim=1,
                    d_latent=d_latent_use, decoder_num_blocks=decoder_num_blocks,
                    decoder_mlp_mult=2,
                    n_context_tokens=64, dec_dim=320,
                    logvar_param=logvar_param,
                    det_path=det_path)
    model = FAENP(**config).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  FAE-NP params: {n_par/1e6:.3f}M  (d_latent={model.d_latent})", flush=True)
    lv_max = model.latent_head.logvar_max

    # Unified objective: independent second view + projected VICReg on mu +
    # symmetric-KL posterior alignment. Active when any weight > 0.
    unified = (align_kl > 0) or (proj_sim > 0) or (proj_std > 0) or (proj_cov > 0)
    projector = None
    if proj_sim > 0 or proj_std > 0 or proj_cov > 0:
        projector = make_projector(d_latent_use).to(device)
        n_proj = sum(p.numel() for p in projector.parameters())
        print(f"  projector params: {n_proj/1e6:.1f}M  (training-only)", flush=True)

    np_config = dict(context_ratio=context_ratio,
                       jitter_ctx_ratio=jitter_ctx_ratio,
                       n_query_offcontext=n_query_offcontext,
                       mcnt_choices=list(mcnt_choices),
                       beta=beta, free_bits=free_bits, anchor=anchor,
                       vicreg_mu_std=vicreg_mu_std, vicreg_mu_cov=vicreg_mu_cov,
                       align_kl=align_kl, proj_sim=proj_sim,
                       proj_std=proj_std, proj_cov=proj_cov,
                       recon_only_epochs=recon_only_epochs,
                       beta_warmup_epochs=beta_warmup_epochs,
                       time_subsample=time_subsample,
                       recon_kind=recon_kind,
                       det_path=det_path, det_drop=det_drop)

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
        agg = {k: 0.0 for k in ("recon", "kl_raw_sum", "kl_max", "kl_active_frac",
                                  "anchor_kl", "mu_abs", "lv_sat_frac",
                                  "vic_std", "vic_cov",
                                  "align_kl", "p_sim", "p_std", "p_cov",
                                  "logvar_C", "logvar_CT", "logvar_y")}
        agg["n"] = 0

        # Shared ramp for beta and anchor: recon-only, then linear warmup.
        if ep < recon_only_epochs:
            ramp = 0.0
        elif ep < recon_only_epochs + beta_warmup_epochs:
            ramp = (ep - recon_only_epochs + 0.5) / max(beta_warmup_epochs, 1)
        else:
            ramp = 1.0
        beta_t = beta * ramp
        # Anchor keeps a 10% floor from step 1: with no scale constraint the
        # recon-only phase lets |mu| drift (attempt 2 reached |mu|~18 with
        # sigma~0.06), and the arriving KL ramp then shocks sigma to the
        # ceiling. A small always-on pull keeps the geometry sane.
        anchor_t = anchor * max(ramp, 0.1)

        for x_flat, _cls, _coeff in loader:
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
            tgt_sensor_idx = sensor_perm[n_ctx:]

            # extra off-context query positions (not in sensor_perm)
            mask = torch.ones(X, dtype=torch.bool, device=device)
            mask[sensor_perm] = False
            offset_pool = torch.arange(X, device=device)[mask]
            n_extra = min(n_query_offcontext, int(offset_pool.shape[0]))
            extra_idx = offset_pool[torch.randperm(offset_pool.shape[0], device=device)[:n_extra]]
            tgt_idx_all = torch.cat([tgt_sensor_idx, extra_idx])

            # --- encode context + rich (C ∪ T_sensor) ---
            u_C = x_flat[:, ctx_idx].unsqueeze(-1)
            tokens_C = model.encoder(u_C, full_coords[ctx_idx])
            mu_C, logvar_C = model.latent_head(tokens_C)

            ct_idx = torch.cat([ctx_idx, tgt_sensor_idx])
            u_CT = x_flat[:, ct_idx].unsqueeze(-1)
            mu_CT, logvar_CT = model.encode_distribution(u_CT, full_coords[ct_idx])

            # --- sample from rich posterior, decode at targets ---
            z = model.reparam(mu_CT, logvar_CT)
            # ANP deterministic path: decoder also cross-attends the CONTEXT
            # tokens. Randomly dropped so the decoder stays usable from z
            # alone (unconditional generation).
            det = None
            if det_path and np.random.rand() >= det_drop:
                det = tokens_C
            mu_y, logvar_y = model.decode(z, full_coords[tgt_idx_all], det_tokens=det)
            y = x_flat[:, tgt_idx_all]

            # --- losses ---
            if recon_kind == "het":
                recon = het_gaussian_nll(y, mu_y, logvar_y).mean()
            else:
                # MSE forces the decoder to need z's information; switch to
                # het only once posterior health is established.
                recon = F.mse_loss(mu_y, y)

            # NP KL with free bits, per-element scale (sum over dims / d)
            kl_elem = gaussian_kl(mu_CT, logvar_CT, mu_C, logvar_C)       # (B, d)
            kl_per_dim = kl_elem.mean(dim=0)                               # (d,)
            kl_floored = kl_per_dim.clamp(min=free_bits).sum() / kl_per_dim.numel()

            # Global prior anchor: KL(q(z|C) || N(0, I)), same per-element scale.
            # Removes the two degenerate escapes of the C/CT-only KL (mu-scale
            # drift, max-variance prior) and licenses z ~ N(0,I) generation.
            anchor_elem = gaussian_kl(mu_C, logvar_C,
                                        torch.zeros_like(mu_C),
                                        torch.zeros_like(logvar_C))
            anchor_kl = anchor_elem.mean(dim=0).sum() / anchor_elem.size(1)

            # Optional VICReg-on-mu (hybrid): decorrelate/spread the mean latent.
            l_vic_std = x_flat.new_zeros(())
            l_vic_cov = x_flat.new_zeros(())
            if vicreg_mu_std > 0 or vicreg_mu_cov > 0:
                zc = mu_CT - mu_CT.mean(0)
                std = torch.sqrt(zc.var(0) + 1e-4)
                l_vic_std = F.relu(1 - std).mean()
                cov = (zc.T @ zc) / (B - 1)
                l_vic_cov = off_diagonal(cov).pow(2).sum() / zc.size(1)

            # Unified objective: INDEPENDENT second view of the same field.
            # The nested KL (C subset of CT) is the weak consistency game —
            # two independent sensor draws are the strong one (VICReg-style),
            # and the symmetric KL aligns the *posteriors* (mean AND sigma),
            # which is the NP-native generalization of view alignment.
            l_align = x_flat.new_zeros(())
            l_psim = x_flat.new_zeros(())
            l_pstd = x_flat.new_zeros(())
            l_pcov = x_flat.new_zeros(())
            if unified:
                n_2 = int(mcnt_choices[np.random.randint(len(mcnt_choices))])
                idx_2 = torch.randperm(X, device=device)[:n_2]
                mu_2, logvar_2 = model.encode_distribution(
                    x_flat[:, idx_2].unsqueeze(-1), full_coords[idx_2])
                if align_kl > 0:
                    # sigma DETACHED: alignment must not be satisfiable by
                    # widening the posteriors (KL ~ dmu^2/sigma^2 — inflating
                    # sigma is the cheap escape; the first unified run drove
                    # logvar to 86% saturation this way). With sigma detached
                    # the term aligns the means, uncertainty-weighted, and
                    # sigma stays governed by recon + anchor + nested KL.
                    lv_C_d, lv_2_d = logvar_C.detach(), logvar_2.detach()
                    sym = 0.5 * (gaussian_kl(mu_C, lv_C_d, mu_2, lv_2_d)
                                   + gaussian_kl(mu_2, lv_2_d, mu_C, lv_C_d))
                    l_align = sym.mean(dim=0).sum() / sym.size(1)
                if projector is not None:
                    xz = projector(mu_C)
                    yz = projector(mu_2)
                    l_psim = F.mse_loss(xz, yz)
                    xz = xz - xz.mean(0); yz = yz - yz.mean(0)
                    std_x = torch.sqrt(xz.var(0) + 1e-4)
                    std_y = torch.sqrt(yz.var(0) + 1e-4)
                    l_pstd = (F.relu(1 - std_x).mean() / 2
                                + F.relu(1 - std_y).mean() / 2)
                    cov_x = (xz.T @ xz) / (B - 1)
                    cov_y = (yz.T @ yz) / (B - 1)
                    n_pd = xz.shape[1]
                    l_pcov = (off_diagonal(cov_x).pow_(2).sum().div(n_pd)
                                + off_diagonal(cov_y).pow_(2).sum().div(n_pd))

            # Projector terms ride the ramp too: unanchored, the variance
            # hinge blew |mu| up to ~160 during the recon-only phase.
            loss = (recon + beta_t * kl_floored + anchor_t * anchor_kl
                      + vicreg_mu_std * l_vic_std + vicreg_mu_cov * l_vic_cov
                      + (align_kl * ramp) * l_align
                      + ramp * (proj_sim * l_psim + proj_std * l_pstd
                                  + proj_cov * l_pcov))

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            with torch.no_grad():
                lv_sat = float((logvar_C > lv_max - 0.05).float().mean())
            agg["recon"]          += float(recon) * B
            agg["kl_raw_sum"]     += float(kl_per_dim.sum()) * B
            agg["kl_max"]         += float(kl_per_dim.max()) * B
            agg["kl_active_frac"] += float((kl_per_dim > free_bits).float().mean()) * B
            agg["anchor_kl"]      += float(anchor_kl) * B
            agg["mu_abs"]         += float(mu_C.abs().mean()) * B
            agg["lv_sat_frac"]    += lv_sat * B
            agg["vic_std"]        += float(l_vic_std) * B
            agg["vic_cov"]        += float(l_vic_cov) * B
            agg["align_kl"]       += float(l_align) * B
            agg["p_sim"]          += float(l_psim) * B
            agg["p_std"]          += float(l_pstd) * B
            agg["p_cov"]          += float(l_pcov) * B
            agg["logvar_C"]       += float(logvar_C.mean())  * B
            agg["logvar_CT"]      += float(logvar_CT.mean()) * B
            agg["logvar_y"]       += float(logvar_y.mean())  * B
            agg["n"] += B

        sched.step()
        n = max(agg["n"], 1)
        elapsed = int(time.time() - t0)
        s = {k: agg[k] / n for k in agg if k != "n"}
        line = (f"ep {ep+1:3d}/{epochs}  recon={s['recon']:+.3e}  "
                 f"KL_sum={s['kl_raw_sum']:.3e}  KL_max={s['kl_max']:.2e}  "
                 f"active={s['kl_active_frac']:.2f}  anchor={s['anchor_kl']:.3e}  "
                 f"|mu|={s['mu_abs']:.2f}  lv_sat={s['lv_sat_frac']:.2f}  "
                 f"logvar(C/CT/y)={s['logvar_C']:+.2f}/{s['logvar_CT']:+.2f}/{s['logvar_y']:+.2f}  "
                 f"β_t={beta_t:.1e} a_t={anchor_t:.1e}")
        if unified:
            line += (f"  align={s['align_kl']:.3e}  "
                      f"p(sim/std/cov)={s['p_sim']:.2e}/{s['p_std']:.2e}/{s['p_cov']:.2e}")
        print(line + f"  ({elapsed}s)", flush=True)
        history.append({"epoch": ep + 1, **s, "beta_t": beta_t,
                          "anchor_t": anchor_t, "elapsed": elapsed})

        # Per-epoch fallback save (overwritten; survives a crash mid-run).
        torch.save({"method": "fae_np", "history": history, "n_par": n_par,
                      "model": model.state_dict(), "config": config,
                      "np_config": np_config, "epoch": ep + 1},
                     out_path.replace(".pt", ".latest.pt"))

    final = {"method": "fae_np", "history": history, "n_par": n_par,
               "model": model.state_dict(), "config": config,
               "np_config": np_config}
    if projector is not None:
        final["projector"] = projector.state_dict()
    torch.save(final, out_path)
    print(f"\ndone in {int(time.time() - t0)}s → {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",            required=True)
    ap.add_argument("--epochs",         type=int,   default=20)
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
    ap.add_argument("--beta",           type=float, default=1e-3)
    ap.add_argument("--free_bits",      type=float, default=0.1)
    ap.add_argument("--anchor",         type=float, default=1e-3,
                     help="weight of KL(q(z|C) || N(0,I)); 0 disables")
    ap.add_argument("--vicreg_mu_std",  type=float, default=0.0,
                     help="VICReg variance hinge on mu_CT (hybrid)")
    ap.add_argument("--vicreg_mu_cov",  type=float, default=0.0,
                     help="VICReg covariance penalty on mu_CT (hybrid)")
    ap.add_argument("--align_kl",       type=float, default=0.0,
                     help="symmetric KL between posteriors of two INDEPENDENT "
                          "views (unified objective); 0 disables")
    ap.add_argument("--proj_sim",       type=float, default=0.0,
                     help="projected VICReg similarity weight on (mu_C, mu_view2)")
    ap.add_argument("--proj_std",       type=float, default=0.0,
                     help="projected VICReg variance-hinge weight")
    ap.add_argument("--proj_cov",       type=float, default=0.0,
                     help="projected VICReg covariance weight")
    ap.add_argument("--recon_only_epochs", type=int, default=2,
                     help="pure-recon epochs before the KL/anchor ramp")
    ap.add_argument("--beta_warmup_epochs", type=int, default=4)
    ap.add_argument("--d_latent",       type=int,   default=None)
    ap.add_argument("--decoder_num_blocks", type=int, default=2)
    ap.add_argument("--recon_kind", choices=["mse", "het"], default="mse",
                     help="mse forces decoder to use z; het is the full NP NLL")
    ap.add_argument("--logvar_param", choices=["sigmoid", "clamp"], default="sigmoid")
    ap.add_argument("--det_path", action="store_true",
                     help="ANP-style deterministic path: decoder cross-attends "
                          "the context tokens alongside the z tokens")
    ap.add_argument("--det_drop", type=float, default=0.25,
                     help="probability of dropping the det path per step")
    args = ap.parse_args()
    train(args.out, args.epochs, args.batch, args.lr, args.gpu, args.workers,
            args.warmup_epochs, args.time_subsample,
            args.n_query_offcontext, args.context_ratio, not args.no_jitter_ctx,
            tuple(args.mcnt_choices),
            args.beta, args.free_bits, args.anchor,
            args.vicreg_mu_std, args.vicreg_mu_cov,
            args.align_kl, args.proj_sim, args.proj_std, args.proj_cov,
            args.recon_only_epochs, args.beta_warmup_epochs,
            args.d_latent, args.decoder_num_blocks, args.recon_kind,
            args.logvar_param, args.det_path, args.det_drop)
