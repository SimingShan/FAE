"""Functional JEPA trainer — temporal latent prediction on a sparse functional
encoder. Dynamics (predict t+h) + invariance (different sparsity per view), no
reconstruction. Instrumented for collapse from step one (PR + probe + pred-var).

  python scripts/train_fjepa.py --loss vicreg --horizon 1 --tag v1
  python scripts/train_fjepa.py --loss ema    --horizon 1 --tag v1
  python scripts/train_fjepa.py --loss raw    --horizon 1 --tag v1
"""
from __future__ import annotations
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.fjepa import FunctionalJEPA
from src.data.well2d import ShearFlowClipDataset, make_coords_2d, fields_to_tokens
from src.metrics import lin_probe

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_projector(in_dim, d):
    f = [in_dim, d, d, d]; L = []
    for i in range(len(f) - 2):
        L += [nn.Linear(f[i], f[i + 1]), nn.BatchNorm1d(f[i + 1]), nn.ReLU(True)]
    L.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*L)


def off_diag(x):
    n, m = x.shape
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(proj, pred_rep, target_rep, enc_rep, B, sim, std, cov):
    # invariance: predicted future ~ target. variance/cov: on the ENCODER rep
    # (`enc_rep`=La) — the thing we probe & measure PR on, so anti-collapse and
    # diagnostic are aligned (not on the discardable predictor output).
    zp, zt = proj(pred_rep), proj(target_rep)
    l_sim = F.mse_loss(zp, zt)
    ze = proj(enc_rep); ze = ze - ze.mean(0)
    l_std = F.relu(1 - torch.sqrt(ze.var(0) + 1e-4)).mean()
    d = ze.shape[1]
    l_cov = off_diag((ze.T @ ze) / (B - 1)).pow_(2).sum() / d
    return sim * l_sim + std * l_std + cov * l_cov, (float(l_sim), float(l_std), float(l_cov))


def vicreg_loss_tokens(proj, pred_tok, target_tok, enc_tok, sim, std, cov):
    # TOKEN-level: match prediction to target PER TOKEN (no pooling) so the predictor
    # must learn how each latent slot evolves — the fix for the time-flat pooled target.
    # variance/cov over the (B*M) token population (à la VICRegL).
    B, M, D = enc_tok.shape; N = B * M
    zp, zt = proj(pred_tok.reshape(N, D)), proj(target_tok.reshape(N, D))
    l_sim = F.mse_loss(zp, zt)
    ze = proj(enc_tok.reshape(N, D)); ze = ze - ze.mean(0)
    l_std = F.relu(1 - torch.sqrt(ze.var(0) + 1e-4)).mean()
    d = ze.shape[1]
    l_cov = off_diag((ze.T @ ze) / (N - 1)).pow_(2).sum() / d
    return sim * l_sim + std * l_std + cov * l_cov, (float(l_sim), float(l_std), float(l_cov))


def temporal_var_loss(ra, rb, margin):
    """Anti-collapse on the TIME axis (the one VICReg's batch-variance ignores).
    Force L(t) != L(t+Δ): per dim, the temporal step's energy must be >= `margin`
    times the cross-sample variance. DC-invariant (it's a difference), scale-free
    (ratio). With the predictor invariance still pulling pred(La)->Lb, the only
    joint solution is predictable temporal variation = real dynamics, not a frozen
    latent (identity) and not unpredictable noise."""
    d = ra - rb                                  # temporal step, rb is sg target
    tvar = d.pow(2).mean(0)                       # per-dim temporal-step energy
    svar = ra.detach().var(0) + 1e-6             # per-dim cross-sample variance
    return F.relu(margin - tvar / svar).mean()


def participation_ratio(Z):
    Z = Z - Z.mean(0)
    e = np.clip(np.linalg.eigvalsh(Z.T @ Z / max(len(Z) - 1, 1)), 0, None)
    return float(e.sum() ** 2 / max((e ** 2).sum(), 1e-30))


@torch.no_grad()
def embed(model, ds, coords, idx, batch=128):
    model.eval(); Z, Y = [], []
    for clip, y in DataLoader(ds, batch_size=batch):
        fa = clip[:, :, 0].to(DEVICE)                        # probe the encoder on frame 0
        tok = model.encode(fields_to_tokens(fa, idx), coords[idx])
        Z.append(model.represent(tok).cpu().numpy()); Y.append(y.numpy())
    model.train()
    return np.concatenate(Z), np.concatenate(Y)


def probe2(Ztr, Ytr, Zva, Yva):
    out = {}
    for j, nm in enumerate(["logRe", "Sc"]):
        yt, yv = Ytr[:, j], Yva[:, j]; m, s = yt.mean(), yt.std() + 1e-8
        out[nm] = lin_probe(Ztr, (yt - m) / s, Zva, (yv - m) / s)
    return out


@torch.no_grad()
def forecast_diag(model, ds, coords, idx, dt_max, batch=128, nb=4):
    """Does the predictor beat persistence? cos(pred,Lb) vs cos(La,Lb) on held-out
    (t, t+Δ) pairs. If persist≈1 the pooled rep is ~time-invariant (=> prediction
    necessarily collapses to static, explaining static≈temporal). If forecast>persist
    the predictor learns real dynamics — so a flat probe is genuine, not a wiring bug."""
    model.eval(); fc = ps = 0.0; n = 0; RA = []; RB = []
    for bi, (clip, _y) in enumerate(DataLoader(ds, batch_size=batch)):
        if bi >= nb: break
        clip = clip.to(DEVICE); B = clip.size(0); K = clip.size(2)
        bidx = torch.arange(B, device=DEVICE)
        delta = torch.randint(1, dt_max + 1, (B,), device=DEVICE)
        ts = (torch.rand(B, device=DEVICE) * (K - delta).float()).long()
        fa = clip[bidx, :, ts]; fb = clip[bidx, :, ts + delta]
        La = model.encode(fields_to_tokens(fa, idx), coords[idx])
        Lb = model.encode(fields_to_tokens(fb, idx), coords[idx])
        pred = model.predict(La, delta.float() / dt_max)
        ra, rb, rp = model.represent(La), model.represent(Lb), model.represent(pred)
        fc += float(F.cosine_similarity(rp, rb, dim=-1).sum())
        ps += float(F.cosine_similarity(ra, rb, dim=-1).sum())
        RA.append(ra); RB.append(rb); n += B
    RA = torch.cat(RA); RB = torch.cat(RB)
    cc = float(F.cosine_similarity(RA - RA.mean(0), RB - RA.mean(0), dim=-1).mean())  # DC-removed
    vr = float(((RA - RB).pow(2).mean(0) / (RA.var(0) + 1e-6)).mean())                # temporal/sample var
    model.train()
    return fc / n, ps / n, cc, vr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", choices=["raw", "ema", "vicreg"], default="vicreg")
    ap.add_argument("--dt_max", type=int, default=8, help="max gap Δ in frame_stride units (1 = fixed horizon)")
    ap.add_argument("--clip_len", type=int, default=12, help="strided frames/clip (clamped to >= dt_max+1)")
    ap.add_argument("--frame_stride", type=int, default=4)
    ap.add_argument("--mcnt", type=int, nargs="+", default=[256, 512], help="sparse sensor counts/view")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--pred_depth", type=int, default=2)
    ap.add_argument("--pred_type", choices=["attn", "mlp"], default="attn",
                    help="attn: cross-token attention predictor; mlp: per-token MLP (weaker, forces encoder)")
    ap.add_argument("--ema_decay", type=float, default=0.996)
    ap.add_argument("--proj_dim", type=int, default=4096)
    ap.add_argument("--sim", type=float, default=25.0)
    ap.add_argument("--std", type=float, default=25.0)
    ap.add_argument("--cov", type=float, default=1.0)
    ap.add_argument("--n_seed", type=int, default=24)
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--static", action="store_true",
                    help="apple-to-apple control: no-recon VICReg invariance between two sparse "
                         "views of the SAME frame (no temporal prediction, no predictor)")
    ap.add_argument("--token_pred", action="store_true",
                    help="match prediction to target PER-TOKEN (not pooled). Fix for the time-flat "
                         "pooled target — forces the predictor to learn per-latent-slot dynamics")
    ap.add_argument("--temporal_var", type=float, default=0.0,
                    help="weight on TIME-axis anti-collapse (force L(t)!=L(t+Δ)); 0=off. The term "
                         "VICReg lacks — guards the time axis, not just the sample axis.")
    ap.add_argument("--tmargin", type=float, default=0.5,
                    help="target temporal/sample var ratio for the temporal anti-collapse hinge")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--save", action="store_true",
                    help="persist the encoder checkpoint (OFF by default — sweeps write nothing, "
                         "only final/confirm runs pass --save, to avoid 26M/run storage blow-up)")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    R = args.resolution; NPIX = R * R
    clip_len = max(args.clip_len, args.dt_max + 1)
    print(f"=== FunctionalJEPA [{args.tag}] loss={args.loss} dt_max={args.dt_max} "
          f"clip_len={clip_len} stride={args.frame_stride} res={R} batch={args.batch} "
          f"mcnt={args.mcnt} ===", flush=True)

    tr = ShearFlowClipDataset("train", n_seed=args.n_seed, frame_stride=args.frame_stride,
                              clip_len=clip_len, side=R)
    va = ShearFlowClipDataset("valid", n_seed=8, frame_stride=args.frame_stride,
                              clip_len=clip_len, side=R, stats=tr.stats)
    print(f"  train clips {len(tr)}  valid {len(va)}", flush=True)
    loader = DataLoader(tr, batch_size=args.batch, shuffle=True, drop_last=True,
                        num_workers=4, pin_memory=True)
    coords = make_coords_2d(n_side=R, device=DEVICE)

    model = FunctionalJEPA(coord_dim=2, in_chans=4, pred_depth=args.pred_depth,
                           pred_type=args.pred_type, use_ema=(args.loss == "ema"),
                           ema_decay=args.ema_decay).to(DEVICE)
    proj = make_projector(model.emb_dim, args.proj_dim).to(DEVICE) if args.loss == "vicreg" else None
    params = list(model.parameters()) + (list(proj.parameters()) if proj is not None else [])
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    g0 = torch.Generator(device=DEVICE).manual_seed(0)
    probe_idx = torch.randperm(NPIX, generator=g0, device=DEVICE)[:1024]
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); run = {"loss": 0, "n": 0}
        for clip, _y in loader:
            clip = clip.to(DEVICE, non_blocking=True); B = clip.size(0); K = clip.size(2)
            bidx = torch.arange(B, device=DEVICE)
            if args.static:                                         # CONTROL: invariance only,
                ts = torch.randint(0, K, (B,), device=DEVICE)       # two sparse views of the SAME
                fa = clip[bidx, :, ts]; fb = fa; dt_norm = None     # frame, no prediction
            else:                                                   # temporal prediction (dynamics)
                delta = torch.randint(1, args.dt_max + 1, (B,), device=DEVICE)
                ts = (torch.rand(B, device=DEVICE) * (K - delta).float()).long()
                fa = clip[bidx, :, ts]; fb = clip[bidx, :, ts + delta]
                dt_norm = delta.float() / args.dt_max
            nA, nB = int(np.random.choice(args.mcnt)), int(np.random.choice(args.mcnt))
            iA = torch.randperm(NPIX, device=DEVICE)[:nA]
            iB = torch.randperm(NPIX, device=DEVICE)[:nB]            # different sparsity/placement
            La = model.encode(fields_to_tokens(fa, iA), coords[iA])
            Lb = model.encode_target(fields_to_tokens(fb, iB), coords[iB])   # no grad (sg/ema)
            pred = La if args.static else model.predict(La, dt_norm)
            if args.loss == "vicreg" and args.token_pred and not args.static:
                loss, _ = vicreg_loss_tokens(proj, pred, Lb, La, args.sim, args.std, args.cov)
            elif args.loss == "vicreg":
                loss, _ = vicreg_loss(proj, model.represent(pred), model.represent(Lb),
                                      model.represent(La), B, args.sim, args.std, args.cov)
            else:  # raw / ema: cosine match (predictor + sg/ema is the anti-collapse)
                loss = -F.cosine_similarity(pred, Lb, dim=-1).mean()
            if args.temporal_var > 0 and not args.static:           # guard the TIME axis
                loss = loss + args.temporal_var * temporal_var_loss(
                    model.represent(La), model.represent(Lb), args.tmargin)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            model.update_ema()
            run["loss"] += float(loss) * B; run["n"] += B
        sched.step()
        if (ep + 1) % args.eval_every == 0 or ep == 0:
            Ztr, Ytr = embed(model, tr, coords, probe_idx)
            Zva, Yva = embed(model, va, coords, probe_idx)
            pr = participation_ratio(Ztr); pb = probe2(Ztr, Ytr, Zva, Yva)
            fcstr = ""
            if not args.static:
                fc, ps, cc, vr = forecast_diag(model, va, coords, probe_idx, args.dt_max)
                fcstr = f"  fcast={fc:+.3f} ccos={cc:+.3f} vr={vr:.3f}"
            print(f"ep {ep+1:3d}/{args.epochs}  loss={float(run['loss']/run['n']):+.4f}  "
                  f"PR={float(pr):.1f}  probe logRe={float(pb['logRe']):+.3f} "
                  f"Sc={float(pb['Sc']):+.3f}{fcstr}  ({time.time()-t0:.0f}s)", flush=True)

    if args.save:
        out = f"results/checkpoints/g1/fjepa_{args.loss}_dt{args.dt_max}_{args.tag}.pt"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        torch.save({"encoder": model.encoder.state_dict(),
                    "predictor": model.predictor.state_dict(),   # latent flow: kept on disk, not used at eval
                    "stats": tr.stats, "train_args": vars(args)}, out)
        print(f"  saved {out}", flush=True)
    else:
        print("  (not saved — pass --save to persist)", flush=True)


if __name__ == "__main__":
    main()
