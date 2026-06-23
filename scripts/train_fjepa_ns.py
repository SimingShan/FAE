"""FunctionalJEPA on NS-2D buoyancy — the no-reconstruction, latent x_t -> x_{t+Δ} VICReg siamese
method, wired onto the NS conditional-generation benchmark (was shear-only; see arxiv/pre_repa_pivot).

Dynamics (predict t+Δ) + invariance (different sparsity per view), NO reconstruction. Encoder is the
product (128 latents); the Δt-conditioned predictor is a discardable latent flow. In-loop buoyancy
linear probe + PR as health signals; authoritative floor/head-to-head via scripts/eval_ns_probe.py.

  python scripts/train_fjepa_ns.py --loss vicreg --dt_max 4 --tag ns_v1 --save
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from src.models.fjepa import FunctionalJEPA
from src.data.ns import NSDataset
from src.data.well2d import make_coords_2d, fields_to_tokens
from src.metrics import lin_probe
from src.utils import set_seed, seed_worker
from src.utils.seed import torch_generator

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
    zp, zt = proj(pred_rep), proj(target_rep)
    l_sim = F.mse_loss(zp, zt)
    ze = proj(enc_rep); ze = ze - ze.mean(0)
    l_std = F.relu(1 - torch.sqrt(ze.var(0) + 1e-4)).mean()
    d = ze.shape[1]
    l_cov = off_diag((ze.T @ ze) / (B - 1)).pow_(2).sum() / d
    return sim * l_sim + std * l_std + cov * l_cov, (float(l_sim), float(l_std), float(l_cov))


def temporal_var_loss(ra, rb, margin):
    d = ra - rb
    tvar = d.pow(2).mean(0)
    svar = ra.detach().var(0) + 1e-6
    return F.relu(margin - tvar / svar).mean()


def participation_ratio(Z):
    Z = Z - Z.mean(0)
    e = np.clip(np.linalg.eigvalsh(Z.T @ Z / max(len(Z) - 1, 1)), 0, None)
    return float(e.sum() ** 2 / max((e ** 2).sum(), 1e-30))


@torch.no_grad()
def embed(model, ds, coords, idx, batch=128):
    model.eval(); Z, Y = [], []
    for clip, y in DataLoader(ds, batch_size=batch):
        fa = clip[:, :, 0].to(DEVICE)                        # probe encoder on frame 0
        tok = model.encode(fields_to_tokens(fa, idx), coords[idx])
        Z.append(model.represent(tok).cpu().numpy()); Y.append(y.numpy())
    model.train()
    return np.concatenate(Z), np.concatenate(Y).ravel()


def probe_buoy(Ztr, Ytr, Zva, Yva):
    m, s = Ytr.mean(), Ytr.std() + 1e-8
    return lin_probe(Ztr, (Ytr - m) / s, Zva, (Yva - m) / s)


def channel_floor(ds):                                       # trivial baseline: frame-0 channel mean+std
    X, Y = [], []
    for clip, y in DataLoader(ds, batch_size=256):
        f0 = clip[:, :, 0]
        X.append(torch.cat([f0.mean((2, 3)), f0.std((2, 3))], -1).numpy()); Y.append(y.numpy())
    return np.concatenate(X), np.concatenate(Y).ravel()


@torch.no_grad()
def forecast_diag(model, ds, coords, idx, dt_max, batch=128, nb=4):
    model.eval(); fc = ps = 0.0; n = 0
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
        ps += float(F.cosine_similarity(ra, rb, dim=-1).sum()); n += B
    model.train()
    return fc / n, ps / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", choices=["raw", "ema", "vicreg"], default="vicreg")
    ap.add_argument("--dt_max", type=int, default=4, help="max gap Δ in frame_stride units")
    ap.add_argument("--clip_len", type=int, default=8); ap.add_argument("--frame_stride", type=int, default=4)
    ap.add_argument("--mcnt", type=int, nargs="+", default=[256, 512])
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--pred_depth", type=int, default=2)
    ap.add_argument("--pred_type", choices=["attn", "mlp"], default="attn")
    ap.add_argument("--ema_decay", type=float, default=0.996); ap.add_argument("--proj_dim", type=int, default=4096)
    ap.add_argument("--sim", type=float, default=25.0); ap.add_argument("--std", type=float, default=25.0)
    ap.add_argument("--cov", type=float, default=1.0)
    ap.add_argument("--n_traj", type=int, default=12); ap.add_argument("--resolution", type=int, default=64)
    ap.add_argument("--emb_dim", type=int, default=320); ap.add_argument("--num_latents", type=int, default=128)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--static", action="store_true", help="control: invariance only, two sparse views of SAME frame")
    ap.add_argument("--temporal_var", type=float, default=0.0); ap.add_argument("--tmargin", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--dataset", default="ns")
    ap.add_argument("--tag", default="ns_v1")
    ap.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args(); set_seed(args.seed)
    R = args.resolution; NPIX = R * R; C = 3
    clip_len = max(args.clip_len, args.dt_max + 1)
    print(f"=== FunctionalJEPA-NS [{args.tag}] loss={args.loss} dt_max={args.dt_max} clip_len={clip_len} "
          f"stride={args.frame_stride} res={R} mcnt={args.mcnt} static={args.static} ===", flush=True)

    tr = NSDataset("train", side=R, mode="clip", clip_len=clip_len, frame_stride=args.frame_stride, n_traj=args.n_traj)
    va = NSDataset("valid", side=R, mode="clip", clip_len=clip_len, frame_stride=args.frame_stride, n_traj=8, stats=tr.stats)
    print(f"  train clips {len(tr)}  valid {len(va)}", flush=True)
    loader = DataLoader(tr, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=4,
                        pin_memory=True, worker_init_fn=seed_worker, generator=torch_generator(args.seed))
    coords = make_coords_2d(n_side=R, device=DEVICE)

    model = FunctionalJEPA(emb_dim=args.emb_dim, num_latents=args.num_latents, coord_dim=2, in_chans=C,
                           pred_depth=args.pred_depth, pred_type=args.pred_type,
                           use_ema=(args.loss == "ema"), ema_decay=args.ema_decay).to(DEVICE)
    proj = make_projector(model.emb_dim, args.proj_dim).to(DEVICE) if args.loss == "vicreg" else None
    params = list(model.parameters()) + (list(proj.parameters()) if proj is not None else [])
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    g0 = torch.Generator(device=DEVICE).manual_seed(0)
    probe_idx = torch.randperm(NPIX, generator=g0, device=DEVICE)[:1024]
    # floor (computed once)
    Xf_tr, yf_tr = channel_floor(tr); Xf_va, yf_va = channel_floor(va)
    floor_r2 = probe_buoy(Xf_tr, yf_tr, Xf_va, yf_va)
    print(f"  FLOOR (channel mean+std) buoyancy R2 = {floor_r2:+.3f}", flush=True)

    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); run = {"loss": 0.0, "n": 0}
        for clip, _y in loader:
            clip = clip.to(DEVICE, non_blocking=True); B = clip.size(0); K = clip.size(2)
            bidx = torch.arange(B, device=DEVICE)
            if args.static:
                ts = torch.randint(0, K, (B,), device=DEVICE); fa = clip[bidx, :, ts]; fb = fa; dt_norm = None
            else:
                delta = torch.randint(1, args.dt_max + 1, (B,), device=DEVICE)
                ts = (torch.rand(B, device=DEVICE) * (K - delta).float()).long()
                fa = clip[bidx, :, ts]; fb = clip[bidx, :, ts + delta]; dt_norm = delta.float() / args.dt_max
            nA, nB = int(np.random.choice(args.mcnt)), int(np.random.choice(args.mcnt))
            iA = torch.randperm(NPIX, device=DEVICE)[:nA]; iB = torch.randperm(NPIX, device=DEVICE)[:nB]
            La = model.encode(fields_to_tokens(fa, iA), coords[iA])
            Lb = model.encode_target(fields_to_tokens(fb, iB), coords[iB])
            pred = La if args.static else model.predict(La, dt_norm)
            if args.loss == "vicreg":
                loss, _ = vicreg_loss(proj, model.represent(pred), model.represent(Lb),
                                      model.represent(La), B, args.sim, args.std, args.cov)
            else:
                loss = -F.cosine_similarity(pred, Lb, dim=-1).mean()
            if args.temporal_var > 0 and not args.static:
                loss = loss + args.temporal_var * temporal_var_loss(model.represent(La), model.represent(Lb), args.tmargin)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 1.0); opt.step(); model.update_ema()
            run["loss"] += float(loss) * B; run["n"] += B
        sched.step()
        if (ep + 1) % args.eval_every == 0 or ep == 0:
            Ztr, Ytr = embed(model, tr, coords, probe_idx); Zva, Yva = embed(model, va, coords, probe_idx)
            pr = participation_ratio(Ztr); pb = probe_buoy(Ztr, Ytr, Zva, Yva)
            fcstr = ""
            if not args.static:
                fc, ps = forecast_diag(model, va, coords, probe_idx, args.dt_max)
                fcstr = f"  fcast={fc:+.3f} persist={ps:+.3f}"
            print(f"ep {ep+1:3d}/{args.epochs}  loss={run['loss']/run['n']:+.4f}  PR={pr:.1f}  "
                  f"buoy R2={pb:+.3f} (floor {floor_r2:+.3f}){fcstr}  ({time.time()-t0:.0f}s)", flush=True)

    if args.save:
        out = f"results/checkpoints/g1/fjepa_{args.loss}_ns_{args.tag}.pt"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        torch.save({"model": model.state_dict(), "encoder": model.encoder.state_dict(),
                    "predictor": model.predictor.state_dict(), "stats": tr.stats, "train_args": vars(args)}, out)
        print(f"  saved {out}", flush=True)


if __name__ == "__main__":
    main()
