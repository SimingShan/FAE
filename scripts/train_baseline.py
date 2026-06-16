"""Matched baseline trainer for the single-frame comparison: AE / MAE / I-JEPA.

One harness, one protocol — same dataset (ShearFlowSnapshotDataset, 11,424
snapshots), same --batch and --epochs, same frozen-encoder ridge-probe eval as
train_fae_shear.py — so AE / MAE / I-JEPA / FAE differ only in the SSL objective
(Hard-Rule #4 fairness). Per-method LEARNING RATE is kept method-appropriate
(matching batch, not lr, is the fair choice).

  python scripts/train_baseline.py --method mae   --epochs 100 --batch 256 --amp
  python scripts/train_baseline.py --method ae    --epochs 100 --batch 256 --amp
  python scripts/train_baseline.py --method ijepa --epochs 100 --batch 256 --amp
  python scripts/eval_linear_probe.py --method mae --ckpt results/checkpoints/g1/mae_shear_v1.pt

Frozen probe must beat the random-projection floor (logRe 0.25 / Sc 0.16),
NOT zero. PR is the collapse guard.
"""
from __future__ import annotations
import sys, os, time, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.well2d import ShearFlowSnapshotDataset
from src.metrics import lin_probe

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PARAMS = ["logRe", "Sc"]

# Method-appropriate optimizer defaults (matched batch/epochs, NOT matched lr).
CFG = {
    "ae":       dict(lr=1.5e-4, wd=0.05, betas=(0.9, 0.95)),   # Kaiming MAE recipe
    "mae":      dict(lr=1.5e-4, wd=0.05, betas=(0.9, 0.95)),
    "videomae": dict(lr=1.5e-4, wd=0.05, betas=(0.9, 0.95)),   # temporal pixel baseline
    "ijepa":    dict(lr=1.0e-3, wd=0.04, betas=(0.9, 0.999)),  # I-JEPA recipe
    "stjepa":   dict(lr=1.0e-3, wd=0.04, betas=(0.9, 0.999)),  # spatio-temporal JEPA (tubelet)
}


def participation_ratio(Z):
    Z = Z - Z.mean(0)
    e = np.clip(np.linalg.eigvalsh(Z.T @ Z / max(len(Z) - 1, 1)), 0, None)
    return float(e.sum() ** 2 / max((e ** 2).sum(), 1e-30))


@torch.no_grad()
def embed(model, ds, batch=128):
    model.eval()
    Z, Y = [], []
    for f, y in DataLoader(ds, batch_size=batch):
        Z.append(model.encode(f.to(DEVICE)).float().cpu().numpy())
        Y.append(y.numpy())
    model.train()
    return np.concatenate(Z), np.concatenate(Y)


def probe2(Ztr, Ytr, Zva, Yva):
    out = {}
    for j, nm in enumerate(PARAMS):
        ytr, yva = Ytr[:, j], Yva[:, j]
        ym, ys = ytr.mean(), ytr.std() + 1e-8
        out[nm] = lin_probe(Ztr, (ytr - ym) / ys, Zva, (yva - ym) / ys)
    return out


def build_model(method, resolution=224, n_frames=1, tubelet=2):
    if method in ("ae", "mae"):
        from benchmarks.mae.mae import mae_physics
        return mae_physics(img_size=resolution).to(DEVICE)
    if method == "videomae":
        from benchmarks.mae.videomae import videomae_physics
        return videomae_physics(img_size=resolution, num_frames=n_frames).to(DEVICE)
    if method == "stjepa":
        from benchmarks.jepa.stjepa import stjepa_physics
        return stjepa_physics(img_size=resolution, num_frames=n_frames, tubelet=tubelet).to(DEVICE)
    from benchmarks.jepa.ijepa2d import ijepa2d_physics
    return ijepa2d_physics(img_size=resolution).to(DEVICE)


def loss_step(method, model, x, args):
    """One forward -> scalar loss (no backward). JEPA EMA handled by caller."""
    if method == "ae":
        return model(x, mask_ratio=0.0)[0]
    if method in ("mae", "videomae"):
        return model(x, mask_ratio=args.mask_ratio)[0]
    from benchmarks.jepa.ijepa2d import sample_masks                     # ijepa / stjepa
    P = model.num_patches                                               # scale masks to patch count
    n_ctx = max(8, int(args.ctx_frac * P)); n_tgt = max(4, int(args.tgt_frac * P))
    ctx, tgt = sample_masks(x.size(0), P, n_ctx, n_tgt, DEVICE)
    pred, h = model(x, ctx, tgt)
    return F.smooth_l1_loss(pred, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["ae", "mae", "ijepa", "videomae", "stjepa"], required=True)
    ap.add_argument("--n_frames", type=int, default=1, help="clip length for videomae/stjepa")
    ap.add_argument("--tubelet", type=int, default=2, help="temporal tubelet size (videomae/stjepa)")
    ap.add_argument("--ctx_frac", type=float, default=0.2, help="JEPA context patches as frac of P")
    ap.add_argument("--tgt_frac", type=float, default=0.06, help="JEPA target patches as frac of P")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=None, help="override method default")
    ap.add_argument("--wd", type=float, default=None)
    ap.add_argument("--n_seed", type=int, default=24)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast (A100-friendly)")
    ap.add_argument("--mask_ratio", type=float, default=0.75, help="MAE only")
    ap.add_argument("--n_ctx", type=int, default=40, help="I-JEPA context patches")
    ap.add_argument("--n_tgt", type=int, default=12, help="I-JEPA target patches")
    ap.add_argument("--ema_start", type=float, default=0.996)
    ap.add_argument("--ema_end", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--resolution", type=int, default=224, help="square resize side (paper: 224)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    cfg = CFG[args.method]
    lr = args.lr if args.lr is not None else cfg["lr"]
    wd = args.wd if args.wd is not None else cfg["wd"]
    print(f"=== {args.method.upper()} shear_flow [{args.tag}]  res={args.resolution} batch={args.batch} "
          f"epochs={args.epochs} lr={lr:.1e} wd={wd} amp={args.amp} ===", flush=True)

    if args.method in ("videomae", "stjepa"):
        from src.data.well2d import ShearFlowWindowDataset
        tr = ShearFlowWindowDataset("train", n_seed=args.n_seed, n_frames=args.n_frames, side=args.resolution)
        va = ShearFlowWindowDataset("valid", n_seed=8, n_frames=args.n_frames, side=args.resolution, stats=tr.stats)
    else:
        tr = ShearFlowSnapshotDataset("train", n_seed=args.n_seed, frame_stride=12, side=args.resolution)
        va = ShearFlowSnapshotDataset("valid", n_seed=8, frame_stride=12, side=args.resolution, stats=tr.stats)
    Ytr_all = np.stack([tr.logRe, tr.Sc], 1)
    Yva_all = np.stack([va.logRe, va.Sc], 1)
    print(f"  train {len(tr)}  valid {len(va)}", flush=True)
    loader = DataLoader(tr, batch_size=args.batch, shuffle=True, drop_last=True,
                        num_workers=args.workers, pin_memory=True)

    model = build_model(args.method, args.resolution, args.n_frames, args.tubelet)
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  params(trainable)={npar:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=cfg["betas"])
    warm = max(1, int(0.05 * args.epochs))

    def lr_lambda(ep):
        if ep < warm:
            return (ep + 1) / warm
        p = (ep - warm) / max(1, args.epochs - warm)
        return 0.5 * (1 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    total_steps = args.epochs * len(loader)
    gstep, t0 = 0, time.time()
    for ep in range(args.epochs):
        model.train()
        run, n = 0.0, 0
        for f, _y in loader:
            x = f.to(DEVICE, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                loss = loss_step(args.method, model, x, args)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            if args.method in ("ijepa", "stjepa"):
                tau = args.ema_start + (args.ema_end - args.ema_start) * (gstep / max(1, total_steps - 1))
                model.update_target(tau)
            run += float(loss) * x.size(0); n += x.size(0); gstep += 1
        sched.step()
        if (ep + 1) % args.eval_every == 0 or ep == 0:
            Ztr, Ytr = embed(model, tr); Zva, Yva = embed(model, va)
            pr = participation_ratio(Ztr); pb = probe2(Ztr, Ytr, Zva, Yva)
            print(f"ep {ep+1:3d}/{args.epochs}  loss={run/n:.4e}  PR={pr:.1f}  "
                  f"probe logRe={pb['logRe']:+.3f} Sc={pb['Sc']:+.3f}  "
                  f"(floor 0.25/0.16)  ({time.time()-t0:.0f}s)", flush=True)

    out = f"results/checkpoints/g1/{args.method}_shear_{args.tag}.pt"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"model": model.state_dict(), "stats": tr.stats, "train_args": vars(args)}, out)
    Ztr, Ytr = embed(model, tr); Zva, Yva = embed(model, va)
    pr = participation_ratio(Ztr); pb = probe2(Ztr, Ytr, Zva, Yva)
    print(f"\n=== [{args.method}/{args.tag}] PR={pr:.2f}  "
          f"probe logRe={pb['logRe']:+.3f} Sc={pb['Sc']:+.3f}  "
          f"(beat floor 0.25/0.16) ===\n  saved {out}", flush=True)


if __name__ == "__main__":
    main()
