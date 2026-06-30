"""Matched single-frame baseline trainer: MAE / I-JEPA. Same dataset/batch/epochs/frozen-probe
as train_fae (fairness rule #4); per-method lr kept method-appropriate. Encoder is the product;
frozen encoder -> ridge probe + participation ratio (collapse guard).
"""
import sys, os, time, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from src.data.well2d import ShearFlowSnapshotDataset
from src.metrics import lin_probe
from benchmarks import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PARAMS = ["logRe", "Sc"]
CFG = {"mae": dict(lr=1.5e-4, wd=0.05, betas=(0.9, 0.95)),       # Kaiming MAE recipe
       "ijepa": dict(lr=1.0e-3, wd=0.04, betas=(0.9, 0.999)),    # I-JEPA recipe
       "dino": dict(lr=5.0e-4, wd=0.04, betas=(0.9, 0.999))}     # DINO recipe (single-GPU)


def participation_ratio(Z):
    Z = Z - Z.mean(0); e = np.clip(np.linalg.eigvalsh(Z.T @ Z / max(len(Z) - 1, 1)), 0, None)
    return float(e.sum() ** 2 / max((e ** 2).sum(), 1e-30))


@torch.no_grad()
def embed(model, ds, batch=128):
    model.eval(); Z, Y = [], []
    for f, y in DataLoader(ds, batch_size=batch):
        Z.append(model.encode(f.to(DEVICE)).float().cpu().numpy()); Y.append(y.numpy())
    model.train()
    return np.concatenate(Z), np.concatenate(Y)


def probe2(Ztr, Ytr, Zva, Yva):
    out = {}
    for j, nm in enumerate(PARAMS):
        yt, yv = Ytr[:, j], Yva[:, j]; m, s = yt.mean(), yt.std() + 1e-8
        out[nm] = lin_probe(Ztr, (yt - m) / s, Zva, (yv - m) / s)
    return out


def loss_step(method, model, x, args):
    if method == "mae":
        return model(x, mask_ratio=args.mask_ratio)[0]
    if method == "dino":
        return model.dino_step(x)                                   # multi-crop self-distillation (center update inside)
    pe = model.encoder.patch_embed                                  # faithful I-JEPA block masking (Assran et al.)
    from benchmarks.jepa.ijepa2d import sample_block_masks
    ctx, tgt = sample_block_masks(x.size(0), pe.grid[0], pe.grid[1], device=DEVICE)
    pred, h = model(x, ctx, tgt)
    return F.smooth_l1_loss(pred, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["mae", "ijepa", "dino"], required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=None, help="override method default")
    ap.add_argument("--wd", type=float, default=None)
    ap.add_argument("--betas", type=float, nargs=2, default=None)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--n_seed", type=int, default=24)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--mask_ratio", type=float, default=0.75, help="MAE")
    ap.add_argument("--ema_start", type=float, default=0.996, help="I-JEPA / DINO target EMA")
    ap.add_argument("--ema_end", type=float, default=1.0)
    ap.add_argument("--teacher_temp_start", type=float, default=0.04, help="DINO teacher temp (warmup)")
    ap.add_argument("--teacher_temp_end", type=float, default=0.07)
    ap.add_argument("--teacher_temp_warmup_frac", type=float, default=0.3, help="DINO teacher-temp ramp")
    ap.add_argument("--freeze_last_epochs", type=int, default=1, help="DINO: freeze prototype layer first N epochs")
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--res_h", type=int, default=None, help="rectangular height (shear native 128x256)")
    ap.add_argument("--res_w", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--ckpt_out", default=None)
    ap.add_argument("--dataset", choices=["shear", "flowbench", "ns", "typhoon", "sw", "rbc"], default="shear")
    ap.add_argument("--in_chans", type=int, default=None)
    ap.add_argument("--norm_pix", action="store_true", help="MAE per-patch normalized target")
    ap.add_argument("--embed_dim", type=int, default=None)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--patch_size", type=int, default=None)
    ap.add_argument("--num_heads", type=int, default=None)
    ap.add_argument("--n_traj", type=int, default=12, help="NS trajectories/file")
    ap.add_argument("--frame_stride", type=int, default=4)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    cfg = CFG[args.method]
    lr = args.lr if args.lr is not None else cfg["lr"]
    wd = args.wd if args.wd is not None else cfg["wd"]
    betas = tuple(args.betas) if args.betas is not None else cfg["betas"]
    print(f"=== {args.method.upper()} [{args.tag}]  res={args.resolution} batch={args.batch} "
          f"epochs={args.epochs} lr={lr:.1e} wd={wd} amp={args.amp} ===", flush=True)

    import json
    REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    meta = json.load(open(os.path.join(REPO, "data", args.dataset, "meta.json")))     # preprocessed spec
    in_chans = meta["C"]; IMG = (meta["H"], meta["W"]) if meta["H"] != meta["W"] else meta["H"]
    PARAMS[:] = meta["label_names"]
    from src.data.preprocessed import PDEDataset
    tr = PDEDataset(args.dataset, "train", mode="single")
    va = PDEDataset(args.dataset, "test", mode="single")
    print(f"  dataset={args.dataset} in_chans={in_chans} res={IMG}  train {len(tr)} test {len(va)}", flush=True)
    loader = DataLoader(tr, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=args.workers, pin_memory=True)

    model = build_model(args.method, IMG, in_chans, args.norm_pix, embed_dim=args.embed_dim, depth=args.depth, patch_size=args.patch_size, num_heads=args.num_heads)
    print(f"  params(trainable)={sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)
    warm = max(1, int(args.warmup_frac * args.epochs))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda ep: (ep + 1) / warm if ep < warm else 0.5 * (1 + math.cos(math.pi * (ep - warm) / max(1, args.epochs - warm))))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    total, gstep, t0 = args.epochs * len(loader), 0, time.time()
    for ep in range(args.epochs):
        te = time.time(); model.train(); run, n = 0.0, 0
        if args.method == "dino":                                    # teacher-temp warmup (0.04 -> 0.07)
            tw = max(1, int(args.teacher_temp_warmup_frac * args.epochs))
            model.teacher_temp = args.teacher_temp_start + (args.teacher_temp_end - args.teacher_temp_start) * min(1.0, ep / tw)
        for f, _y in loader:
            x = f.to(DEVICE, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                loss = loss_step(args.method, model, x, args)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if args.method == "dino" and ep < args.freeze_last_epochs:    # freeze prototype layer early (DINO)
                model.student_head.last_layer.weight.grad = None
            scaler.step(opt); scaler.update()
            if args.method in ("ijepa", "dino"):
                model.update_target(args.ema_start + (args.ema_end - args.ema_start) * (gstep / max(1, total - 1)))
            run += float(loss) * x.size(0); n += x.size(0); gstep += 1
        sched.step()
        print(f"ep {ep+1:3d}/{args.epochs} train={time.time()-te:.1f}s", flush=True)   # pure train-epoch (no probe)
        if (ep + 1) % args.eval_every == 0 or ep == 0:
            Ztr, Ytr = embed(model, tr); Zva, Yva = embed(model, va)
            pr = participation_ratio(Ztr); ps = " ".join(f"{k}={v:+.3f}" for k, v in probe2(Ztr, Ytr, Zva, Yva).items())
            print(f"ep {ep+1:3d}/{args.epochs}  loss={run/n:.4e}  PR={pr:.1f}  probe {ps}  ({time.time()-t0:.0f}s)", flush=True)

    out = args.ckpt_out or f"results/checkpoints/g1/{args.method}_{args.tag}.pt"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"model": model.state_dict(), "stats": tr.stats, "train_args": vars(args)}, out)
    print(f"  saved {out}", flush=True)


if __name__ == "__main__":
    main()
