"""Supervised single-frame upper bound + TWO-FRAME (physical) regression of (logRe, Sc).

Single-frame asks Re from one snapshot's spatial statistics. But Re is the inertial/
viscous balance — physically a statement about the RATE ∂u/∂t — so the physically
complete identification needs two frames + a known dt. This script does both:
  --frames 1 : one sparse frame -> regress.
  --frames 2 : two sparse frames (t, t+Δ) at known dt, SHARED single-frame encoder,
               head sees [r(t), r(t+Δ), r(t+Δ)-r(t), dt-embed] -> regress.
Reports R^2 (standardized labels) for the head AND a linear probe of frozen features.
  2-frame >> 1-frame  => the dynamics/rate carries Re that one snapshot can't.
  2-frame ~= 1-frame  => spatial statistics already saturate the available signal.
"""
import sys, os, time, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from src.models.fae import FAEEncoder
from src.data.well2d import ShearFlowClipDataset, make_coords_2d, fields_to_tokens
from src.metrics import lin_probe

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def readout(tok, mode):
    return tok.mean(1) if mode == "mean" else torch.cat([tok.mean(1), tok.std(1)], -1)


def dt_embed(dt, n=8):                                   # dt: (B,) in [0,1] -> (B,2n)
    f = torch.arange(1, n + 1, device=dt.device, dtype=dt.dtype)
    a = dt[:, None] * f[None, :] * math.pi
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


def feat(enc, coords, idx, fa, fb, dt, mode, two):
    ra = readout(enc(fields_to_tokens(fa, idx), coords[idx]), mode)
    if not two:
        return ra
    rb = readout(enc(fields_to_tokens(fb, idx), coords[idx]), mode)
    return torch.cat([ra, rb, rb - ra, dt_embed(dt)], -1)


@torch.no_grad()
def evaluate(enc, head, ds, coords, idx, mode, two, dt_max, lab_m, lab_s):
    enc.eval(); head.eval(); P, Y, Z = [], [], []
    for clip, y in DataLoader(ds, batch_size=128):
        clip = clip.to(DEVICE); B = clip.size(0); K = clip.size(2)
        bidx = torch.arange(B, device=DEVICE)
        if two:
            d = torch.full((B,), dt_max, device=DEVICE)
            t0 = (torch.rand(B, device=DEVICE) * (K - d).float()).long()
            fa = clip[bidx, :, t0]; fb = clip[bidx, :, t0 + d]; dt = d.float() / dt_max
        else:
            fa = clip[:, :, 0]; fb = None; dt = None
        z = feat(enc, coords, idx, fa, fb, dt, mode, two)
        P.append(head(z).cpu().numpy()); Z.append(z.cpu().numpy()); Y.append(y.numpy())
    P, Y, Z = np.concatenate(P), np.concatenate(Y), np.concatenate(Z)
    Ys = (Y - lab_m) / lab_s
    r2 = lambda p, t: float(1.0 - ((p - t) ** 2).sum() / (((t - t.mean()) ** 2).sum() + 1e-9))
    return [r2(P[:, j], Ys[:, j]) for j in range(2)], Z, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--frames", type=int, choices=[1, 2], default=1)
    ap.add_argument("--dt_max", type=int, default=4)
    ap.add_argument("--mcnt", type=int, nargs="+", default=[256, 512])
    ap.add_argument("--readout", choices=["mean", "meanstd"], default="meanstd")
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--n_seed", type=int, default=24)
    ap.add_argument("--frame_stride", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="sup")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    R = args.resolution; NPIX = R * R; two = args.frames == 2
    print(f"=== SUPERVISED [{args.tag}] frames={args.frames} dt_max={args.dt_max} "
          f"readout={args.readout} mcnt={args.mcnt} ===", flush=True)
    clip_len = (args.dt_max + 1) if two else 2
    tr = ShearFlowClipDataset("train", n_seed=args.n_seed, frame_stride=args.frame_stride,
                              clip_len=clip_len, side=R)
    va = ShearFlowClipDataset("valid", n_seed=8, frame_stride=args.frame_stride,
                              clip_len=clip_len, side=R, stats=tr.stats)
    loader = DataLoader(tr, batch_size=args.batch, shuffle=True, drop_last=True,
                        num_workers=4, pin_memory=True)
    coords = make_coords_2d(n_side=R, device=DEVICE)
    enc = FAEEncoder(emb_dim=320, num_iter=4, depth_per_iter=4, num_cross_heads=4,
                     num_self_heads=8, n_freq=32, max_freq=32, num_latents=128,
                     coord_dim=2, in_chans=4).to(DEVICE)
    rdim = 320 * (2 if args.readout == "meanstd" else 1)
    in_dim = (3 * rdim + 16) if two else rdim
    head = nn.Sequential(nn.Linear(in_dim, 256), nn.GELU(), nn.Linear(256, 2)).to(DEVICE)

    Ytr_all = np.stack([tr[i][1] for i in range(len(tr))])
    lab_m, lab_s = Ytr_all.mean(0), Ytr_all.std(0) + 1e-8
    tm = torch.tensor(lab_m, device=DEVICE, dtype=torch.float32)
    tsd = torch.tensor(lab_s, device=DEVICE, dtype=torch.float32)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    probe_idx = torch.randperm(NPIX, generator=torch.Generator(device=DEVICE).manual_seed(0), device=DEVICE)[:1024]
    t0 = time.time()
    for ep in range(args.epochs):
        enc.train(); head.train()
        for clip, y in loader:
            clip = clip.to(DEVICE, non_blocking=True); B = clip.size(0); K = clip.size(2)
            bidx = torch.arange(B, device=DEVICE)
            n = int(np.random.choice(args.mcnt)); idx = torch.randperm(NPIX, device=DEVICE)[:n]
            if two:
                delta = torch.randint(1, args.dt_max + 1, (B,), device=DEVICE)
                ts = (torch.rand(B, device=DEVICE) * (K - delta).float()).long()
                fa = clip[bidx, :, ts]; fb = clip[bidx, :, ts + delta]; dt = delta.float() / args.dt_max
            else:
                fa = clip[:, :, torch.randint(0, K, (1,)).item()]; fb = None; dt = None
            z = feat(enc, coords, idx, fa, fb, dt, args.readout, two)
            loss = F.mse_loss(head(z), (y.to(DEVICE) - tm) / tsd)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep + 1) % 20 == 0 or ep == 0 or ep == args.epochs - 1:
            hr, Zva, Yva = evaluate(enc, head, va, coords, probe_idx, args.readout, two, args.dt_max, lab_m, lab_s)
            _, Ztr, Ytr = evaluate(enc, head, tr, coords, probe_idx, args.readout, two, args.dt_max, lab_m, lab_s)
            lp = []
            for j in range(2):
                yt, yv = Ytr[:, j], Yva[:, j]; m, s = yt.mean(), yt.std() + 1e-8
                lp.append(lin_probe(Ztr, (yt - m) / s, Zva, (yv - m) / s))
            print(f"ep {ep+1:3d}/{args.epochs}  HEAD R2 logRe={hr[0]:+.3f} Sc={hr[1]:+.3f}  "
                  f"| LINPROBE logRe={lp[0]:+.3f} Sc={lp[1]:+.3f}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
