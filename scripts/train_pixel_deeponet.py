"""Pixel-space DeepONet forecasting on NS (the no-autoencoder baseline).

field_t --branch CNN--> coeffs ; (coord, dt) --trunk--> basis ; Sum_p(branch.trunk) -> field_{t+dt}(coord).
Trained end-to-end on all (i,j) gaps. Scored direct + stepwise vs persistence — same protocol as the
latent arms (no recon floor: it predicts the field directly, no AE round-trip).

  python scripts/train_pixel_deeponet.py --smoke
"""
import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.deeponet import PixelDeepONet
from src.data.ns import NSDataset
from src.data.well2d import make_coords_2d

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIDE = 64
DT_DIV = 8.0                                                   # match the latent arms' dt normalization


def relL2(pred, true):
    return (torch.linalg.vector_norm((pred - true).flatten(1), dim=1) /
            torch.linalg.vector_norm(true.flatten(1), dim=1).clamp_min(1e-8)).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--frame_stride", type=int, default=4)
    ap.add_argument("--rollout", type=int, default=4)
    ap.add_argument("--p", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_traj", type=int, default=8)
    ap.add_argument("--ckpt_out", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.n_traj, args.rollout = 2, 2, 2

    C = 3
    coords = make_coords_2d(n_side=SIDE, device=DEVICE)            # (SIDE^2, 2)
    cl = args.rollout + 1
    tr = NSDataset("train", side=SIDE, mode="clip", clip_len=cl, frame_stride=args.frame_stride, n_traj=args.n_traj)
    va = NSDataset("valid", side=SIDE, mode="clip", clip_len=cl, frame_stride=args.frame_stride,
                   n_traj=args.n_traj, stats=tr.stats)
    tl = DataLoader(tr, batch_size=32, shuffle=True, drop_last=True)
    model = PixelDeepONet(in_ch=C, side=SIDE, p=args.p).to(DEVICE)
    # initialize lazy layers
    with torch.no_grad():
        model(torch.zeros(1, C, SIDE, SIDE, device=DEVICE), coords, torch.ones(1, device=DEVICE))
    print(f"=== forecast [PIXEL-DeepONet] NS {C}ch res={SIDE} rollout={args.rollout} train_clips={len(tr)} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M ===", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def to_field(out):                                            # (B,N,C) -> (B,C,SIDE,SIDE)
        return out.permute(0, 2, 1).reshape(-1, C, SIDE, SIDE)

    @torch.no_grad()
    def evaluate():
        model.eval(); R = args.rollout
        direct = np.zeros(R); step = np.zeros(R); persist = np.zeros(R); nb = 0
        for clip, _ in DataLoader(va, batch_size=64):
            clip = clip.to(DEVICE); f0 = clip[:, :, 0]; n = f0.size(0); fr = f0
            for k in range(R):
                fd = to_field(model(f0, coords, torch.full((n,), (k + 1) / DT_DIV, device=DEVICE)))   # DIRECT
                fr = to_field(model(fr, coords, torch.full((n,), 1.0 / DT_DIV, device=DEVICE)))        # STEPWISE
                true = clip[:, :, k + 1]
                direct[k] += relL2(fd, true) * n; step[k] += relL2(fr, true) * n; persist[k] += relL2(f0, true) * n
            nb += n
        model.train(); return direct / nb, step / nb, persist / nb

    for ep in range(args.epochs):
        t0 = time.time()
        for clip, _ in tl:
            clip = clip.to(DEVICE); n, T = clip.size(0), clip.size(2)
            loss = 0.0; npair = 0
            for i in range(T - 1):
                for j in range(i + 1, T):
                    pred = to_field(model(clip[:, :, i], coords, torch.full((n,), (j - i) / DT_DIV, device=DEVICE)))
                    loss = loss + F.mse_loss(pred, clip[:, :, j]); npair += 1
            loss = loss / npair
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 9 or ep == args.epochs - 1 or args.smoke:
            d, s, p = evaluate()
            msg = "  ".join(f"t+{k+1}:dir{d[k]:.3f}/step{s[k]:.3f}/per{p[k]:.3f}" for k in range(args.rollout))
            print(f"  ep {ep+1:3d}/{args.epochs}  loss={loss.item():.4f}  {msg}  ({time.time()-t0:.0f}s)", flush=True)

    if args.ckpt_out:
        os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)
        d, s, p = evaluate()
        torch.save({"model": model.state_dict(), "args": vars(args), "direct": d.tolist(),
                    "step": s.tolist(), "persist": p.tolist()}, args.ckpt_out)
        print(f"=== saved {args.ckpt_out}  direct={np.round(d,4).tolist()}  step={np.round(s,4).tolist()} ===", flush=True)
    if args.smoke:
        print("=== SMOKE OK ===", flush=True)


if __name__ == "__main__":
    main()
