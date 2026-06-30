"""FNO forecasting on NS — the dense full-grid ORACLE. field_t -> field_{t+dt}, dt-conditioned, trained
end-to-end on all gaps. Scored direct + stepwise vs persistence (same protocol as the other arms).

  python scripts/train_fno.py --smoke
"""
import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.baselines.fno import FNO2d
from src.data.ns import NSDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIDE, DT_DIV = 64, 8.0


def relL2(pred, true):
    return (torch.linalg.vector_norm((pred - true).flatten(1), dim=1) /
            torch.linalg.vector_norm(true.flatten(1), dim=1).clamp_min(1e-8)).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--frame_stride", type=int, default=4)
    ap.add_argument("--rollout", type=int, default=4)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_traj", type=int, default=8)
    ap.add_argument("--ckpt_out", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.n_traj, args.rollout = 2, 2, 2

    C = 3
    cl = args.rollout + 1
    tr = NSDataset("train", side=SIDE, mode="clip", clip_len=cl, frame_stride=args.frame_stride, n_traj=args.n_traj)
    va = NSDataset("valid", side=SIDE, mode="clip", clip_len=cl, frame_stride=args.frame_stride,
                   n_traj=args.n_traj, stats=tr.stats)
    tl = DataLoader(tr, batch_size=32, shuffle=True, drop_last=True)
    model = FNO2d(in_ch=C, out_ch=C, width=args.width, modes=args.modes, n_layers=args.layers).to(DEVICE)
    print(f"=== forecast [FNO oracle] NS {C}ch res={SIDE} width={args.width} modes={args.modes} "
          f"rollout={args.rollout} train_clips={len(tr)} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M ===", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    @torch.no_grad()
    def evaluate():
        model.eval(); R = args.rollout
        direct = np.zeros(R); step = np.zeros(R); persist = np.zeros(R); nb = 0
        for clip, _ in DataLoader(va, batch_size=64):
            clip = clip.to(DEVICE); f0 = clip[:, :, 0]; n = f0.size(0); fr = f0
            for k in range(R):
                fd = model(f0, torch.full((n,), (k + 1) / DT_DIV, device=DEVICE))   # DIRECT
                fr = model(fr, torch.full((n,), 1.0 / DT_DIV, device=DEVICE))        # STEPWISE
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
                    pred = model(clip[:, :, i], torch.full((n,), (j - i) / DT_DIV, device=DEVICE))
                    loss = loss + F.mse_loss(pred, clip[:, :, j]); npair += 1
            loss = loss / npair
            opt.zero_grad(); loss.backward(); opt.step()
        ev = (ep % 10 == 9 or ep == args.epochs - 1 or args.smoke)         # full eval (valid set) every 10
        msg = ""
        if ev:
            d, s, p = evaluate()
            msg = "  | " + "  ".join(f"t+{k+1}:dir{d[k]:.3f}/step{s[k]:.3f}/per{p[k]:.3f}" for k in range(args.rollout))
        print(f"  ep {ep+1:3d}/{args.epochs}  loss={loss.item():.4f}{msg}  ({time.time()-t0:.0f}s/ep)", flush=True)  # per-epoch header

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
