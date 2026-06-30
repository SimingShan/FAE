"""Pretrain the grid CAE (L-DeepONet encoder) on NS frames with reconstruction loss, then freeze it.
Its recon relL2 = the grid-AE's 'recon floor' (the forecasting ceiling) — report it so the FAE-vs-grid
comparison is read against each AE's own ceiling.

  python scripts/train_grid_ae.py --ckpt_out results/checkpoints/ns/grid_ae/grid_cae_s0.pt
"""
import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.baselines.grid_ae import GridCAE
from src.data.ns import NSDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIDE = 64


def relL2(pred, true):
    return (torch.linalg.vector_norm((pred - true).flatten(1), dim=1) /
            torch.linalg.vector_norm(true.flatten(1), dim=1).clamp_min(1e-8)).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--latent", type=int, default=512)
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_traj", type=int, default=8)
    ap.add_argument("--ckpt_out", default="results/checkpoints/ns/grid_ae/grid_cae_s0.pt")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.n_traj = 2, 2

    tr = NSDataset("train", side=SIDE, mode="single", n_traj=args.n_traj)
    va = NSDataset("valid", side=SIDE, mode="single", n_traj=args.n_traj, stats=tr.stats)
    C = 3
    tl = DataLoader(tr, batch_size=64, shuffle=True, drop_last=True)
    model = GridCAE(in_ch=C, side=SIDE, latent=args.latent, ch=args.ch).to(DEVICE)
    print(f"=== grid-CAE pretrain  NS {C}ch res={SIDE} latent={args.latent} ch={args.ch}  "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M  train_frames={len(tr)} ===", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    @torch.no_grad()
    def recon_floor():
        model.eval(); e = 0.0; n = 0
        for x, _ in DataLoader(va, batch_size=128):
            x = x.to(DEVICE); e += relL2(model(x), x) * x.size(0); n += x.size(0)
        model.train(); return e / n

    for ep in range(args.epochs):
        t0 = time.time()
        for x, _ in tl:
            x = x.to(DEVICE)
            loss = F.mse_loss(model(x), x)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 20 == 19 or ep == args.epochs - 1 or args.smoke:
            print(f"  ep {ep+1:3d}/{args.epochs}  loss={loss.item():.4f}  recon_relL2={recon_floor():.4f}  ({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)
    rf = recon_floor()
    torch.save({"model": model.state_dict(),
                "train_args": {"in_chans": C, "side": SIDE, "latent": args.latent, "ch": args.ch, "dataset": "ns"},
                "recon_floor": rf}, args.ckpt_out)
    print(f"=== saved {args.ckpt_out}  recon_floor={rf:.4f}  (FAE's was ~0.11) ===", flush=True)
    if args.smoke:
        print("=== SMOKE OK ===", flush=True)


if __name__ == "__main__":
    main()
