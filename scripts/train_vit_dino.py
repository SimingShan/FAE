"""Conventional ViT-DINO on PDE fields — DECOUPLES the DINO objective from the FAE architecture.
Authentic DINO: a standard ViT encoder + multi-crop random-resized-crop augmentation + DINOHead + DINOLoss
(reused from train_fae_dino). If this LEARNS (probe recovers) but FAE-DINO stalls -> the FAE architecture
is the blocker. If this ALSO stalls -> DINO struggles on small PDE data regardless. The control."""
import os, sys, argparse, copy, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from benchmarks.jepa.ijepa2d import ViT2D
from scripts.train_fae_dino import DINOHead, DINOLoss, ridge_r2, participation_ratio
from src.data.ns import NSDataset
from src.utils import set_seed, seed_worker
from src.utils.seed import torch_generator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def multicrop(x, out, smin, smax):
    """per-sample random-resized-crop + horizontal flip via affine grid (DINO's augmentation, fields)."""
    B, C, H, W = x.shape
    s = (smin + (smax - smin) * torch.rand(B, device=x.device)).sqrt()      # side scale (area-uniform)
    tx = (torch.rand(B, device=x.device) * 2 - 1) * (1 - s)
    ty = (torch.rand(B, device=x.device) * 2 - 1) * (1 - s)
    flip = (torch.rand(B, device=x.device) < 0.5).float() * 2 - 1
    th = torch.zeros(B, 2, 3, device=x.device)
    th[:, 0, 0] = s * flip; th[:, 1, 1] = s; th[:, 0, 2] = tx; th[:, 1, 2] = ty
    grid = F.affine_grid(th, (B, C, out, out), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False)


@torch.no_grad()
def embed_vit(enc, ds, n=512):
    xs = torch.from_numpy(np.stack([ds[i][0].numpy() for i in range(min(len(ds), n))])).to(DEVICE)
    ys = torch.tensor(np.stack([np.atleast_1d(ds[i][1]) for i in range(min(len(ds), n))]), dtype=torch.float32, device=DEVICE)
    return enc(xs).mean(1), ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ns"); ap.add_argument("--resolution", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=80); ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embed_dim", type=int, default=256); ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--out_dim", type=int, default=4096); ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--teacher_temp", type=float, default=0.04); ap.add_argument("--n_local", type=int, default=2)
    ap.add_argument("--tag", default="vit_dino"); ap.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args(); set_seed(args.seed); R = args.resolution; C = 3 if args.dataset == "ns" else 4
    ncrops = 2 + args.n_local
    print(f"=== ViT-DINO [{args.tag}] out_dim={args.out_dim} Tt={args.teacher_temp} ncrops={ncrops} res={R} ===", flush=True)

    tr = NSDataset("train", side=R, mode="single", clip_len=2, frame_stride=4, n_traj=12)
    va = NSDataset("valid", side=R, mode="single", clip_len=2, frame_stride=4, n_traj=8, stats=tr.stats)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True,
                    worker_init_fn=seed_worker, generator=torch_generator(args.seed))
    enc = ViT2D(img_size=R, patch_size=args.patch, in_chans=C, embed_dim=args.embed_dim, depth=args.depth, num_heads=8).to(DEVICE)
    s_head = DINOHead(args.embed_dim, args.out_dim).to(DEVICE)
    t_enc = copy.deepcopy(enc).eval()
    t_head = DINOHead(args.embed_dim, args.out_dim).to(DEVICE); t_head.load_state_dict(s_head.state_dict()); t_head.eval()
    for p in list(t_enc.parameters()) + list(t_head.parameters()): p.requires_grad_(False)
    dino = DINOLoss(args.out_dim, ncrops, args.teacher_temp, args.teacher_temp, 0, args.epochs).to(DEVICE)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(s_head.parameters()), lr=args.lr)

    for ep in range(args.epochs):
        enc.train(); s_head.train(); ad = an = 0; t0 = time.time()
        for x, _ in tl:
            x = x.to(DEVICE)
            gl = [multicrop(x, R, 0.4, 1.0) for _ in range(2)]
            lo = [multicrop(x, R, 0.05, 0.4) for _ in range(args.n_local)]
            s_out = s_head(torch.cat([enc(c).mean(1) for c in gl + lo], 0))
            with torch.no_grad():
                t_out = t_head(torch.cat([t_enc(c).mean(1) for c in gl], 0))
            loss = dino(s_out, t_out, ep)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for pt, pe in zip(t_enc.parameters(), enc.parameters()): pt.mul_(args.ema).add_(pe, alpha=1 - args.ema)
                for pt, pe in zip(t_head.parameters(), s_head.parameters()): pt.mul_(args.ema).add_(pe, alpha=1 - args.ema)
            ad += loss.item() * x.size(0); an += x.size(0)
        if ep % 10 == 9 or ep == args.epochs - 1:
            enc.eval(); Ztr, Ytr = embed_vit(enc, tr); Zva, Yva = embed_vit(enc, va)
            print(f"ep {ep+1:3d}/{args.epochs}  distill={ad/an:.3f}  PR={participation_ratio(Ztr):.1f}  "
                  f"probe R2={ridge_r2(Ztr, Ytr, Zva, Yva):+.3f}  ({time.time()-t0:.0f}s)", flush=True)
    if args.save:
        out = f"results/checkpoints/g1/vit_dino_{args.tag}.pt"; os.makedirs(os.path.dirname(out), exist_ok=True)
        torch.save({"model": enc.state_dict(), "stats": tr.stats, "train_args": vars(args)}, out); print(f"  saved {out}", flush=True)


if __name__ == "__main__":
    main()
