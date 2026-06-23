"""Masked self-distillation FAE (Sonata/BYOL-grounded; addresses the 'reconstruction = geometric shortcut'
critique and our own MAE-best result).

  - STUDENT encodes a sparse/masked sensor subset -> latents.
  - EMA TEACHER encodes a fuller subset -> latents (stop-grad).
  - A predictor maps student latents to teacher latents; loss = 1 - cosine  (no decoder on this branch ->
    no reconstruction shortcut). EMA target prevents collapse (no VICReg crutch).
  - The DECODER is trained separately on DETACHED latents (so it stays usable for the REPA per-patch
    readout, without the reconstruction gradient shaping the encoder).
  - Geometric Fourier coordinate features (lifts the 8-cycle band-limit of the legacy linear encoding).
Encoder is the product; predictor/teacher discarded at eval.
"""
import os, sys, argparse, copy, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from src.models.fae import FAE, TokenPredictor
from src.data.ns import NSDataset
from src.data.well2d import make_coords_2d, fields_to_tokens
from src.utils import set_seed, seed_worker
from src.utils.seed import torch_generator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ridge_r2(Ztr, Ytr, Zva, Yva, lam=1.0):
    Ztr = (Ztr - Ztr.mean(0)) / Ztr.std(0).clamp_min(1e-6); Zva = (Zva - Zva.mean(0)) / Zva.std(0).clamp_min(1e-6)
    A = Ztr.T @ Ztr + lam * torch.eye(Ztr.size(1), device=Ztr.device)
    W = torch.linalg.solve(A, Ztr.T @ Ytr); pred = Zva @ W
    ss = ((Yva - pred) ** 2).sum(0); tot = ((Yva - Yva.mean(0)) ** 2).sum(0).clamp_min(1e-6)
    return (1 - ss / tot).mean().item()


def participation_ratio(Z):
    Z = Z - Z.mean(0); ev = torch.linalg.eigvalsh(torch.cov(Z.T)).clamp_min(0)
    return (ev.sum() ** 2 / (ev ** 2).sum().clamp_min(1e-12)).item()


@torch.no_grad()
def embed(model, ds, coords, idx, n=512):
    xs = torch.from_numpy(np.stack([ds[i][0].numpy() for i in range(min(len(ds), n))])).to(DEVICE)
    ys = torch.tensor(np.stack([np.atleast_1d(ds[i][1]) for i in range(min(len(ds), n))]), dtype=torch.float32, device=DEVICE)
    Z = model.encode_tokens(fields_to_tokens(xs, idx), coords[idx]).mean(1)
    return Z, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ns"); ap.add_argument("--resolution", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--student_frac", type=float, default=0.10, help="student sensor fraction (masked view)")
    ap.add_argument("--teacher_sensors", type=int, default=1024); ap.add_argument("--n_query", type=int, default=1024)
    ap.add_argument("--lam_dec", type=float, default=1.0); ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--emb_dim", type=int, default=320); ap.add_argument("--num_latents", type=int, default=128)
    ap.add_argument("--fourier_geometric", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--tag", default="fae_masked"); ap.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args(); set_seed(args.seed); R = args.resolution; C = 3 if args.dataset == "ns" else 4
    print(f"=== FAE-masked [{args.tag}] geo={args.fourier_geometric} student_frac={args.student_frac} "
          f"teacher={args.teacher_sensors} lam_dec={args.lam_dec} ema={args.ema} res={R} ===", flush=True)

    tr = NSDataset("train", side=R, mode="single", clip_len=2, frame_stride=4, n_traj=12)
    va = NSDataset("valid", side=R, mode="single", clip_len=2, frame_stride=4, n_traj=8, stats=tr.stats)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True,
                    worker_init_fn=seed_worker, generator=torch_generator(args.seed))
    model = FAE(emb_dim=args.emb_dim, num_latents=args.num_latents, coord_dim=2, in_chans=C,
                fourier_geometric=args.fourier_geometric).to(DEVICE)
    teacher = copy.deepcopy(model.encoder).eval(); [p.requires_grad_(False) for p in teacher.parameters()]
    predictor = TokenPredictor(args.emb_dim, depth=2, heads=8).to(DEVICE)
    opt = torch.optim.AdamW(list(model.parameters()) + list(predictor.parameters()), lr=args.lr)
    coords = make_coords_2d(n_side=R, device=DEVICE); NPIX = R * R
    n_stu = max(16, int(NPIX * args.student_frac))
    g = torch.Generator(device=DEVICE).manual_seed(0); pidx = torch.randperm(NPIX, generator=g, device=DEVICE)[:1024]

    for ep in range(args.epochs):
        model.train(); ag = {"d": 0.0, "r": 0.0, "n": 0}; t0 = time.time()
        for x, _ in tl:
            x = x.to(DEVICE); n = x.size(0)
            iA = torch.randperm(NPIX, device=DEVICE)[:n_stu]                    # student: masked
            iB = torch.randperm(NPIX, device=DEVICE)[:args.teacher_sensors]     # teacher: fuller
            iq = torch.randperm(NPIX, device=DEVICE)[:args.n_query]
            lat_s = model.encode_tokens(fields_to_tokens(x, iA), coords[iA])
            with torch.no_grad():
                lat_t = teacher(fields_to_tokens(x, iB), coords[iB])
            pred = predictor(lat_s, torch.zeros(n, device=DEVICE))
            distill = (1 - F.cosine_similarity(F.normalize(pred, dim=-1), F.normalize(lat_t, dim=-1), dim=-1)).mean()
            rec = F.mse_loss(model.decoder(lat_s.detach(), coords[iq]), fields_to_tokens(x, iq))  # decoder only
            loss = distill + args.lam_dec * rec
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for pt, pe in zip(teacher.parameters(), model.encoder.parameters()):
                    pt.mul_(args.ema).add_(pe, alpha=1 - args.ema)
            ag["d"] += distill.item() * n; ag["r"] += rec.item() * n; ag["n"] += n
        if ep % 10 == 9 or ep == args.epochs - 1:
            model.eval(); Ztr, Ytr = embed(model, tr, coords, pidx); Zva, Yva = embed(model, va, coords, pidx)
            r2 = ridge_r2(Ztr, Ytr, Zva, Yva); pr = participation_ratio(Ztr)
            print(f"ep {ep+1:3d}/{args.epochs}  distill={ag['d']/ag['n']:.4f}  rec={ag['r']/ag['n']:.3e}  "
                  f"PR={pr:.1f}  probe buoyancy R2={r2:+.3f}  ({time.time()-t0:.0f}s)", flush=True)
    if args.save:
        out = f"results/checkpoints/g1/faep_masked_{args.tag}.pt"; os.makedirs(os.path.dirname(out), exist_ok=True)
        torch.save({"model": model.state_dict(), "stats": tr.stats, "train_args": vars(args)}, out)
        print(f"  saved {out}", flush=True)


if __name__ == "__main__":
    main()
