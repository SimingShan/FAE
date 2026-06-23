"""Validate the 'lead': spatially-anchored latent FIELD + discretization-consistency.

2x2 ablation (anchoring on/off x disc-consistency on/off) on NS-2D 64^2, each capacity-matched (~7M):
  --no-vanilla --lam_disc 1   = ANCHORED + DISC   (the lead)
  --no-vanilla --lam_disc 0   = ANCHORED only
  --vanilla    --lam_disc 1   = vanilla FAE + DISC
  --vanilla    --lam_disc 0   = vanilla FAE recon-only (current FAE baseline)

Two validation curves at eval (the claims that must beat baseline):
  CONV  ‖z_n − z_dense‖ vs sensor count n  -> faster/smoother monotone decrease = better convergence
  SPARSE relL2(full-grid recon from n sensors) vs n -> lower at LOW n = better sparse reconstruction
"""
import os, sys, argparse, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from src.models.anchored_fae import AnchoredFAE
from src.data.ns import NSDataset
from src.data.well2d import make_coords_2d, fields_to_tokens
from src.utils import set_seed, seed_worker
from src.utils.seed import torch_generator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COUNTS = [16, 32, 64, 128, 256, 512, 1024, 2048]


def relL2(pred, tgt):
    return (torch.linalg.norm((pred - tgt).flatten(1), dim=1) /
            torch.linalg.norm(tgt.flatten(1), dim=1).clamp_min(1e-6)).mean().item()


@torch.no_grad()
def sparse_recon_curve(enc, dec, x, coords, counts, R, C, NPIX):
    """relL2 of the FULL-grid reconstruction from n scattered sensors, vs n. Lower at low n = better."""
    out = []
    for n in counts:
        i = torch.randperm(NPIX, device=DEVICE)[:n]
        pred = dec(enc(fields_to_tokens(x, i), coords[i]), coords).reshape(x.size(0), R, R, C).permute(0, 3, 1, 2)
        out.append((n, relL2(pred, x)))
    return out


@torch.no_grad()
def convergence_curve(enc, x, coords, counts, ref_n, NPIX):
    """‖z_n − z_dense‖ per-latent vs sensor count n. Monotone, fast decrease = good convergence."""
    iref = torch.randperm(NPIX, generator=torch.Generator(device=DEVICE).manual_seed(0), device=DEVICE)[:ref_n]
    zref = enc(fields_to_tokens(x, iref), coords[iref])
    out = []
    for n in counts:
        i = torch.randperm(NPIX, device=DEVICE)[:n]
        out.append((n, ((enc(fields_to_tokens(x, i), coords[i]) - zref) ** 2).mean().item()))
    return out


def build(args, C):
    M = args.n_anchor ** 2
    if args.vanilla:
        from src.models.fae import FAE
        m = FAE(emb_dim=320, num_latents=M, in_chans=C, coord_dim=2).to(DEVICE)
        return m, (lambda u, c: m.encode_tokens(u, c)), (lambda z, c: m.decoder(z, c))
    m = AnchoredFAE(emb_dim=320, n_anchor_side=args.n_anchor, in_chans=C).to(DEVICE)
    return m, (lambda u, c: m.encode(u, c)), (lambda z, c: m.decode(z, c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--resolution", type=int, default=64)
    ap.add_argument("--n_anchor", type=int, default=12); ap.add_argument("--lam_disc", type=float, default=1.0)
    ap.add_argument("--vanilla", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--mcnt", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--n_query", type=int, default=1024); ap.add_argument("--n_traj", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default="anc")
    ap.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args(); set_seed(args.seed); R = args.resolution; NPIX = R * R; C = 3
    arm = ("vanilla" if args.vanilla else "anchored") + ("+disc" if args.lam_disc > 0 else "+nodisc")
    print(f"=== AnchoredFAE ablation [{args.tag}] arm={arm} M={args.n_anchor**2} lam_disc={args.lam_disc} "
          f"mcnt={args.mcnt} res={R} ===", flush=True)
    tr = NSDataset("train", side=R, mode="single", clip_len=2, frame_stride=4, n_traj=args.n_traj)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True,
                    worker_init_fn=seed_worker, generator=torch_generator(args.seed))
    model, enc, dec = build(args, C)
    print(f"  params {sum(p.numel() for p in model.parameters())/1e6:.2f}M  train {len(tr)}", flush=True)
    coords = make_coords_2d(n_side=R, device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); ar = ad = an = 0.0
        for x, _ in tl:
            x = x.to(DEVICE); B = x.size(0)
            iA = torch.randperm(NPIX, device=DEVICE)[:int(np.random.choice(args.mcnt))]
            iB = torch.randperm(NPIX, device=DEVICE)[:int(np.random.choice(args.mcnt))]
            iq = torch.randperm(NPIX, device=DEVICE)[:args.n_query]
            zA = enc(fields_to_tokens(x, iA), coords[iA])
            rec = F.mse_loss(dec(zA, coords[iq]), fields_to_tokens(x, iq))
            loss = rec
            if args.lam_disc > 0:
                zB = enc(fields_to_tokens(x, iB), coords[iB])
                disc = F.mse_loss(zA, zB); loss = rec + args.lam_disc * disc
            else:
                disc = torch.zeros((), device=DEVICE)
            opt.zero_grad(); loss.backward(); opt.step()
            ar += rec.item() * B; ad += disc.item() * B; an += B
        if ep % 10 == 9 or ep == 0:
            model.eval()
            xb = next(iter(tl))[0].to(DEVICE)
            cc = convergence_curve(enc, xb, coords, COUNTS, 4096, NPIX)
            sr = sparse_recon_curve(enc, dec, xb, coords, COUNTS, R, C, NPIX)
            mono = all(cc[k][1] >= cc[k + 1][1] for k in range(len(cc) - 1))
            ccs = " ".join(f"{n}:{d:.3f}" for n, d in cc)
            srs = " ".join(f"{n}:{d:.3f}" for n, d in sr)
            print(f"ep {ep+1:3d}/{args.epochs}  rec={ar/an:.3e} disc={ad/an:.3e}  "
                  f"CONV[{'MONO' if mono else 'NON'}] {ccs}  | SPARSE-relL2 {srs}  ({time.time()-t0:.0f}s)", flush=True)
            model.train()
    if args.save:
        out = f"results/checkpoints/g1/anchored_fae_{args.tag}.pt"; os.makedirs(os.path.dirname(out), exist_ok=True)
        torch.save({"model": model.state_dict(), "stats": tr.stats, "train_args": vars(args)}, out)
        print(f"  saved {out}", flush=True)


if __name__ == "__main__":
    main()
