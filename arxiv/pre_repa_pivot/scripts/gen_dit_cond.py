"""Conditional generative rollout: condition on frame_t, GENERATE frame_{t+dt} by sampling
p(x_{t+dt} | x_t) with a flow-matching DiT (no VAE). Unlike a deterministic UNet (blurry mean future),
the generative model produces a SHARP plausible future. --align {none,fae} = with/without REPA (align
DiT tokens to the FAE features of the clean future). Conditioning = clean present frame concatenated
as extra channels. ACCURACY metric = relative-L2 vs the true future (+ spectrum for sharpness)."""
import os, sys, argparse, copy, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external/REPA"))
from torch.utils.data import DataLoader
from models.sit import SiT_models
from src.data.ns import NSDataset
from src.utils import set_seed, seed_worker
from src.utils.seed import torch_generator
from scripts.gen_dit import radial_spectrum, fae_setup, fae_feats

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_clips(dataset, split, R, stats):
    """(present, future) clips: NS 3ch / shear_flow 4ch, both (C, 2, H, W)."""
    if dataset == "ns":
        from src.data.ns import NSDataset
        return NSDataset(split, side=R, mode="clip", clip_len=2, frame_stride=4, n_traj=12 if split == "train" else 8, stats=stats)
    from src.data.well2d import ShearFlowWindowDataset
    return ShearFlowWindowDataset(split, n_seed=24 if split == "train" else 8, n_frames=2, side=R, stats=stats)


@torch.no_grad()
def sample(model, cond, C, R, steps=50):
    """cond:(B,C,R,R) clean present -> generate future (B,C,R,R)."""
    n = cond.size(0); xf = torch.randn(n, C, R, R, device=DEVICE); y = torch.zeros(n, dtype=torch.long, device=DEVICE)
    for i in range(steps):
        v, _ = model(torch.cat([xf, cond], 1), torch.full((n,), 1 - i / steps, device=DEVICE), y)
        xf = xf - (1 / steps) * v[:, :C]
    return xf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=64); ap.add_argument("--size", default="SiT-S/4")
    ap.add_argument("--align", choices=["none", "fae"], default="none"); ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=80); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default="cond")
    ap.add_argument("--dataset", default="ns")
    ap.add_argument("--fae_ckpt", default="results/checkpoints/g1/faep_twoview_fae_ns_tw.pt")
    args = ap.parse_args(); set_seed(args.seed); R = args.resolution; C = 3 if args.dataset == "ns" else 4
    print(f"=== cond-gen ROLLOUT [{args.tag}] {args.size} align={args.align} res={R} ds={args.dataset} ({C}ch) seed={args.seed} ===", flush=True)

    tr = get_clips(args.dataset, "train", R, None)
    va = get_clips(args.dataset, "valid", R, tr.stats)
    print(f"  train {len(tr)} pairs ; valid {len(va)} pairs", flush=True)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True,
                    worker_init_fn=seed_worker, generator=torch_generator(args.seed))
    vl = DataLoader(va, batch_size=64, shuffle=False, num_workers=4)
    sz = args.size.split("-")[1].split("/")[0]; patch = int(args.size.split("/")[1])
    extra = {"decoder_hidden_size": 384} if sz == "S" else {}
    model = SiT_models[args.size](input_size=R, in_channels=2 * C, num_classes=1, z_dims=[320], encoder_depth=4,
                                  fused_attn=False, qk_norm=False, **extra).to(DEVICE)
    print(f"  SiT params {sum(p.numel() for p in model.parameters())/1e6:.1f}M (cond=present+noised future)", flush=True)
    fae = fc = fs = fp = None
    if args.align == "fae":
        fae, fc, fs, fp = fae_setup(args.fae_ckpt, R, patch); print(f"  REPA align -> FAE features of future (lam={args.lam})", flush=True)
    ema = copy.deepcopy(model).eval(); [p.requires_grad_(False) for p in ema.parameters()]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    @torch.no_grad()
    def evaluate():
        ema.eval(); rel = sp = n = 0; gs = []
        for clip, _ in vl:
            pres, fut = clip[:, :, 0].to(DEVICE), clip[:, :, 1].to(DEVICE)
            g = sample(ema, pres, C, R)
            rel += (torch.linalg.norm((g - fut).flatten(1), dim=1) / torch.linalg.norm(fut.flatten(1), dim=1).clamp_min(1e-6)).sum().item()
            n += pres.size(0); gs.append(g)
        gen = torch.cat(gs); ref = radial_spectrum(torch.cat([c[:, :, 1] for c, _ in vl]).to(DEVICE))
        sd = (radial_spectrum(gen) - ref).abs().mean().item() / ref.abs().mean().item()
        return rel / n, sd

    for ep in range(args.epochs):
        model.train()
        for clip, _ in tl:
            pres, fut = clip[:, :, 0].to(DEVICE), clip[:, :, 1].to(DEVICE); nb = pres.size(0)
            y = torch.zeros(nb, dtype=torch.long, device=DEVICE)
            t = torch.rand(nb, device=DEVICE); eps = torch.randn_like(fut)
            xtf = (1 - t)[:, None, None, None] * fut + t[:, None, None, None] * eps
            v, zs = model(torch.cat([xtf, pres], 1), t, y)
            loss = F.mse_loss(v[:, :C], eps - fut)
            if args.align == "fae":
                tgt = fae_feats(fae, fut, fc, fs, fp)
                loss = loss + args.lam * (1 - F.cosine_similarity(F.normalize(zs[0], dim=-1), F.normalize(tgt, dim=-1), dim=-1).mean())
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()): pe.mul_(0.999).add_(pm, alpha=0.001)
        if ep % 20 == 19 or ep == args.epochs - 1:
            rel, sd = evaluate()
            print(f"  ep {ep+1:3d}/{args.epochs}  loss={loss.item():.4f}  forecast_relL2={rel:.4f}  spectrum_dist={sd:.4f}", flush=True)
    out = f"results/checkpoints/g1/ditcond_{args.dataset}_{args.align}_s{args.seed}.pt"; os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"ema": ema.state_dict(), "args": vars(args)}, out); print(f"DONE saved {out}", flush=True)


if __name__ == "__main__":
    main()
