"""Spatio-temporal DiT generation: generate a T-frame NS trajectory (frames stacked as channels,
T*3 chans). REPA --align fae aligns the SiT tokens to the FROZEN FAE's per-frame per-patch features
(stacked over time) -> tests whether a dynamics-trained encoder helps generate coherent evolutions.
Eval: per-frame energy-spectrum dist (avg over T) + a temporal-coherence proxy (mean |frame_{t+1}-frame_t|
distance to real)."""
import os, sys, argparse, math, copy, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external/REPA"))
from torch.utils.data import DataLoader
from models.sit import SiT_models
from src.data.ns import NSDataset
from src.utils import set_seed, seed_worker
from src.utils.seed import torch_generator
from scripts.gen_dit import radial_spectrum

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def fae_setup(ckpt, R, patch):
    from scripts.eval_ns_probe import load_fae
    from src.data.well2d import make_coords_2d
    m, _, _ = load_fae(ckpt); m.eval(); [p.requires_grad_(False) for p in m.parameters()]
    coords = make_coords_2d(R, DEVICE); g = torch.Generator(device=DEVICE).manual_seed(0)
    sidx = torch.randperm(R * R, generator=g, device=DEVICE)[:512]
    return m, coords, sidx, make_coords_2d(R // patch, DEVICE)


@torch.no_grad()
def fae_feats_st(m, x, coords, sidx, pcoords, T, dyn=False):
    """x:(B,T*3,H,W) -> per-frame FAE per-patch features; dyn=True also appends feature DELTAS
    (feat_{t+1}-feat_t) so the alignment carries temporal CHANGE, not just static frames."""
    B, _, H, W = x.shape
    xf = x.view(B, T, 3, H, W).reshape(B * T, 3, H, W)
    from src.data.well2d import fields_to_tokens
    lat = m.encode_tokens(fields_to_tokens(xf, sidx), coords[sidx])
    feat = m.decoder(lat, pcoords, return_feats=True)                 # (B*T, n_patch, 320)
    P, D = feat.shape[1], feat.shape[2]
    f = feat.view(B, T, P, D).permute(0, 2, 1, 3)                     # (B, P, T, D)
    if dyn:
        d = f[:, :, 1:] - f[:, :, :-1]                               # (B, P, T-1, D)  temporal change
        return torch.cat([f.reshape(B, P, T * D), d.reshape(B, P, (T - 1) * D)], dim=-1)
    return f.reshape(B, P, T * D)


@torch.no_grad()
def sample(model, n, ch, R, steps=50):
    model.eval(); x = torch.randn(n, ch, R, R, device=DEVICE); y = torch.zeros(n, dtype=torch.long, device=DEVICE)
    for i in range(steps):
        v, _ = model(x, torch.full((n,), 1 - i / steps, device=DEVICE), y)
        x = x - (1 / steps) * v
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=4); ap.add_argument("--resolution", type=int, default=64)
    ap.add_argument("--size", default="SiT-S/4"); ap.add_argument("--align", choices=["none", "fae"], default="none")
    ap.add_argument("--lam", type=float, default=0.5); ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=24); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default="st")
    ap.add_argument("--dyn", action="store_true", help="align to FAE feature deltas (temporal change), not just static frames")
    ap.add_argument("--fae_ckpt", default="results/checkpoints/g1/faep_twoview_fae_ns_tw.pt")
    args = ap.parse_args(); set_seed(args.seed); R, T = args.resolution, args.frames; ch = 3 * T
    print(f"=== gen-ST DiT [{args.tag}] {args.size} align={args.align} T={T} res={R} ({ch}ch) seed={args.seed} ===", flush=True)

    tr = NSDataset("train", side=R, mode="clip", clip_len=T, frame_stride=4, n_traj=12)
    print(f"  {len(tr)} clips", flush=True)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True,
                    worker_init_fn=seed_worker, generator=torch_generator(args.seed))
    sz = args.size.split("-")[1].split("/")[0]; patch = int(args.size.split("/")[1])
    extra = {"decoder_hidden_size": 384} if sz == "S" else {}
    zdim = (2 * T - 1) * 320 if args.dyn else T * 320
    model = SiT_models[args.size](input_size=R, in_channels=ch, num_classes=1, z_dims=[zdim], encoder_depth=4,
                                  fused_attn=False, qk_norm=False, **extra).to(DEVICE)
    print(f"  SiT params {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)
    fae = fc = fs = fp = None
    if args.align == "fae":
        fae, fc, fs, fp = fae_setup(args.fae_ckpt, R, patch); print(f"  REPA align -> FAE per-frame features (T={T}, lam={args.lam})", flush=True)
    ema = copy.deepcopy(model).eval(); [p.requires_grad_(False) for p in ema.parameters()]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def stack(clip): return clip.permute(0, 2, 1, 3, 4).reshape(clip.size(0), ch, R, R)   # (B,3,T,H,W)->(B,T*3,H,W)
    realc = torch.from_numpy(np.stack([tr[i][0].numpy() for i in range(0, min(len(tr), 256))]))
    real = stack(realc).to(DEVICE).view(-1, 3, R, R)
    ref = radial_spectrum(real)

    for ep in range(args.epochs):
        model.train()
        for clip, _ in tl:
            x = stack(clip).to(DEVICE); n = x.size(0); y = torch.zeros(n, dtype=torch.long, device=DEVICE)
            t = torch.rand(n, device=DEVICE); eps = torch.randn_like(x)
            xt = (1 - t)[:, None, None, None] * x + t[:, None, None, None] * eps
            v, zs = model(xt, t, y); loss = F.mse_loss(v, eps - x)
            if args.align == "fae":
                tgt = fae_feats_st(fae, x, fc, fs, fp, T, dyn=args.dyn)
                loss = loss + args.lam * (1 - F.cosine_similarity(F.normalize(zs[0], dim=-1), F.normalize(tgt, dim=-1), dim=-1).mean())
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()): pe.mul_(0.999).add_(pm, alpha=0.001)
        if ep % 20 == 19 or ep == args.epochs - 1:
            g = sample(ema, 128, ch, R).view(-1, 3, R, R)
            sd = (radial_spectrum(g) - ref).abs().mean().item() / ref.abs().mean().item()
            gT = sample(ema, 64, ch, R).view(64, T, 3, R, R)
            dt = (gT[:, 1:] - gT[:, :-1]).abs().mean().item()                  # temporal increment magnitude
            rT = stack(realc[:64]).to(DEVICE).view(-1, T, 3, R, R); dtr = (rT[:, 1:] - rT[:, :-1]).abs().mean().item()
            print(f"  ep {ep+1:3d}/{args.epochs}  loss={loss.item():.4f}  spectrum_dist={sd:.4f}  dt_gen={dt:.3f} (real {dtr:.3f})", flush=True)
    out = f"results/checkpoints/g1/ditst_{args.align}_s{args.seed}.pt"; os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"ema": ema.state_dict(), "args": vars(args)}, out); print(f"DONE saved {out}", flush=True)


if __name__ == "__main__":
    main()
