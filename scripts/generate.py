"""REPA generation — ONE of the two evaluation categories (the other is eval_probe.py).

Pixel-space SiT flow-matching on PDE fields (NO VAE). Representation alignment (REPA): cosine-align the
SiT's intermediate tokens to a frozen encoder's per-patch features.

  --mode  {uncond, param, sparse}     conditioning regime
      uncond : unconditional generation.                          metric: spectrum_dist
      param  : condition on the physical PARAMETER as a class via the SiT LabelEmbedder + AdaLN + CFG
               (REPA's ImageNet-class mechanism; label binned from buoyancy / Re,Sc). metric: spectrum_dist
      sparse : condition on a SPARSE observation. The FAE encodes scattered sensors -> a DENSE field guess
               (a capability fixed-grid ViTs lack); the DiT refines it.   metric: forecast/recon relL2
  --align {none, fae, mae, jepa}      REPA alignment target (none = pixel-DiT benchmark; ours = fae)
  --dataset {ns, shear}               3-ch NS / 4-ch shear_flow

Eval = radial energy-spectrum distance (physics-FID surrogate) and, for sparse, relative-L2 to the truth.
"""
import os, sys, argparse, copy, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external/REPA"))
from torch.utils.data import DataLoader
from models.sit import SiT_models
from src.utils import set_seed, seed_worker
from src.utils.seed import torch_generator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- metrics + data
def radial_spectrum(x):                              # x:(B,C,H,W) -> mean radial power
    f = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1)).abs() ** 2
    H, W = x.shape[-2:]; cy, cx = H // 2, W // 2
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    r = torch.sqrt((yy - cy).float() ** 2 + (xx - cx).float() ** 2).long().to(x.device)
    out = torch.zeros(r.max() + 1, device=x.device); fm = f.mean((0, 1))
    out.scatter_add_(0, r.flatten(), fm.flatten())
    return out / torch.bincount(r.flatten(), minlength=len(out)).clamp_min(1)


def spectrum_dist(gen, real):
    rg, rr = radial_spectrum(gen), radial_spectrum(real)
    return (rg - rr).abs().mean().item() / rr.abs().mean().item()


def get_frames(dataset, split, R):
    if dataset == "ns":
        from src.data.ns import NSDataset
        return NSDataset(split, side=R, mode="single", clip_len=2, frame_stride=4, n_traj=12)
    from src.data.well2d import ShearFlowSnapshotDataset
    return ShearFlowSnapshotDataset(split, n_seed=24 if split == "train" else 8, frame_stride=12, side=R)


# ---------------------------------------------------------------- REPA alignment targets
def fae_setup(ckpt, R, patch):
    from scripts.eval_ns_probe import load_fae
    from src.data.well2d import make_coords_2d
    m, _, _ = load_fae(ckpt); m.eval(); [p.requires_grad_(False) for p in m.parameters()]
    coords = make_coords_2d(R, DEVICE); g = torch.Generator(device=DEVICE).manual_seed(0)
    sidx = torch.randperm(R * R, generator=g, device=DEVICE)[:512]
    return m, coords, sidx, make_coords_2d(R // patch, DEVICE)


@torch.no_grad()
def fae_feats(m, fields, coords, sidx, pcoords):
    from src.data.well2d import fields_to_tokens
    lat = m.encode_tokens(fields_to_tokens(fields, sidx), coords[sidx])
    return m.decoder(lat, pcoords, return_feats=True)                # (B, n_patch, dec_dim)


@torch.no_grad()
def fae_dense(m, fields, coords, sidx):
    """FAE reconstructs the FULL grid from `sidx` scattered sensors -> dense conditioning field (sparse mode)."""
    from src.data.well2d import fields_to_tokens
    lat = m.encode_tokens(fields_to_tokens(fields, sidx), coords[sidx])
    B, C, H, W = fields.shape
    return m.decoder(lat, coords).reshape(B, H, W, C).permute(0, 3, 1, 2)


def enc_setup(method, ckpt, R, in_chans):
    from scripts.train_baseline import build_model
    m = build_model("mae" if method == "mae" else "ijepa", resolution=R, in_chans=in_chans)
    ck = torch.load(ckpt, map_location=DEVICE); m.load_state_dict(ck["model"]); m.eval()
    [p.requires_grad_(False) for p in m.parameters()]; return m


@torch.no_grad()
def enc_feats(m, method, x, grid):
    tok = m.forward_encoder(x, 0.0)[0][:, 1:] if method == "mae" else m.target(x)
    B, N, D = tok.shape; g = int(N ** 0.5)
    t = F.interpolate(tok.transpose(1, 2).reshape(B, D, g, g), size=(grid, grid), mode="bilinear", align_corners=False)
    return t.flatten(2).transpose(1, 2)


def align_target(args, x, fae, fc, fs, fp, enc, grid):
    if args.align == "fae":
        return fae_feats(fae, x, fc, fs, fp)
    return enc_feats(enc, args.align, x, grid)


# ---------------------------------------------------------------- label map (param mode)
def _key(y):
    y = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
    return tuple(np.round(np.atleast_1d(y).astype(float), 3))


def build_label_map(ds, n=1200):
    idx = np.unique(np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int))
    uniq = sorted({_key(ds[i][1]) for i in idx})
    return {l: j for j, l in enumerate(uniq)}, len(uniq)


def labels_to_cls(yb, m):
    return torch.tensor([m.get(_key(y), 0) for y in yb], device=DEVICE, dtype=torch.long)


# ---------------------------------------------------------------- samplers
@torch.no_grad()
def sample(model, n, C, R, y=None, cond=None, ncls=None, cfg=1.0, steps=50):
    """Euler ODE t:1->0. y=class indices (param), cond=extra cond channels (sparse), cfg>1 -> guidance."""
    x = torch.randn(n, C, R, R, device=DEVICE)
    if y is None: y = torch.zeros(n, dtype=torch.long, device=DEVICE)
    yn = torch.full((n,), ncls, dtype=torch.long, device=DEVICE) if (cfg > 1 and ncls) else None
    for i in range(steps):
        t = torch.full((n,), 1 - i / steps, device=DEVICE)
        inp = x if cond is None else torch.cat([x, cond], 1)
        v, _ = model(inp, t, y)
        if yn is not None:
            vu, _ = model(inp, t, yn); v = vu + cfg * (v[:, :C] - vu[:, :C])
        x = x - (1 / steps) * v[:, :C]
    return x


# ---------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["uncond", "param", "sparse"], default="uncond")
    ap.add_argument("--align", choices=["none", "fae", "mae", "jepa"], default="none")
    ap.add_argument("--dataset", default="ns"); ap.add_argument("--resolution", type=int, default=64)
    ap.add_argument("--size", default="SiT-S/4"); ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--lam", type=float, default=0.5); ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--n_sensors", type=int, default=256, help="sparse-mode observed sensors")
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="gen"); ap.add_argument("--enc_ckpt", default="")
    ap.add_argument("--fae_ckpt", default="results/checkpoints/g1/faep_twoview_fae_ns_tw.pt")
    args = ap.parse_args(); set_seed(args.seed)
    R = args.resolution; C = 3 if args.dataset == "ns" else 4
    print(f"=== generate [{args.tag}] mode={args.mode} align={args.align} ds={args.dataset}({C}ch) "
          f"size={args.size} d={args.depth} seed={args.seed} ===", flush=True)

    tr = get_frames(args.dataset, "train", R)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True,
                    worker_init_fn=seed_worker, generator=torch_generator(args.seed))
    sz = args.size.split("-")[1].split("/")[0]; patch = int(args.size.split("/")[1]); grid = R // patch
    extra = {"decoder_hidden_size": 384} if sz == "S" else {}
    zdim = 320 if args.align in ("none", "fae") else 256
    in_ch = 2 * C if args.mode == "sparse" else C                    # sparse concats the FAE dense guess
    ncls = 1
    if args.mode == "param":
        lab2cls, ncls = build_label_map(tr); print(f"  {ncls} param-classes", flush=True)
    model = SiT_models[args.size](input_size=R, in_channels=in_ch, num_classes=ncls, class_dropout_prob=0.1,
                                  z_dims=[zdim], encoder_depth=args.depth, fused_attn=False, qk_norm=False, **extra).to(DEVICE)
    print(f"  SiT params {sum(p.numel() for p in model.parameters())/1e6:.1f}M  (in_ch={in_ch})", flush=True)

    # alignment + sparse-conditioning encoders (FAE is needed for sparse regardless of --align)
    fae = fc = fs = fp = enc = None
    need_fae = args.align == "fae" or args.mode == "sparse"
    if need_fae:
        fae, fc, fs, fp = fae_setup(args.fae_ckpt, R, patch)
    if args.align in ("mae", "jepa"):
        enc = enc_setup(args.align, args.enc_ckpt, R, C)
    if args.align != "none":
        print(f"  REPA align -> {args.align} (lam={args.lam})", flush=True)

    ema = copy.deepcopy(model).eval(); [p.requires_grad_(False) for p in ema.parameters()]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    real = torch.from_numpy(np.stack([tr[i][0].numpy() for i in range(min(len(tr), 512))])).to(DEVICE)
    ref = radial_spectrum(real)
    g0 = torch.Generator(device=DEVICE).manual_seed(1)
    ssub = torch.randperm(R * R, generator=g0, device=DEVICE)[:args.n_sensors] if args.mode == "sparse" else None

    for ep in range(args.epochs):
        model.train()
        for x, yl in tl:
            x = x.to(DEVICE); n = x.size(0)
            y = labels_to_cls(yl, lab2cls) if args.mode == "param" else torch.zeros(n, dtype=torch.long, device=DEVICE)
            cond = fae_dense(fae, x, fc, ssub) if args.mode == "sparse" else None
            t = torch.rand(n, device=DEVICE); eps = torch.randn_like(x)
            xt = (1 - t)[:, None, None, None] * x + t[:, None, None, None] * eps
            inp = xt if cond is None else torch.cat([xt, cond], 1)
            v, zs = model(inp, t, y); loss = F.mse_loss(v[:, :C], eps - x)
            if args.align != "none":
                tgt = align_target(args, x, fae, fc, fs, fp, enc, grid)
                loss = loss + args.lam * (1 - F.cosine_similarity(F.normalize(zs[0], dim=-1), F.normalize(tgt, dim=-1), dim=-1).mean())
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()): pe.mul_(0.999).add_(pm, alpha=0.001)
        if ep % 20 == 19 or ep == args.epochs - 1:
            if args.mode == "sparse":
                idx = torch.randperm(len(tr))[:256]
                xv = torch.from_numpy(np.stack([tr[int(i)][0].numpy() for i in idx])).to(DEVICE)
                cond = fae_dense(fae, xv, fc, ssub); g = sample(ema, xv.size(0), C, R, cond=cond)
                rel = (torch.linalg.norm((g - xv).flatten(1), dim=1) / torch.linalg.norm(xv.flatten(1), dim=1).clamp_min(1e-6)).mean().item()
                print(f"  ep {ep+1:3d}/{args.epochs}  loss={loss.item():.4f}  recon_relL2={rel:.4f}  spectrum_dist={spectrum_dist(g, xv):.4f}", flush=True)
            else:
                yc = torch.randint(0, ncls, (256,), device=DEVICE) if args.mode == "param" else None
                g = sample(ema, 256, C, R, y=yc, ncls=ncls if args.mode == "param" else None, cfg=args.cfg if args.mode == "param" else 1.0)
                print(f"  ep {ep+1:3d}/{args.epochs}  loss={loss.item():.4f}  spectrum_dist={spectrum_dist(g, real):.4f}", flush=True)
    out = f"results/checkpoints/g1/gen_{args.dataset}_{args.mode}_{args.align}_s{args.seed}.pt"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"ema": ema.state_dict(), "args": vars(args), "ncls": ncls}, out); print(f"DONE saved {out}", flush=True)


if __name__ == "__main__":
    main()
