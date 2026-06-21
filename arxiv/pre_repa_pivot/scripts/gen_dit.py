"""Pixel-space DiT/SiT generation of PDE fields (REPA testbed). Flow-matching (linear interpolant) on
raw fields -- NO VAE. --align {none, fae} adds REPA representation alignment: cosine-align the SiT's
intermediate tokens to a frozen encoder's per-patch features.
  none = the PIXEL-DiT BENCHMARK; fae = align to our physics encoder.
Eval = radial energy-spectrum distance between generated and real fields (physics FID surrogate)."""
import os, sys, argparse, math, copy, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external/REPA"))
from torch.utils.data import DataLoader
from models.sit import SiT_models
from src.data.ns import NSDataset
from src.utils import set_seed, seed_worker
from src.utils.seed import torch_generator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def radial_spectrum(x):                              # x:(B,C,H,W) -> mean radial power
    f = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1)).abs() ** 2
    H, W = x.shape[-2:]; cy, cx = H // 2, W // 2
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    r = torch.sqrt((yy - cy).float() ** 2 + (xx - cx).float() ** 2).long().to(x.device)
    out = torch.zeros(r.max() + 1, device=x.device)
    fm = f.mean((0, 1))
    out.scatter_add_(0, r.flatten(), fm.flatten())
    return out / torch.bincount(r.flatten(), minlength=len(out)).clamp_min(1)


def fae_setup(ckpt, R, patch):
    """frozen FAE + field-grid coords + sensor idx + patch-center coords (REPA target)."""
    from scripts.eval_ns_probe import load_fae
    from src.data.well2d import make_coords_2d
    m, _, _ = load_fae(ckpt); m.eval(); [p.requires_grad_(False) for p in m.parameters()]
    coords = make_coords_2d(R, DEVICE); g = torch.Generator(device=DEVICE).manual_seed(0)
    sidx = torch.randperm(R * R, generator=g, device=DEVICE)[:512]
    pcoords = make_coords_2d(R // patch, DEVICE)            # (R/patch)^2 patch centers
    return m, coords, sidx, pcoords


@torch.no_grad()
def fae_feats(m, fields, coords, sidx, pcoords):
    from src.data.well2d import fields_to_tokens
    lat = m.encode_tokens(fields_to_tokens(fields, sidx), coords[sidx])
    return m.decoder(lat, pcoords, return_feats=True)       # (B, n_patch, 320)


def get_frames(dataset, split, R):
    """single-frame fields for the chosen benchmark (NS 3ch / shear_flow 4ch)."""
    if dataset == "ns":
        from src.data.ns import NSDataset
        return NSDataset(split, side=R, mode="single", clip_len=2, frame_stride=4, n_traj=12)
    from src.data.well2d import ShearFlowSnapshotDataset
    return ShearFlowSnapshotDataset(split, n_seed=24 if split == "train" else 8, frame_stride=12, side=R)


def enc_setup(method, ckpt, R, in_chans=3):
    """frozen MAE/JEPA ViT encoder (trained on the same data) as the REPA alignment target."""
    from scripts.train_baseline import build_model
    m = build_model("mae" if method == "mae" else "ijepa", resolution=R, in_chans=in_chans)
    ck = torch.load(ckpt, map_location=DEVICE); m.load_state_dict(ck["model"]); m.eval()
    [p.requires_grad_(False) for p in m.parameters()]; return m


@torch.no_grad()
def enc_feats(m, method, x, grid):
    """ViT patch tokens -> interpolate to the SiT's grid x grid token layout -> (B, grid^2, 256)."""
    tok = m.forward_encoder(x, 0.0)[0][:, 1:] if method == "mae" else m.target(x)   # (B, N, 256)
    B, N, D = tok.shape; g = int(N ** 0.5)
    t = torch.nn.functional.interpolate(tok.transpose(1, 2).reshape(B, D, g, g), size=(grid, grid),
                                        mode="bilinear", align_corners=False)
    return t.flatten(2).transpose(1, 2)


@torch.no_grad()
def sample(model, n, C, R, steps=50):
    model.eval(); x = torch.randn(n, C, R, R, device=DEVICE); y = torch.zeros(n, dtype=torch.long, device=DEVICE)
    for i in range(steps):                            # Euler ODE from t=1 (noise) -> t=0 (data)
        t = torch.full((n,), 1 - i / steps, device=DEVICE)
        v, _ = model(x, t, y)
        x = x - (1 / steps) * v
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ns"); ap.add_argument("--resolution", type=int, default=64)
    ap.add_argument("--size", default="SiT-S/4"); ap.add_argument("--align", choices=["none", "fae", "mae", "jepa"], default="none")
    ap.add_argument("--enc_ckpt", default="", help="MAE/JEPA encoder checkpoint (for --align mae/jepa)")
    ap.add_argument("--lam", type=float, default=0.5); ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default="dit")
    ap.add_argument("--depth", type=int, default=4, help="SiT encoder_depth = which layer's tokens to align (REPA hp)")
    ap.add_argument("--fae_ckpt", default="results/checkpoints/g1/faep_twoview_fae_ns_tw.pt")
    args = ap.parse_args(); set_seed(args.seed); R = args.resolution; C = 3 if args.dataset == "ns" else 4
    print(f"=== gen DiT [{args.tag}] {args.size} align={args.align} res={R} ds={args.dataset} ({C}ch) seed={args.seed} ===", flush=True)

    tr = get_frames(args.dataset, "train", R)
    print(f"  {len(tr)} frames", flush=True)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True,
                    worker_init_fn=seed_worker, generator=torch_generator(args.seed))
    sz = args.size.split("-")[1].split("/")[0]; patch = int(args.size.split("/")[1])    # S/B/L , patch size
    extra = {"decoder_hidden_size": 384} if sz == "S" else {}      # S factory omits it; B/L set it themselves
    grid = R // patch                                              # SiT token grid side
    zdim = 320 if args.align in ("none", "fae") else 256           # FAE dec_dim / MAE-JEPA embed_dim
    model = SiT_models[args.size](input_size=R, in_channels=C, num_classes=1, z_dims=[zdim], encoder_depth=args.depth,
                                  fused_attn=False, qk_norm=False, **extra).to(DEVICE)
    print(f"  SiT params {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)
    fae = fcoords = fsidx = fpcoords = enc = None
    if args.align == "fae":
        fae, fcoords, fsidx, fpcoords = fae_setup(args.fae_ckpt, R, patch)
        print(f"  REPA align -> FAE per-patch features (lam={args.lam})", flush=True)
    elif args.align in ("mae", "jepa"):
        enc = enc_setup(args.align, args.enc_ckpt, R, C)
        print(f"  REPA align -> {args.align.upper()} patch tokens (lam={args.lam})", flush=True)
    ema = copy.deepcopy(model).eval(); [p.requires_grad_(False) for p in ema.parameters()]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    # real-data reference spectrum
    real = torch.from_numpy(np.stack([tr[i][0].numpy() for i in range(0, min(len(tr), 512))])).to(DEVICE)
    ref_spec = radial_spectrum(real)

    for ep in range(args.epochs):
        model.train()
        for x, _ in tl:
            x = x.to(DEVICE); n = x.size(0); y = torch.zeros(n, dtype=torch.long, device=DEVICE)
            t = torch.rand(n, device=DEVICE); eps = torch.randn_like(x)
            xt = (1 - t)[:, None, None, None] * x + t[:, None, None, None] * eps   # linear interpolant
            target = eps - x                                                       # velocity
            v, zs = model(xt, t, y)
            loss = F.mse_loss(v, target)
            if args.align != "none":                                 # REPA: align SiT tokens to the encoder's per-patch features
                tgt = fae_feats(fae, x, fcoords, fsidx, fpcoords) if args.align == "fae" else enc_feats(enc, args.align, x, grid)
                loss = loss + args.lam * (1 - F.cosine_similarity(
                    F.normalize(zs[0], dim=-1), F.normalize(tgt, dim=-1), dim=-1).mean())
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()): pe.mul_(0.999).add_(pm, alpha=0.001)
        if ep % 20 == 19 or ep == args.epochs - 1:
            gen = sample(ema, 256, C, R)
            sd = (radial_spectrum(gen) - ref_spec).abs().mean().item() / ref_spec.abs().mean().item()
            print(f"  ep {ep+1:3d}/{args.epochs}  loss={loss.item():.4f}  spectrum_dist={sd:.4f}", flush=True)
    out = f"results/checkpoints/g1/dit_{args.dataset}_{args.align}_s{args.seed}.pt"; os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"ema": ema.state_dict(), "args": vars(args)}, out); print(f"DONE saved {out}", flush=True)


if __name__ == "__main__":
    main()
