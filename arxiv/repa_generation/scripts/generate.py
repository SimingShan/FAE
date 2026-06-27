"""REPA generation — ONE of the two evaluation categories (the other is eval_probe.py).

BUILT ON (vendored in external/REPA, imported as `models.sit`):
  REPA — https://github.com/sihyun-yu/REPA   (Yu et al., 2024; representation alignment for generation)
  SiT  — https://github.com/willisma/SiT     (Ma et al., 2024; the diffusion-transformer backbone)
The SiT model is theirs; the PDE modes (uncond/param/sparse), the alignment targets, and the data
wiring below are ours.

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


def get_frames(dataset, split, R, n_traj):
    if dataset == "ns":
        from src.data.ns import NSDataset
        return NSDataset(split, side=R, mode="single", clip_len=2, frame_stride=4, n_traj=n_traj)
    from src.data.well2d import ShearFlowSnapshotDataset
    return ShearFlowSnapshotDataset(split, n_seed=24 if split == "train" else 8, frame_stride=12, side=R)


# ---------------------------------------------------------------- REPA alignment targets
def patch_center_coords(R, patch, device):
    """Query coords at SiT PATCH CENTERS (not grid vertices), in the same i/(R-1) convention as
    make_coords_2d(R), row-major to match the ViT patch raster. Fixes the ~half-patch align offset."""
    g = R // patch
    c = (patch * torch.arange(g, device=device).float() + (patch - 1) / 2) / (R - 1)
    gi, gj = torch.meshgrid(c, c, indexing="ij")
    return torch.stack([gi.flatten(), gj.flatten()], dim=-1)


def fae_setup(ckpt, R, patch):
    from scripts.eval_ns_probe import load_fae
    from src.data.well2d import make_coords_2d
    m, _, _ = load_fae(ckpt); m.eval(); [p.requires_grad_(False) for p in m.parameters()]
    coords = make_coords_2d(R, DEVICE); g = torch.Generator(device=DEVICE).manual_seed(0)
    sidx = torch.randperm(R * R, generator=g, device=DEVICE)[:512]
    return m, coords, sidx, patch_center_coords(R, patch, DEVICE)


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
    ck = torch.load(ckpt, map_location=DEVICE); a = ck.get("train_args", {})
    m = build_model("mae" if method == "mae" else "ijepa", resolution=R, in_chans=in_chans,
                    embed_dim=a.get("embed_dim"), depth=a.get("depth"), patch_size=a.get("patch_size"))
    m.load_state_dict(ck["model"]); m.eval()
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


# ---------------------------------------------------------------- Reverse REPA (align in FAE's SET space)
class ReverseREPAHead(nn.Module):
    """Encode the SiT's per-patch tokens into FAE's 128-token SET via cross-attention with FAE's FROZEN
    learned latents as queries; aligned (token-by-token) to the FAE latent of the clean field. The
    alignment lives in FAE's semantic set space — no grid correspondence, no near-pixel decoder."""
    def __init__(self, fae_latents, dim, heads=8):
        super().__init__()
        self.register_buffer("q", fae_latents.detach().clone())          # (1, 128, dim) frozen FAE queries
        self.nq = nn.LayerNorm(dim); self.nk = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, dit_tokens):                                        # (B, T, dim) -> (B, 128, dim)
        q = self.nq(self.q.expand(dit_tokens.size(0), -1, -1)); kv = self.nk(dit_tokens)
        z, _ = self.attn(q, kv, kv)
        return z + self.mlp(z)


@torch.no_grad()
def fae_set_target(fae, x, coords, sidx):
    """The clean field's FAE latent (128 set tokens) — the Reverse-REPA target. Frozen."""
    from src.data.well2d import fields_to_tokens
    return fae.encode_tokens(fields_to_tokens(x, sidx), coords[sidx])    # (B, 128, emb_dim)


class ReverseREPAHeadV2(nn.Module):
    """Principled Reverse-REPA (per the advice): place the SiT patch tokens at their patch-center coords,
    run them through the FROZEN FAE encoder (frozen coord-embed + frozen pooling + frozen slot queries) ->
    128 set tokens. ONLY a tiny value-adapter trains (DiT-dim -> FAE value-dim), so there is no cheat path
    and the SiT must produce features the frozen FAE genuinely encodes near the clean-field latent."""
    def __init__(self, fae_encoder, dit_dim, patch_coords):
        super().__init__()
        self.enc = fae_encoder                                           # FROZEN (caller froze fae)
        self.val_adapter = nn.Linear(dit_dim, fae_encoder.val_proj.out_features)   # the ONLY trainable part
        self.register_buffer("coords", patch_coords)                     # (T_sit, 2) patch centers

    def forward(self, dit_tokens):                                       # (B, T, dit_dim) -> (B, 128, emb)
        from src.models.fae import fourier_features
        e = self.enc; B = dit_tokens.size(0)
        cf = fourier_features(self.coords.unsqueeze(0).expand(B, -1, -1), e.n_freq, e.max_freq, e.fgeo)
        tokens = torch.cat([e.coord_proj(cf), self.val_adapter(dit_tokens)], dim=-1)   # FAE-input tokens
        q = e.latents.expand(B, -1, -1); q = e.layer_1(q, tokens)
        for _ in range(e.num_iter - 1):
            q = e.layer_n(q, tokens)
        return q


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
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
        for i in range(steps):
            t = torch.full((n,), 1 - i / steps, device=DEVICE)
            inp = x if cond is None else torch.cat([x, cond], 1)
            v, _ = model(inp, t, y); v = v[:, :C]                # velocity is on the target channels only
            if yn is not None:
                vu, _ = model(inp, t, yn); v = vu[:, :C] + cfg * (v - vu[:, :C])
            x = (x - (1 / steps) * v).float()
    return x


@torch.no_grad()
def sample_many(model, total, C, R, y=None, cond=None, ncls=None, cfg=1.0, steps=50, chunk=256):
    """Sample `total` (>=1024 for trustworthy spectrum_dist) in chunks to bound memory."""
    outs = []
    for i in range(0, total, chunk):
        n = min(chunk, total - i)
        yi = y[i:i + n] if torch.is_tensor(y) else None
        ci = cond[i:i + n] if cond is not None else None
        outs.append(sample(model, n, C, R, y=yi, cond=ci, ncls=ncls, cfg=cfg, steps=steps))
    return torch.cat(outs)


# ---------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)        # config controls everything — NO hyperparameter defaults
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt_out", required=True)      # runner sets the organized output path
    ap.add_argument("--smoke", action="store_true")   # tiny wiring test (2 ep / 256 samples / 2 traj)
    args = ap.parse_args()
    from src.config import load_config, ckpt_file
    cfg = load_config(args.config); set_seed(args.seed)
    if args.smoke:
        cfg.gen_epochs = 2; cfg.n_samples = 256; cfg.gen_n_traj = 2
    # EVERY knob from the config (injected into args so the loop stays config-locked)
    args.mode, args.align, args.dataset = cfg.mode, cfg.align, cfg.dataset
    args.resolution, args.size, args.depth = cfg.resolution, cfg.sit_size, cfg.sit_depth
    args.lam, args.cfg, args.n_sensors = cfg.lam, cfg.cfg_scale, cfg.n_sensors
    args.epochs, args.batch, args.lr, args.tag = cfg.gen_epochs, cfg.gen_batch, cfg.gen_lr, cfg.tag
    args.fae_ckpt = ckpt_file("fae", cfg.fae_tag, 0)
    args.enc_ckpt = (ckpt_file("mae", cfg.mae_tag, 0) if cfg.align == "mae"
                     else ckpt_file("jepa", cfg.jepa_tag, 0) if cfg.align == "jepa" else "")
    R = args.resolution; C = cfg.in_chans
    print(f"=== generate [{args.tag}] mode={args.mode} align={args.align} ds={args.dataset}({C}ch) "
          f"res={R} sit={args.size} d={args.depth} seed={args.seed} ===", flush=True)

    tr = get_frames(args.dataset, "train", R, cfg.gen_n_traj)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True,
                    worker_init_fn=seed_worker, generator=torch_generator(args.seed))
    sz = args.size.split("-")[1].split("/")[0]; patch = int(args.size.split("/")[1]); grid = R // patch
    extra = {"decoder_hidden_size": 384} if sz == "S" else {}
    if args.align in ("none", "fae", "fae_set", "fae_set2"):
        zdim = torch.load(args.fae_ckpt, map_location="cpu")["train_args"]["emb_dim"]
    else:                                                            # mae/jepa: the encoder's actual width
        zdim = torch.load(args.enc_ckpt, map_location="cpu")["train_args"]["embed_dim"]
    print(f"  REPA target dim (zdim) = {zdim}", flush=True)
    in_ch = 2 * C if args.mode in ("sparse", "all") else C           # sparse/all concat the FAE dense guess
    ncls = 1
    if args.mode in ("param", "all"):
        lab2cls, ncls = build_label_map(tr); print(f"  {ncls} param-classes", flush=True)
    model = SiT_models[args.size](input_size=R, in_channels=in_ch, num_classes=ncls, class_dropout_prob=0.1,
                                  z_dims=[zdim], encoder_depth=args.depth, fused_attn=False, qk_norm=False, **extra).to(DEVICE)
    print(f"  SiT params {sum(p.numel() for p in model.parameters())/1e6:.1f}M  (in_ch={in_ch})", flush=True)

    # alignment + sparse-conditioning encoders (FAE is needed for sparse regardless of --align)
    fae = fc = fs = fp = enc = repa_head = None
    need_fae = args.align in ("fae", "fae_set", "fae_set2") or args.mode in ("sparse", "all")
    if need_fae:
        fae, fc, fs, fp = fae_setup(args.fae_ckpt, R, patch)
    if args.align in ("mae", "jepa"):
        enc = enc_setup(args.align, args.enc_ckpt, R, C)
    if args.align == "fae_set":                                      # Reverse REPA (naive: trainable pooling)
        repa_head = ReverseREPAHead(fae.encoder.latents, zdim).to(DEVICE)
    elif args.align == "fae_set2":                                   # Reverse REPA (frozen FAE pooling + coord val-adapter)
        repa_head = ReverseREPAHeadV2(fae.encoder, zdim, fp).to(DEVICE)
    if repa_head is not None:
        print(f"  Reverse-REPA head {sum(p.numel() for p in repa_head.parameters() if p.requires_grad)/1e6:.2f}M trainable", flush=True)
    if args.align != "none":
        print(f"  REPA align -> {args.align} (lam={args.lam})", flush=True)

    ema = copy.deepcopy(model).eval(); [p.requires_grad_(False) for p in ema.parameters()]
    params = list(model.parameters()) + ([p for p in repa_head.parameters() if p.requires_grad] if repa_head is not None else [])
    opt = torch.optim.AdamW(params, lr=args.lr)
    last_acos = 0.0                                                  # last alignment cosine (the REPA diagnostic)
    real = torch.from_numpy(np.stack([tr[i][0].numpy() for i in range(min(len(tr), 512))])).to(DEVICE)
    ref = radial_spectrum(real)
    g0 = torch.Generator(device=DEVICE).manual_seed(1)
    ssub = torch.randperm(R * R, generator=g0, device=DEVICE)[:args.n_sensors] if args.mode in ("sparse", "all") else None

    for ep in range(args.epochs):
        model.train()
        for x, yl in tl:
            x = x.to(DEVICE); n = x.size(0)
            if args.mode == "all":                                   # mixed conditioning + per-cond dropout (CFG-style)
                up = (torch.rand(n, device=DEVICE) < 0.45); us = (torch.rand(n, device=DEVICE) < 0.45)
                cls = labels_to_cls(yl, lab2cls)
                y = torch.where(up, cls, torch.full_like(cls, ncls))         # null param-class = ncls
                cond = fae_dense(fae, x, fc, ssub) * us[:, None, None, None].float()
            elif args.mode == "param":
                y = labels_to_cls(yl, lab2cls); cond = None
            elif args.mode == "sparse":
                y = torch.zeros(n, dtype=torch.long, device=DEVICE); cond = fae_dense(fae, x, fc, ssub)
            else:
                y = torch.zeros(n, dtype=torch.long, device=DEVICE); cond = None
            t = torch.rand(n, device=DEVICE); eps = torch.randn_like(x)
            xt = (1 - t)[:, None, None, None] * x + t[:, None, None, None] * eps
            inp = xt if cond is None else torch.cat([xt, cond], 1)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                v, zs = model(inp, t, y); loss = F.mse_loss(v[:, :C], eps - x)
                if args.align in ("fae_set", "fae_set2"):            # Reverse REPA: align in FAE's set space
                    z_fae = fae_set_target(fae, x, fc, fs)            # (B,128,D) frozen FAE latent of clean field
                    z_dit = repa_head(zs[0])                         # (B,128,D) SiT per-patch -> FAE set
                    if args.align == "fae_set2":                     # center -> kill the shared-offset shortcut
                        z_fae = z_fae - z_fae.mean((0, 1), keepdim=True); z_dit = z_dit - z_dit.mean((0, 1), keepdim=True)
                    acos = F.cosine_similarity(F.normalize(z_dit, dim=-1), F.normalize(z_fae, dim=-1), dim=-1).mean()
                    loss = loss + args.lam * (1 - acos); last_acos = acos.item()
                elif args.align != "none":
                    tgt = align_target(args, x, fae, fc, fs, fp, enc, grid)
                    acos = F.cosine_similarity(F.normalize(zs[0], dim=-1), F.normalize(tgt, dim=-1), dim=-1).mean()
                    loss = loss + args.lam * (1 - acos); last_acos = acos.item()
            loss.backward(); opt.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()): pe.mul_(0.999).add_(pm, alpha=0.001)
        ac = ("  align_cos=%.3f" % last_acos) if args.align != "none" else ""
        heavy = (ep % 20 == 19 or ep == args.epochs - 1)
        if args.mode in ("all", "sparse") and not heavy:            # cheap per-epoch line; full multi-task eval @20
            print(f"  ep {ep+1:3d}/{args.epochs}  loss={loss.item():.4f}{ac}", flush=True)
        elif args.mode == "all":
            Z = torch.zeros(256, C, R, R, device=DEVICE); ynull = torch.full((256,), ncls, dtype=torch.long, device=DEVICE)
            gu = sample(ema, 256, C, R, y=ynull, cond=Z)
            gp = sample(ema, 256, C, R, y=torch.randint(0, ncls, (256,), device=DEVICE), cond=Z, ncls=ncls, cfg=args.cfg)
            idx = torch.randperm(len(tr))[:256]; xv = torch.from_numpy(np.stack([tr[int(i)][0].numpy() for i in idx])).to(DEVICE)
            gs = sample(ema, 256, C, R, y=ynull, cond=fae_dense(fae, xv, fc, ssub))
            rel = (torch.linalg.norm((gs - xv).flatten(1), dim=1) / torch.linalg.norm(xv.flatten(1), dim=1).clamp_min(1e-6)).mean().item()
            print(f"  ep {ep+1:3d}/{args.epochs}  loss={loss.item():.4f}  uncond_sd={spectrum_dist(gu, real):.4f}  "
                  f"param_sd={spectrum_dist(gp, real):.4f}  sparse_relL2={rel:.4f}", flush=True)
        elif args.mode == "sparse":
            idx = torch.randperm(len(tr))[:256]
            xv = torch.from_numpy(np.stack([tr[int(i)][0].numpy() for i in idx])).to(DEVICE)
            cond = fae_dense(fae, xv, fc, ssub); g = sample(ema, xv.size(0), C, R, cond=cond)
            rel = (torch.linalg.norm((g - xv).flatten(1), dim=1) / torch.linalg.norm(xv.flatten(1), dim=1).clamp_min(1e-6)).mean().item()
            print(f"  ep {ep+1:3d}/{args.epochs}  loss={loss.item():.4f}  recon_relL2={rel:.4f}  spectrum_dist={spectrum_dist(g, xv):.4f}", flush=True)
        else:                                                        # uncond/param: EVERY epoch (128-sample sd)
            yc = torch.randint(0, ncls, (128,), device=DEVICE) if args.mode == "param" else None
            g = sample(ema, 128, C, R, y=yc, ncls=ncls if args.mode == "param" else None, cfg=args.cfg if args.mode == "param" else 1.0)
            print(f"  ep {ep+1:3d}/{args.epochs}  loss={loss.item():.4f}  spectrum_dist={spectrum_dist(g, real):.4f}{ac}", flush=True)
    # ---- AUTHORITATIVE metric: n_samples (>=1024) on held-out eval_split; stored for the runner ----
    from omegaconf import OmegaConf
    ema.eval(); N = cfg.n_samples
    ref = get_frames(args.dataset, cfg.eval_split, R, cfg.gen_n_traj)
    real_ref = torch.from_numpy(np.stack([ref[i][0].numpy() for i in range(min(len(ref), N))])).to(DEVICE)
    if args.mode == "sparse":
        idx = torch.randperm(len(ref))[:N]
        xv = torch.from_numpy(np.stack([ref[int(i)][0].numpy() for i in idx])).to(DEVICE)
        gen = sample_many(ema, xv.size(0), C, R, cond=fae_dense(fae, xv, fc, ssub), steps=cfg.steps)
        rel = (torch.linalg.norm((gen - xv).flatten(1), dim=1) / torch.linalg.norm(xv.flatten(1), dim=1).clamp_min(1e-6)).mean().item()
        metric = {"recon_relL2": rel, "spectrum_dist": spectrum_dist(gen, real_ref)}
    else:
        yval = (torch.randint(0, ncls, (N,), device=DEVICE) if args.mode == "param"
                else torch.full((N,), ncls, dtype=torch.long, device=DEVICE))
        gen = sample_many(ema, N, C, R, y=yval, ncls=(ncls if args.mode == "param" else None),
                          cfg=(cfg.cfg_scale if args.mode == "param" else 1.0), steps=cfg.steps)
        metric = {"spectrum_dist": spectrum_dist(gen, real_ref)}
    os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)
    torch.save({"ema": ema.state_dict(), "cfg": OmegaConf.to_container(cfg, resolve=True),
                "ncls": ncls, "seed": args.seed, "metric": metric}, args.ckpt_out)
    print(f"FINAL metric={metric}", flush=True); print(f"DONE saved {args.ckpt_out}", flush=True)


if __name__ == "__main__":
    main()
