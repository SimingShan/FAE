"""Parameter-conditional generation, REPA-style: condition on the PHYSICAL PARAMETER as the 'class label'
via the SiT LabelEmbedder -> AdaLN + classifier-free guidance (the exact mechanism REPA uses for ImageNet
classes -- NOT channel concatenation). Label = buoyancy (NS) / (logRe,logSc) (shear), binned to discrete
classes. --align {none,fae,mae,jepa} keeps the REPA alignment. Tests whether FAE keeps its edge under
PROPER weak/global conditioning (where present-frame conditioning made it redundant). Eval = spectrum_dist
of CFG samples vs real."""
import os, sys, argparse, copy, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external/REPA"))
from torch.utils.data import DataLoader
from models.sit import SiT_models
from src.utils import set_seed, seed_worker
from src.utils.seed import torch_generator
from scripts.gen_dit import radial_spectrum, fae_setup, fae_feats, enc_setup, enc_feats, get_frames, DEVICE


def _key(y):
    y = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
    return tuple(np.round(np.atleast_1d(y).astype(float), 3))


def build_label_map(ds, n=1200):
    idx = np.unique(np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int))
    uniq = sorted({_key(ds[i][1]) for i in idx})
    return {l: j for j, l in enumerate(uniq)}, len(uniq)


def labels_to_cls(yb, lab2cls):
    return torch.tensor([lab2cls.get(_key(y), 0) for y in yb], device=DEVICE, dtype=torch.long)


@torch.no_grad()
def sample_cfg(model, n, C, R, ncls, w=1.5, steps=50):
    """classifier-free guidance: v = v_uncond + w*(v_cond - v_uncond), null class = ncls."""
    x = torch.randn(n, C, R, R, device=DEVICE)
    y = torch.randint(0, ncls, (n,), device=DEVICE); yn = torch.full((n,), ncls, device=DEVICE, dtype=torch.long)
    for i in range(steps):
        t = torch.full((n,), 1 - i / steps, device=DEVICE)
        vc, _ = model(x, t, y); vu, _ = model(x, t, yn)
        x = x - (1 / steps) * (vu + w * (vc - vu))
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ns"); ap.add_argument("--resolution", type=int, default=64)
    ap.add_argument("--size", default="SiT-S/4"); ap.add_argument("--align", choices=["none", "fae", "mae", "jepa"], default="none")
    ap.add_argument("--enc_ckpt", default=""); ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--cfg", type=float, default=1.5); ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default="param")
    ap.add_argument("--fae_ckpt", default="results/checkpoints/g1/faep_twoview_fae_ns_tw.pt")
    args = ap.parse_args(); set_seed(args.seed); R = args.resolution; C = 3 if args.dataset == "ns" else 4
    print(f"=== PARAM-cond DiT [{args.tag}] align={args.align} ds={args.dataset} ({C}ch) cfg={args.cfg} seed={args.seed} ===", flush=True)

    tr = get_frames(args.dataset, "train", R)
    lab2cls, ncls = build_label_map(tr); print(f"  {len(tr)} frames, {ncls} param-classes", flush=True)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True,
                    worker_init_fn=seed_worker, generator=torch_generator(args.seed))
    sz = args.size.split("-")[1].split("/")[0]; patch = int(args.size.split("/")[1])
    extra = {"decoder_hidden_size": 384} if sz == "S" else {}
    grid = R // patch; zdim = 320 if args.align in ("none", "fae") else 256
    model = SiT_models[args.size](input_size=R, in_channels=C, num_classes=ncls, class_dropout_prob=0.1,
                                  z_dims=[zdim], encoder_depth=4, fused_attn=False, qk_norm=False, **extra).to(DEVICE)
    print(f"  SiT params {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)
    fae = fc = fs = fp = enc = None
    if args.align == "fae":
        fae, fc, fs, fp = fae_setup(args.fae_ckpt, R, patch); print(f"  align -> FAE (lam={args.lam})", flush=True)
    elif args.align in ("mae", "jepa"):
        enc = enc_setup(args.align, args.enc_ckpt, R, C); print(f"  align -> {args.align.upper()} (lam={args.lam})", flush=True)
    ema = copy.deepcopy(model).eval(); [p.requires_grad_(False) for p in ema.parameters()]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    real = torch.from_numpy(np.stack([tr[i][0].numpy() for i in range(0, min(len(tr), 512))])).to(DEVICE)
    ref = radial_spectrum(real)

    for ep in range(args.epochs):
        model.train()
        for x, yl in tl:
            x = x.to(DEVICE); n = x.size(0); y = labels_to_cls(yl, lab2cls)
            t = torch.rand(n, device=DEVICE); eps = torch.randn_like(x)
            xt = (1 - t)[:, None, None, None] * x + t[:, None, None, None] * eps
            v, zs = model(xt, t, y); loss = F.mse_loss(v, eps - x)
            if args.align != "none":
                tgt = fae_feats(fae, x, fc, fs, fp) if args.align == "fae" else enc_feats(enc, args.align, x, grid)
                loss = loss + args.lam * (1 - F.cosine_similarity(F.normalize(zs[0], dim=-1), F.normalize(tgt, dim=-1), dim=-1).mean())
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()): pe.mul_(0.999).add_(pm, alpha=0.001)
        if ep % 20 == 19 or ep == args.epochs - 1:
            g = sample_cfg(ema, 256, C, R, ncls, args.cfg)
            sd = (radial_spectrum(g) - ref).abs().mean().item() / ref.abs().mean().item()
            print(f"  ep {ep+1:3d}/{args.epochs}  loss={loss.item():.4f}  spectrum_dist={sd:.4f}", flush=True)
    out = f"results/checkpoints/g1/ditparam_{args.dataset}_{args.align}_s{args.seed}.pt"; os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"ema": ema.state_dict(), "args": vars(args), "ncls": ncls}, out); print(f"DONE saved {out}", flush=True)


if __name__ == "__main__":
    main()
