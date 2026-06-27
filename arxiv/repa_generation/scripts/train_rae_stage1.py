"""RAE stage 1 — train the UNIFORM coordinate decoder on a FROZEN encoder.
Loss = L1 recon + lam_spec * spectral(grad+FFT) + adaptive-weighted PatchGAN (after gan_start).
Config-driven, NO hyperparameter defaults. Saves {decoder, lat_mean, lat_std} for stage 2.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from src.config import load_config
from src.rae import RAE
from src.disc import PatchDisc, hinge_d, hinge_g, spectral_loss, adaptive_gan_weight
from src.utils import set_seed, seed_worker
from src.utils.seed import torch_generator
from scripts.generate import get_frames

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ENC_TAG = {"fae": "fae_tag", "mae": "mae_tag", "jepa": "jepa_tag"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt_out", required=True); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config); set_seed(args.seed)
    if args.smoke:
        cfg.s1_epochs = 2; cfg.gan_start = 1; cfg.gen_n_traj = 2
    R, C, enc = cfg.resolution, cfg.in_chans, cfg.encoder
    print(f"=== RAE-stage1 [{cfg.tag}] enc={enc} res={R} ep={cfg.s1_epochs} gan@{cfg.gan_start} seed={args.seed} ===", flush=True)

    rae = RAE(enc, cfg[ENC_TAG[enc]], 0, in_chans=C, side=R, device=DEV)     # frozen encoder + fresh decoder
    tr = get_frames(cfg.dataset, "train", R, cfg.gen_n_traj)
    tl = DataLoader(tr, batch_size=cfg.s1_batch, shuffle=True, num_workers=4, drop_last=True,
                    worker_init_fn=seed_worker, generator=torch_generator(args.seed))
    stat = torch.stack([tr[i][0] for i in range(min(256, len(tr)))]).to(DEV)
    rae.set_latent_stats(stat)
    print(f"  latent N={rae.encode(stat[:2]).shape[1]} D={rae.D}  decoder params {sum(p.numel() for p in rae.decoder.parameters())/1e6:.2f}M", flush=True)

    disc = PatchDisc(C).to(DEV)
    opt = torch.optim.AdamW(rae.decoder.parameters(), lr=cfg.s1_lr, betas=(0.5, 0.9))
    optd = torch.optim.AdamW(disc.parameters(), lr=cfg.disc_lr, betas=(0.5, 0.9))
    last = rae.decoder.head.weight                                            # for adaptive GAN weight

    for ep in range(cfg.s1_epochs):
        rae.decoder.train()
        agg = {"rec": 0.0, "n": 0}
        for x, _ in tl:
            x = x.to(DEV); z = rae.encode(x)                                  # frozen latent (no grad)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV == "cuda")):
                xhat = rae.decode(z)
            xhat = xhat.float()                                               # losses in fp32 (FFT/disc/grad-balance)
            rec = F.l1_loss(xhat, x); spec = spectral_loss(xhat, x); g = rec + cfg.lam_spec * spec
            if ep >= cfg.gan_start:
                gan = hinge_g(disc(xhat))
                w = adaptive_gan_weight(rec, gan, last)
                g = g + w * gan
            opt.zero_grad(); g.backward(); opt.step()
            if ep >= cfg.gan_start:                                           # discriminator step
                optd.zero_grad(); ld = hinge_d(disc(x), disc(xhat.detach())); ld.backward(); optd.step()
            agg["rec"] += rec.item() * x.size(0); agg["n"] += x.size(0)
        if ep % 10 == 9 or ep == cfg.s1_epochs - 1:
            with torch.no_grad():
                xb, _ = next(iter(tl)); xb = xb.to(DEV); xh = rae.decode(rae.encode(xb))
                rel = (torch.linalg.vector_norm((xh - xb).flatten(1), ord=2, dim=1) /
                       torch.linalg.vector_norm(xb.flatten(1), ord=2, dim=1).clamp_min(1e-6)).mean().item()
            print(f"  ep {ep+1:3d}/{cfg.s1_epochs}  L1={agg['rec']/agg['n']:.4f}  recon_relL2={rel:.4f}", flush=True)

    os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)
    torch.save({"decoder": rae.decoder.state_dict(), "lat_mean": rae.lat_mean, "lat_std": rae.lat_std,
                "encoder": enc, "enc_tag": cfg[ENC_TAG[enc]], "seed": args.seed}, args.ckpt_out)
    print(f"DONE saved {args.ckpt_out}", flush=True)


if __name__ == "__main__":
    main()
