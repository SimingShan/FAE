"""RAE stage 2 — flow-matching DiT IN the frozen encoder latent (linear-path velocity, like generate.py).
Loads the stage-1 decoder + latent stats. Config-driven. Saves the latent DiT for the gFID eval.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from src.config import load_config
from src.rae import RAE
from src.models.latent_dit import LatentDiT, sample_latent
from src.utils import set_seed, seed_worker
from src.utils.seed import torch_generator
from scripts.generate import get_frames, spectrum_dist, radial_spectrum

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ENC_TAG = {"fae": "fae_tag", "mae": "mae_tag", "jepa": "jepa_tag"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt_out", required=True); ap.add_argument("--stage1_ckpt", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config); set_seed(args.seed)
    if args.smoke:
        cfg.s2_epochs = 2; cfg.gen_n_traj = 2
    R, C, enc = cfg.resolution, cfg.in_chans, cfg.encoder
    print(f"=== RAE-stage2 [{cfg.tag}] enc={enc} res={R} ep={cfg.s2_epochs} depth={cfg.dit_depth} seed={args.seed} ===", flush=True)

    rae = RAE(enc, cfg[ENC_TAG[enc]], 0, in_chans=C, side=R, device=DEV)
    s1 = torch.load(args.stage1_ckpt, map_location=DEV)
    rae.decoder.load_state_dict(s1["decoder"]); rae.lat_mean.copy_(s1["lat_mean"]); rae.lat_std.copy_(s1["lat_std"])
    rae.decoder.eval()

    tr = get_frames(cfg.dataset, "train", R, cfg.gen_n_traj)
    tl = DataLoader(tr, batch_size=cfg.s2_batch, shuffle=True, num_workers=4, drop_last=True,
                    worker_init_fn=seed_worker, generator=torch_generator(args.seed))
    N = rae.encode(torch.stack([tr[i][0] for i in range(2)]).to(DEV)).shape[1]
    real = torch.from_numpy(np.stack([tr[i][0].numpy() for i in range(min(len(tr), 512))])).to(DEV)
    dit = LatentDiT(rae.D, depth=cfg.dit_depth, heads=cfg.dit_heads).to(DEV)
    print(f"  latent N={N} D={rae.D}  DiT params {sum(p.numel() for p in dit.parameters())/1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(dit.parameters(), lr=cfg.s2_lr)

    for ep in range(cfg.s2_epochs):
        dit.train()
        for x, _ in tl:
            z = rae.normalize(rae.encode(x.to(DEV)))                          # frozen, normalized latent
            t = torch.rand(z.size(0), device=DEV); eps = torch.randn_like(z)
            zt = (1 - t)[:, None, None] * z + t[:, None, None] * eps
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV == "cuda")):
                v = dit(zt, t)
            loss = F.mse_loss(v.float(), eps - z)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 25 == 24 or ep == cfg.s2_epochs - 1:
            dit.eval()
            with torch.no_grad():
                zs = sample_latent(dit, 256, N, rae.D, DEV, steps=cfg.steps)
                gen = rae.decode(rae.denormalize(zs))
            print(f"  ep {ep+1:3d}/{cfg.s2_epochs}  loss={loss.item():.4f}  gen_spectrum_dist={spectrum_dist(gen, real):.4f}", flush=True)

    os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)
    torch.save({"dit": dit.state_dict(), "N": N, "D": rae.D, "depth": cfg.dit_depth, "heads": cfg.dit_heads,
                "seed": args.seed}, args.ckpt_out)
    print(f"DONE saved {args.ckpt_out}", flush=True)


if __name__ == "__main__":
    main()
