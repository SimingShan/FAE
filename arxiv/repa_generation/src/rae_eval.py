"""Two-axis RAE eval (the rFID != gFID concern made explicit):
  rFID  — decoder ceiling: recon relL2 + reconstruction spectrum_dist (encode->decode held-out).
  gFID  — latent diffusability: spectrum_dist of GENERATED samples (sample latent -> decode).
A sharper decoder can raise rFID without raising gFID — so we report BOTH, separately, per encoder.
"""
import numpy as np, torch
from src.rae import RAE
from src.models.latent_dit import LatentDiT, sample_latent
from scripts.generate import get_frames, spectrum_dist

ENC_TAG = {"fae": "fae_tag", "mae": "mae_tag", "jepa": "jepa_tag"}


@torch.no_grad()
def rae_eval(cfg, stage1_ckpt, stage2_ckpt, device, chunk=128):
    enc, R, C, N_s = cfg.encoder, cfg.resolution, cfg.in_chans, cfg.n_samples
    rae = RAE(enc, cfg[ENC_TAG[enc]], 0, in_chans=C, side=R, device=device)
    s1 = torch.load(stage1_ckpt, map_location=device)
    rae.decoder.load_state_dict(s1["decoder"]); rae.lat_mean.copy_(s1["lat_mean"]); rae.lat_std.copy_(s1["lat_std"]); rae.decoder.eval()

    ref = get_frames(cfg.dataset, cfg.eval_split, R, cfg.gen_n_traj)
    real = torch.from_numpy(np.stack([ref[i][0].numpy() for i in range(min(len(ref), N_s))])).to(device)

    # ---- rFID: reconstruction ----
    rels, recons = [], []
    for i in range(0, len(real), chunk):
        x = real[i:i + chunk]; xh = rae.decode(rae.encode(x))
        rels.append((torch.linalg.norm((xh - x).flatten(1), dim=1) / torch.linalg.norm(x.flatten(1), dim=1).clamp_min(1e-6)))
        recons.append(xh)
    recon_relL2 = float(torch.cat(rels).mean())
    recon_sd = spectrum_dist(torch.cat(recons), real)

    # ---- gFID: generation ----
    s2 = torch.load(stage2_ckpt, map_location=device)
    dit = LatentDiT(s2["D"], depth=s2["depth"], heads=s2["heads"]).to(device); dit.load_state_dict(s2["dit"]); dit.eval()
    gens = []
    for i in range(0, N_s, chunk):
        n = min(chunk, N_s - i)
        z = sample_latent(dit, n, s2["N"], s2["D"], device, steps=cfg.steps)
        gens.append(rae.decode(rae.denormalize(z)))
    gen_sd = spectrum_dist(torch.cat(gens), real)

    return {"recon_relL2": recon_relL2, "recon_spectrum_dist": recon_sd, "gen_spectrum_dist": gen_sd}
