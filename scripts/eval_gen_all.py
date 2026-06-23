"""Low-noise eval of a --mode all checkpoint: uncond_sd, param_sd, sparse_relL2 with MANY samples
(averages out the single-batch spectrum noise that makes per-epoch numbers bounce ~0.06).
Usage: python eval_gen_all.py <ckpt.pt> [n_samples=2048]"""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external/REPA"))
from models.sit import SiT_models
from scripts.generate import sample, radial_spectrum, spectrum_dist, fae_setup, fae_dense, get_frames, DEVICE


def load(ckpt):
    ck = torch.load(ckpt, map_location=DEVICE); a = ck["args"]; ncls = ck["ncls"]
    R = a["resolution"]; C = 3 if a["dataset"] == "ns" else 4
    zdim = 320 if a["align"] in ("none", "fae") else 256
    patch = int(a["size"].split("/")[1])
    extra = {"decoder_hidden_size": 384} if "S/" in a["size"] else {}
    m = SiT_models[a["size"]](input_size=R, in_channels=2 * C, num_classes=ncls, class_dropout_prob=0.1,
                              z_dims=[zdim], encoder_depth=a["depth"], fused_attn=False, qk_norm=False, **extra).to(DEVICE)
    m.load_state_dict(ck["ema"]); m.eval()
    return m, a, ncls, R, C, patch


@torch.no_grad()
def main():
    ckpt = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    m, a, ncls, R, C, patch = load(ckpt)
    fae, fc, fs, fp = fae_setup(a["fae_ckpt"], R, patch)
    g0 = torch.Generator(device=DEVICE).manual_seed(1); ssub = torch.randperm(R * R, generator=g0, device=DEVICE)[:a["n_sensors"]]
    tr = get_frames(a["dataset"], "train", R)
    real = torch.from_numpy(np.stack([tr[i][0].numpy() for i in range(min(len(tr), N))])).to(DEVICE)

    def chunks(total, fn, cat=True):
        out = [fn(min(256, total - i), i) for i in range(0, total, 256)]
        return torch.cat(out) if cat else out

    ynull = lambda nb: torch.full((nb,), ncls, dtype=torch.long, device=DEVICE)
    Z = lambda nb: torch.zeros(nb, C, R, R, device=DEVICE)
    gu = chunks(N, lambda nb, i: sample(m, nb, C, R, y=ynull(nb), cond=Z(nb)))
    gp = chunks(N, lambda nb, i: sample(m, nb, C, R, y=torch.randint(0, ncls, (nb,), device=DEVICE), cond=Z(nb), ncls=ncls, cfg=a["cfg"]))
    idx = torch.randperm(len(tr))[:N]; xv = torch.from_numpy(np.stack([tr[int(j)][0].numpy() for j in idx])).to(DEVICE)
    rels = []
    for i in range(0, N, 256):
        xb = xv[i:i + 256]; gb = sample(m, xb.size(0), C, R, y=ynull(xb.size(0)), cond=fae_dense(fae, xb, fc, ssub))
        rels.append(torch.linalg.norm((gb - xb).flatten(1), dim=1) / torch.linalg.norm(xb.flatten(1), dim=1).clamp_min(1e-6))
    print(f"{a['align']:6s}  uncond_sd={spectrum_dist(gu, real):.4f}  param_sd={spectrum_dist(gp, real):.4f}  "
          f"sparse_relL2={torch.cat(rels).mean().item():.4f}   (N={N}, {os.path.basename(ckpt)})", flush=True)


if __name__ == "__main__":
    main()
