"""Per-channel spectrum_dist for shear generation. Tests whether the POOLED metric is dominated by the
3 smooth/easy channels (pressure, vx, vy) and masks the hard TRACER channel. If FAE wins on tracer but
the pooled metric is carried by the easy channels, that explains why the headline didn't replicate.
Reports a method x channel table (lower=better)."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external/REPA"))
from models.sit import SiT_models
from scripts.gen_dit import radial_spectrum, sample, get_frames, DEVICE

CH = ["tracer", "pressure", "vx", "vy"]


def load(ckpt):
    ck = torch.load(ckpt, map_location=DEVICE); a = ck["args"]; R, C = a["resolution"], 4
    zdim = ck["ema"]["projectors.0.4.weight"].shape[0]
    extra = {"decoder_hidden_size": 384} if "S/" in a["size"] else {}
    m = SiT_models[a["size"]](input_size=R, in_channels=C, num_classes=1, z_dims=[zdim], encoder_depth=4,
                              fused_attn=False, qk_norm=False, **extra).to(DEVICE)
    m.load_state_dict(ck["ema"]); m.eval(); return m, R, C


def sd_channel(gen, real, c):
    rg, rr = radial_spectrum(gen[:, c:c + 1]), radial_spectrum(real[:, c:c + 1])
    return (rg - rr).abs().mean().item() / rr.abs().mean().item()


@torch.no_grad()
def main():
    va = get_frames("shear", "valid", 64)
    real = torch.from_numpy(np.stack([va[i][0].numpy() for i in range(min(len(va), 256))])).to(DEVICE)
    R = real.shape[-1]
    rows = {}
    for nm in ["none", "fae", "mae", "jepa"]:
        m, _, C = load(f"results/checkpoints/g1/dit_shear_{nm}_s0.pt")
        accs = []
        for sd in range(3):                                  # 3 seeds x 512 = 1536 effective, kills single-batch noise
            torch.manual_seed(sd); g = sample(m, 512, C, R)
            accs.append([sd_channel(g, real, c) for c in range(4)] +
                        [(radial_spectrum(g) - radial_spectrum(real)).abs().mean().item() / radial_spectrum(real).abs().mean().item()])
        rows[nm] = np.mean(accs, 0).tolist()
        print(f"  {nm} done  (pooled std over seeds {np.std([a[4] for a in accs]):.4f})", flush=True)
    print("\n  === per-channel spectrum_dist (lower=better) ===")
    print(f"  {'method':6s} " + " ".join(f"{c:>9s}" for c in CH) + f"{'POOLED':>9s}")
    for nm, v in rows.items():
        print(f"  {nm:6s} " + " ".join(f"{x:9.4f}" for x in v[:4]) + f"{v[4]:9.4f}")
    print("\n  best-per-column:")
    for j, c in enumerate(CH + ["POOLED"]):
        best = min(rows, key=lambda k: rows[k][j]); print(f"    {c:9s}: {best} ({rows[best][j]:.4f})")


if __name__ == "__main__":
    main()
