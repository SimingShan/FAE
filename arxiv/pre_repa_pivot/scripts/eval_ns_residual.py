"""Physics residual of generated NS trajectories (channels = smoke, vx, vy). Two parameter-free residuals:
  divergence |dvx/dx + dvy/dy|         -> incompressibility (per frame)
  advection  |d(smoke)/dt + u.grad smoke| -> temporal physical consistency (smoke must be advected by u)
Lower = more physical. Compares none / fae(-dyn) generated vs REAL. Run on the spatio-temporal checkpoints."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external/REPA"))
from models.sit import SiT_models
from scripts.gen_dit_st import sample, DEVICE
from src.data.ns import NSDataset


def load(ckpt):
    ck = torch.load(ckpt, map_location=DEVICE); a = ck["args"]; R, T = a["resolution"], a["frames"]; ch = 3 * T
    zdim = ck["ema"]["projectors.0.4.weight"].shape[0]
    extra = {"decoder_hidden_size": 384} if a["size"].split("/")[0].endswith("S") else {}
    m = SiT_models[a["size"]](input_size=R, in_channels=ch, num_classes=1, z_dims=[zdim], encoder_depth=4,
                              fused_attn=False, qk_norm=False, **extra).to(DEVICE)
    m.load_state_dict(ck["ema"]); m.eval(); return m, R, T, ch


def residuals(traj):                                  # traj:(B,T,3,H,W) smoke=0,vx=1,vy=2
    s, vx, vy = traj[:, :, 0], traj[:, :, 1], traj[:, :, 2]
    div = (torch.gradient(vx, dim=-1)[0] + torch.gradient(vy, dim=-2)[0]).abs().mean().item()
    dsdt = s[:, 1:] - s[:, :-1]                        # (B,T-1,H,W)
    sx = torch.gradient(s, dim=-1)[0]; sy = torch.gradient(s, dim=-2)[0]
    adv = (vx * sx + vy * sy)[:, :-1]                  # frame-t velocity advecting smoke
    advres = (dsdt + adv).abs().mean().item()
    return div, advres


@torch.no_grad()
def gen(ckpt, n=128):
    m, R, T, ch = load(ckpt); torch.manual_seed(1)
    return sample(m, n, ch, R).view(n, T, 3, R, R), R, T


def main():
    out = {}
    for nm, ck in [("none", "results/checkpoints/g1/ditst_none_s0.pt"), ("fae-dyn", "results/checkpoints/g1/ditst_fae_s0.pt")]:
        g, R, T = gen(ck); out[nm] = residuals(g)
        print(f"  generated {nm}: div={out[nm][0]:.4f}  advection_res={out[nm][1]:.4f}", flush=True)
    va = NSDataset("valid", side=R, mode="clip", clip_len=T, frame_stride=4, n_traj=8)
    real = torch.stack([torch.from_numpy(va[i][0].numpy()) for i in range(min(len(va), 128))]).permute(0, 2, 1, 3, 4).to(DEVICE)
    rd, ra = residuals(real)
    print(f"  REAL: div={rd:.4f}  advection_res={ra:.4f}", flush=True)
    print("\n  === physics residual (lower=more physical; compare to REAL) ===")
    print(f"  {'source':10s} {'divergence':>11s} {'advection_res':>14s}")
    for nm in out: print(f"  {nm:10s} {out[nm][0]:>11.4f} {out[nm][1]:>14.4f}")
    print(f"  {'REAL':10s} {rd:>11.4f} {ra:>14.4f}")


if __name__ == "__main__":
    main()
