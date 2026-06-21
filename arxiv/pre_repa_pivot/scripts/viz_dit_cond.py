"""Conditional generative rollout viz: start from a real frame, AUTOREGRESSIVELY generate the trajectory
(each step conditions on the previous generated frame), animate generated vs ground-truth. Shows the
conditional DiT as a forecaster. Usage: python viz_dit_cond.py <ckpt> [K]. -> results/figs/ditcond_<tag>.gif"""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external/REPA"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.animation as anim
from models.sit import SiT_models
from scripts.gen_dit_cond import sample, DEVICE


def load(ckpt):
    ck = torch.load(ckpt, map_location=DEVICE); a = ck["args"]; R = a["resolution"]; ds = a.get("dataset", "ns"); C = 3 if ds == "ns" else 4
    extra = {"decoder_hidden_size": 384} if "S/" in a["size"] else {}
    m = SiT_models[a["size"]](input_size=R, in_channels=2 * C, num_classes=1, z_dims=[ck["ema"]["projectors.0.4.weight"].shape[0]],
                              encoder_depth=4, fused_attn=False, qk_norm=False, **extra).to(DEVICE)
    m.load_state_dict(ck["ema"]); m.eval(); return m, R, C, ds


@torch.no_grad()
def rollout(ckpt, K=8):
    m, R, C, ds = load(ckpt)
    if ds == "ns":
        from src.data.ns import NSDataset
        d = NSDataset("valid", side=R, mode="clip", clip_len=K, frame_stride=4, n_traj=4)
    else:
        from src.data.well2d import ShearFlowWindowDataset
        d = ShearFlowWindowDataset("valid", n_seed=8, n_frames=K, side=R)
    real = torch.from_numpy(d[len(d) // 2][0].numpy()).to(DEVICE).permute(1, 0, 2, 3)   # (K, C, R, R)
    x = real[0:1]; gen = [x[0]]                                                          # seed = true frame 0
    for _ in range(K - 1):
        x = sample(m, x, C, R); gen.append(x[0])
    return real.cpu().numpy(), torch.stack(gen).cpu().numpy(), ds


def main():
    ckpt = sys.argv[1]; K = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    tag = os.path.basename(ckpt).replace(".pt", "")
    real, gen, ds = rollout(ckpt, K)
    err = np.linalg.norm((gen - real).reshape(K, -1), axis=1) / np.linalg.norm(real.reshape(K, -1), axis=1)
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 3.5))
    for a, t in zip(ax, ["GROUND TRUTH", "GENERATED rollout"]): a.set_xticks([]); a.set_yticks([]); a.set_title(t, fontsize=10)
    v = np.abs(real[:, 0]).max() + 1e-6
    im0 = ax[0].imshow(real[0, 0], cmap="RdBu_r", vmin=-v, vmax=v, animated=True)
    im1 = ax[1].imshow(gen[0, 0], cmap="RdBu_r", vmin=-v, vmax=v, animated=True)

    def upd(t):
        k = t % K; im0.set_array(real[k, 0]); im1.set_array(gen[k, 0])
        fig.suptitle(f"{ds} conditional rollout — step {k} (relL2={err[k]:.3f})", fontsize=11); return im0, im1
    fig.tight_layout()
    out = f"results/figs/ditcond_{tag}.gif"
    anim.FuncAnimation(fig, upd, frames=K * 3, interval=500).save(out, writer=anim.PillowWriter(fps=2))
    print(f"saved {out}  (per-step relL2: {np.round(err, 3).tolist()})", flush=True)


if __name__ == "__main__":
    main()
