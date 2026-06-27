"""Visualize DiT-generated PDE fields vs real. Loads a trained gen_dit checkpoint (EMA), ODE-samples,
plots real (top) vs generated (bottom) for the first channel. results/figs/dit_samples_<tag>.png"""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external/REPA"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from models.sit import SiT_models
from scripts.generate import sample, DEVICE
from src.data.ns import NSDataset

CKPT = sys.argv[1] if len(sys.argv) > 1 else "results/checkpoints/g1/dit_ns_none_s0.pt"
N = 8


def main():
    ck = torch.load(CKPT, map_location=DEVICE); a = ck["args"]; R, C = a["resolution"], 3
    sz = a["size"].split("-")[1].split("/")[0]; extra = {"decoder_hidden_size": 384} if sz == "S" else {}
    zdim = ck["ema"]["projectors.0.4.weight"].shape[0]      # match the checkpoint's projector dim (640 old / 320 new)
    model = SiT_models[a["size"]](input_size=R, in_channels=C, num_classes=1, z_dims=[zdim], encoder_depth=4,
                                  fused_attn=False, qk_norm=False, **extra).to(DEVICE)
    model.load_state_dict(ck["ema"]); model.eval()
    print(f"loaded {os.path.basename(CKPT)}  align={a.get('align')}  res={R}", flush=True)

    gen = sample(model, N, C, R).cpu().numpy()
    va = NSDataset("valid", side=R, mode="single", clip_len=2, frame_stride=4, n_traj=4)
    real = np.stack([va[int(len(va) * k / N)][0].numpy() for k in range(N)])

    fig, ax = plt.subplots(2, N, figsize=(2 * N, 4.2))
    for k in range(N):
        v = np.abs(real[k, 0]).max() + 1e-6
        ax[0, k].imshow(real[k, 0], cmap="RdBu_r", vmin=-v, vmax=v)
        ax[1, k].imshow(gen[k, 0], cmap="RdBu_r", vmin=-v, vmax=v)
        for r in (0, 1): ax[r, k].set_xticks([]); ax[r, k].set_yticks([])
    ax[0, 0].set_ylabel("REAL", fontsize=11); ax[1, 0].set_ylabel("GENERATED", fontsize=11)
    fig.suptitle(f"NS smoke — DiT samples (align={a.get('align')}, res={R})", fontsize=12)
    fig.tight_layout(); out = f"results/figs/dit_samples_{a.get('align','x')}.png"; os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110); print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
