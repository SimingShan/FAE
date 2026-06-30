"""FAE attention saliency (the DINO-style 'where it looks' map, adapted to a Perceiver).
The FAE has no self-attention; its attention is the 128 latents CROSS-attending the sensors. We hook the
first cross-attention, sum the attention each spatial location receives across all 128 latents -> a
saliency over the field. High = the latents concentrate there. Diagnostic for physics vs interpolation."""
import os, sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external/REPA"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.eval_ns_probe import load_fae
from scripts.generate import get_frames, DEVICE
from src.data.well2d import make_coords_2d, fields_to_tokens


@torch.no_grad()
def fae_saliency(ckpt, field, R):
    m, _, _ = load_fae(ckpt); m = m.to(DEVICE).eval()
    coords = make_coords_2d(n_side=R, device=DEVICE)
    allidx = torch.arange(R * R, device=DEVICE)                     # all pixels as sensors -> dense saliency
    attn_mod = next(mod for _, mod in m.encoder.named_modules() if isinstance(mod, nn.MultiheadAttention))
    store = {}
    def hook(mod, inp, out): store["w"] = out[1]                    # (B, M_latents, N_sensors), head-averaged
    h = attn_mod.register_forward_hook(hook)
    m.encode_tokens(fields_to_tokens(field, allidx), coords[allidx])
    h.remove()
    w = store["w"]
    assert w is not None, "no attention weights captured"
    sal = w[0].sum(0).reshape(R, R).cpu().numpy()                   # total attention each sensor receives
    return sal


def main():
    R = 64; va = get_frames("ns", "valid", R)
    field = torch.from_numpy(va[len(va) // 2][0].numpy()).unsqueeze(0).to(DEVICE)
    sal = fae_saliency("results/checkpoints/g1/faep_twoview_fae_ns_tw.pt", field, R)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-9)
    smoke = field[0, 0].cpu().numpy()
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.8))
    ax[0].imshow(smoke, cmap="RdBu_r"); ax[0].set_title("field (smoke)", fontsize=11)
    ax[1].imshow(sal, cmap="inferno"); ax[1].set_title("FAE latent→sensor attention", fontsize=11)
    ax[2].imshow(smoke, cmap="gray"); ax[2].imshow(sal, cmap="inferno", alpha=0.55)
    ax[2].set_title("overlay", fontsize=11)
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Where the FAE latents attend (recon-FAE) — bright = high attention", fontsize=12)
    fig.tight_layout(); fig.savefig("results/figs/fae_attention.png", dpi=120)
    print("saved results/figs/fae_attention.png", flush=True)


if __name__ == "__main__":
    main()
