"""DINO-style attention map + feature PCA for the trained ViT-DINO benchmark, side-by-side with the FAE.
ViT-DINO attention = last-block self-attention received per patch (the classic DINO map). PCA = patch
features. NOTE: ViT-DINO is patch-16 at 64^2 -> only 4x4 patches (coarse); FAE decodes 16x16 (fine)."""
import os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external/REPA"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benchmarks.jepa.ijepa2d import ViT2D
from scripts.generate import get_frames, DEVICE
from scripts.viz_encoder_pca import pca_rgb, fae_feat
from scripts.viz_fae_attention import fae_saliency


@torch.no_grad()
def vit_dino_maps(ckpt, field, R):
    ck = torch.load(ckpt, map_location=DEVICE); a = ck["train_args"]
    m = ViT2D(img_size=R, patch_size=a.get("patch", 16), in_chans=3, embed_dim=a["embed_dim"], depth=a["depth"], num_heads=8).to(DEVICE)
    m.load_state_dict(ck["model"]); m.eval()
    blk = m.blocks[-1]; cap = {}
    h = blk.attn.register_forward_pre_hook(lambda mod, inp: cap.__setitem__("x", inp[0]))
    feats = m(field)                                                   # (1, N, D) patch tokens
    h.remove()
    x = cap["x"]; B, N, Cd = x.shape; Hh = blk.attn.num_heads
    qkv = blk.attn.qkv(x).reshape(B, N, 3, Hh, Cd // Hh).permute(2, 0, 3, 1, 4)
    q, k = qkv[0], qkv[1]
    att = ((q @ k.transpose(-2, -1)) * blk.attn.scale).softmax(-1)     # (B, H, N, N)
    g = int(round(N ** 0.5))
    sal = att.mean(1).mean(1)[0].reshape(g, g).cpu().numpy()           # attention received per patch
    return (sal - sal.min()) / (sal.max() - sal.min() + 1e-9), pca_rgb(feats[0])


def main():
    R = 64; va = get_frames("ns", "valid", R)
    field = torch.from_numpy(va[len(va) // 2][0].numpy()).unsqueeze(0).to(DEVICE)
    smoke = field[0, 0].cpu().numpy()
    dsal, dpca = vit_dino_maps("results/checkpoints/g1/vit_dino_ns_p4.pt", field, R)
    fsal = fae_saliency("results/checkpoints/g1/faep_twoview_fae_ns_tw.pt", field, R)
    fsal = (fsal - fsal.min()) / (fsal.max() - fsal.min() + 1e-9)
    fpca = pca_rgb(fae_feat(field, R))
    fig, ax = plt.subplots(2, 3, figsize=(10.5, 7.2))
    rows = [("ViT-DINO patch-4 (16x16)", smoke, dsal, dpca), ("FAE recon (16x16)", smoke, fsal, fpca)]
    for r, (nm, fld, sal, pca) in enumerate(rows):
        ax[r, 0].imshow(fld, cmap="RdBu_r"); ax[r, 0].set_ylabel(nm, fontsize=11)
        ax[r, 1].imshow(sal, cmap="inferno", interpolation="nearest")
        ax[r, 2].imshow(pca, interpolation="nearest")
    for j, t in enumerate(["field (smoke)", "attention map", "feature PCA"]): ax[0, j].set_title(t, fontsize=12)
    for a in ax.flat: a.set_xticks([]); a.set_yticks([])
    fig.suptitle("DINO-style attention + PCA  —  ViT-DINO vs FAE (same field)", fontsize=13)
    fig.tight_layout(); fig.savefig("results/figs/dino_attention_pca.png", dpi=120)
    print("saved results/figs/dino_attention_pca.png", flush=True)


if __name__ == "__main__":
    main()
