"""DINO-paper-style attention: the per-head [CLS]-query attention map (the figure that segments the
bird/dog in the DINO paper). Our ViT2D is mean-pooled with NO [CLS] token, so we form a CLS-PROXY
query = the global pooled token, run it against the last-layer patch keys, and show ALL heads
separately (in DINO different heads segment different object parts). Smoothly upsampled + overlaid on
the field. On NS smoke the 'object' is the coherent plume / shear structure, not a bird."""
import os, sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benchmarks.jepa.ijepa2d import ViT2D
from scripts.generate import get_frames, DEVICE


@torch.no_grad()
def cls_head_attention(ckpt, field, R):
    ck = torch.load(ckpt, map_location=DEVICE); a = ck["train_args"]; P = a.get("patch", 16)
    m = ViT2D(img_size=R, patch_size=P, in_chans=3, embed_dim=a["embed_dim"], depth=a["depth"], num_heads=8).to(DEVICE)
    m.load_state_dict(ck["model"]); m.eval()
    blk = m.blocks[-1]; cap = {}
    h = blk.attn.register_forward_pre_hook(lambda mod, inp: cap.__setitem__("x", inp[0]))
    m(field); h.remove()
    x = cap["x"]; B, N, Cd = x.shape; H = blk.attn.num_heads; d = Cd // H
    xcls = x.mean(1, keepdim=True)                                            # CLS-proxy = global pooled token
    qkv = blk.attn.qkv(torch.cat([xcls, x], 1)).reshape(B, N + 1, 3, H, d).permute(2, 0, 3, 1, 4)
    q, k = qkv[0], qkv[1]
    att = ((q[:, :, :1] @ k[:, :, 1:].transpose(-2, -1)) * blk.attn.scale).softmax(-1)  # (B,H,1,N): CLS->patches
    g = int(round(N ** 0.5))
    maps = att[0, :, 0].reshape(H, g, g)                                      # (H, g, g)
    return F.interpolate(maps[None], size=(R, R), mode="bilinear", align_corners=False)[0].cpu().numpy()


def main():
    R = 64; va = get_frames("ns", "valid", R)
    field = torch.from_numpy(va[len(va) // 2][0].numpy()).unsqueeze(0).to(DEVICE)
    smoke = field[0, 0].cpu().numpy()
    A = cls_head_attention("results/checkpoints/g1/vit_dino_ns_p4.pt", field, R)   # (8, R, R)
    H = A.shape[0]
    fig, ax = plt.subplots(3, 3, figsize=(9, 9)); ax = ax.flat
    ax[0].imshow(smoke, cmap="RdBu_r"); ax[0].set_title("field (smoke)", fontsize=11)
    for hh in range(H):
        a = A[hh]; a = (a - a.min()) / (a.max() - a.min() + 1e-9)
        ax[hh + 1].imshow(smoke, cmap="gray", alpha=0.45)
        ax[hh + 1].imshow(a, cmap="inferno", alpha=0.6)
        ax[hh + 1].set_title(f"head {hh}", fontsize=11)
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.suptitle("DINO-style [CLS]-proxy attention, per head  —  ViT-DINO patch-4 (NS smoke)", fontsize=12)
    fig.tight_layout(); fig.savefig("results/figs/dino_cls_heads.png", dpi=120)
    print("saved results/figs/dino_cls_heads.png", flush=True)


if __name__ == "__main__":
    main()
