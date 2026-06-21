"""Sensor-overlay reconstruction figures (ours / twoview, frame 2 & 15 of the most
diverged clip): (A) two sparse views with sensors as green dots on GT -> invariance;
(B) single-view recon across a sparsity sweep. Saves to results/plots/."""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.models import FAE
from src.data.well2d import ShearFlowClipDataset, make_coords_2d, fields_to_tokens
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("results/plots", exist_ok=True)
CLIP_LEN, STRIDE, T0, T1 = 16, 4, 2, 15
CH = 0


def load(ckpt):
    ck = torch.load(ckpt, map_location="cpu"); R = ck["train_args"]["resolution"]
    m = FAE(emb_dim=320, num_iter=4, depth_per_iter=4, num_latents=128, num_cross_heads=4,
            num_self_heads=8, n_freq=32, max_freq=32, coord_dim=2, in_chans=4).to(DEVICE)
    m.load_state_dict(ck["model"]); m.eval()
    va = ShearFlowClipDataset("valid", n_seed=8, frame_stride=STRIDE, clip_len=CLIP_LEN, side=R, stats=ck["stats"])
    return m, va, R


@torch.no_grad()
def recon(m, x, idx, coords, R):
    tok = m.encode_tokens(fields_to_tokens(x, idx), coords[idx])
    rec = m.decoder(tok, coords)[0].permute(1, 0).reshape(4, R, R).cpu()
    return rec, m.represent(tok)


def dots(ax, idx, R):
    ii = (idx.cpu().numpy() // R); jj = (idx.cpu().numpy() % R)
    ax.scatter(jj, ii, c="lime", s=3, alpha=0.8, linewidths=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/checkpoints/g1/faep_twoview_tvb_s0.pt")
    ap.add_argument("--tag", default="twoview")
    args = ap.parse_args()
    TAG = args.tag
    m, va, R = load(args.ckpt); NPIX = R * R
    coords = make_coords_2d(R, DEVICE)
    best, bidx = -1.0, 0
    for i in range(0, len(va), max(1, len(va) // 40)):
        c = va[i][0]; d = float(((c[:, T0] - c[:, T1]) ** 2).mean())
        if d > best: best, bidx = d, i
    clip = va[bidx][0].to(DEVICE); print(f"clip {bidx} gap={best:.3f}")
    g = torch.Generator(device=DEVICE).manual_seed(0)
    mse = lambda a, b: float(((a - b) ** 2).mean())

    # ---- (A) two-view invariance with sensor dots, frame T1 ----
    x = clip[:, T1].unsqueeze(0); gt = x[0, CH].cpu()
    iA = torch.randperm(NPIX, generator=g, device=DEVICE)[:256]
    iB = torch.randperm(NPIX, generator=g, device=DEVICE)[:512]
    rA, pA = recon(m, x, iA, coords, R); rB, pB = recon(m, x, iB, coords, R)
    cos = float(F.cosine_similarity(pA, pB, dim=-1))
    f, ax = plt.subplots(1, 4, figsize=(13, 3.5))
    for a in ax: a.axis("off")
    ax[0].imshow(gt, cmap="RdBu_r"); dots(ax[0], iA, R); ax[0].set_title("GT + 256 sensors", fontsize=9)
    ax[1].imshow(rA[CH], cmap="RdBu_r"); ax[1].set_title(f"recon (256)\nMSE={mse(rA[CH],gt):.3f}", fontsize=9)
    ax[2].imshow(gt, cmap="RdBu_r"); dots(ax[2], iB, R); ax[2].set_title("GT + 512 sensors", fontsize=9)
    ax[3].imshow(rB[CH], cmap="RdBu_r"); ax[3].set_title(f"recon (512)\nMSE={mse(rB[CH],gt):.3f}", fontsize=9)
    f.suptitle(f"two sparse views of frame {T1} -> same recon  (rep cosine={cos:.3f})", fontsize=11)
    f.tight_layout(); f.savefig(f"results/plots/views_invariance_{TAG}.png", dpi=110)
    print("  saved results/plots/views_invariance_{TAG}.png")

    # ---- (B) sparsity sweep, frames T0 and T1 ----
    sps = [64, 128, 256, 512, 1024]
    fig, ax = plt.subplots(2, len(sps) + 1, figsize=(3 * (len(sps) + 1), 6.2))
    for a in ax.ravel(): a.axis("off")
    for r, T in enumerate([T0, T1]):
        x = clip[:, T].unsqueeze(0); gt = x[0, CH].cpu()
        ax[r, 0].imshow(gt, cmap="RdBu_r"); ax[r, 0].set_title(f"GT frame {T}", fontsize=9)
        for k, n in enumerate(sps):
            idx = torch.randperm(NPIX, generator=torch.Generator(device=DEVICE).manual_seed(k), device=DEVICE)[:n]
            rec, _ = recon(m, x, idx, coords, R)
            ax[r, k + 1].imshow(rec[CH], cmap="RdBu_r")
            ax[r, k + 1].set_title(f"{n} sensors\nMSE={mse(rec[CH],gt):.3f}", fontsize=9)
    fig.suptitle("single-view reconstruction vs sensor density (top: frame 2, bottom: frame 15)", fontsize=11)
    fig.tight_layout(); fig.savefig(f"results/plots/views_sparsity_{TAG}.png", dpi=110)
    print("  saved results/plots/views_sparsity_{TAG}.png")


if __name__ == "__main__":
    main()
