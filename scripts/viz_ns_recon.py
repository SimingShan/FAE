"""Visualize our FAE reconstructions on NS-2D-conditioned valid clips.
Sparse-sensor encode (frame 0) -> coordinate decode at the full 128x128 grid.
Rows: per sample [GT, ours-recon]; cols: u(smoke), vx, vy. Green dots = sensors the encoder sees.
Inspection only (not a headline metric). Saves results/figs/ns_recon.png."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data.well2d import fields_to_tokens
from src.data.ns import NSDataset
from scripts.eval_ns_probe import load_fae, DEVICE

CKPT = sys.argv[1] if len(sys.argv) > 1 else "results/checkpoints/g1/faep_twoview_fae_ns_tw.pt"
N_SAMPLES, N_SENS = 3, 512
CH = ["u (smoke)", "vx", "vy"]


@torch.no_grad()
def main():
    model, coords, _ = load_fae(CKPT); model.eval()
    R = int(coords.shape[0] ** 0.5)
    g = torch.Generator(device=DEVICE).manual_seed(1)
    iA = torch.randperm(coords.shape[0], generator=g, device=DEVICE)[:N_SENS]
    srow, scol = (iA.cpu().numpy() // R), (iA.cpu().numpy() % R)

    va = NSDataset("valid", side=R, mode="clip", clip_len=2, frame_stride=4, n_traj=4)
    idxs = [int(len(va) * k / N_SAMPLES) for k in range(N_SAMPLES)]
    print(f"valid {len(va)} clips; rendering {idxs} with {N_SENS} sensors", flush=True)

    fig, ax = plt.subplots(2 * N_SAMPLES, 3, figsize=(9, 6 * N_SAMPLES))
    for n, i in enumerate(idxs):
        clip, buo = va[i]
        xt = clip[:, 0].unsqueeze(0).to(DEVICE)                          # (1,3,R,R) frame 0
        tok = model.encode_tokens(fields_to_tokens(xt, iA), coords[iA])
        rec = model.decoder(tok, coords)[0].permute(1, 0).reshape(3, R, R).cpu().numpy()
        gt = xt[0].cpu().numpy()
        for c in range(3):
            v = np.abs(gt[c]).max() + 1e-6
            a0, a1 = ax[2 * n, c], ax[2 * n + 1, c]
            a0.imshow(gt[c], cmap="RdBu_r", vmin=-v, vmax=v)
            a1.imshow(rec[c], cmap="RdBu_r", vmin=-v, vmax=v)
            a1.scatter(scol, srow, s=1.2, c="lime", marker=".", linewidths=0)
            mse = float(((gt[c] - rec[c]) ** 2).mean())
            a0.set_title(f"GT {CH[c]}" + (f"  (buo={float(np.asarray(buo).ravel()[0]):.3f})" if c == 0 else ""), fontsize=9)
            a1.set_title(f"ours recon  nMSE={mse:.3f}", fontsize=9)
            for a in (a0, a1): a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"NS recon — {os.path.basename(CKPT)}  ({N_SENS}/{R*R} sensors, single frame)", fontsize=11)
    fig.tight_layout()
    out = "results/figs/ns_recon.png"; os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110); print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
