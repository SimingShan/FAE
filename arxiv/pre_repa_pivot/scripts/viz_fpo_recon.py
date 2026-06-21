"""FlowBench FPO reconstructions: our FAE (sparse-sensor encode -> coordinate decode), GT vs recon,
3 channels (u, v, p), on UNSEEN-geometry valid sims. Green dots = sensors. results/figs/fpo_recon.png."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data.well2d import fields_to_tokens
from src.data.flowbench import FlowBenchFPO
from scripts.eval_ns_probe import load_fae, DEVICE

CKPT = sys.argv[1] if len(sys.argv) > 1 else "results/checkpoints/g1/faep_twoview_fae_fpo_s0.pt"
N, NSENS, CH = 3, 512, ["u", "v", "p"]


@torch.no_grad()
def main():
    model, coords, _ = load_fae(CKPT); model.eval()
    R = int(coords.shape[0] ** 0.5)
    g = torch.Generator(device=DEVICE).manual_seed(1)
    iA = torch.randperm(coords.shape[0], generator=g, device=DEVICE)[:NSENS]
    srow, scol = iA.cpu().numpy() // R, iA.cpu().numpy() % R
    va = FlowBenchFPO("valid", side=R, mode="clip", clip_len=2, frame_stride=8)
    idxs = [int(len(va) * k / N) for k in range(N)]
    print(f"valid {len(va)} clips (unseen geometries); rendering {idxs}", flush=True)
    fig, ax = plt.subplots(2 * N, 3, figsize=(9, 6 * N))
    for n, i in enumerate(idxs):
        clip, _ = va[i]
        xt = clip[:, 0].unsqueeze(0).to(DEVICE)
        tok = model.encode_tokens(fields_to_tokens(xt, iA), coords[iA])
        rec = model.decoder(tok, coords)[0].permute(1, 0).reshape(3, R, R).cpu().numpy()
        gt = xt[0].cpu().numpy()
        for c in range(3):
            v = np.abs(gt[c]).max() + 1e-6
            a0, a1 = ax[2 * n, c], ax[2 * n + 1, c]
            a0.imshow(gt[c], cmap="RdBu_r", vmin=-v, vmax=v)
            a1.imshow(rec[c], cmap="RdBu_r", vmin=-v, vmax=v)
            a1.scatter(scol, srow, s=1.0, c="lime", marker=".", linewidths=0)
            mse = float(((gt[c] - rec[c]) ** 2).mean())
            a0.set_title(f"GT {CH[c]}", fontsize=9); a1.set_title(f"ours recon nMSE={mse:.3f}", fontsize=9)
            for a in (a0, a1): a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"FlowBench FPO recon (FAE, {NSENS}/{R*R} sensors, single frame, UNSEEN geometry)", fontsize=11)
    fig.tight_layout(); out = "results/figs/fpo_recon.png"; os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110); print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
