"""Spatio-temporal generation viz: real vs none vs fae trajectories. Makes a filmstrip PNG
(rows=real/none/fae, cols=frames) and a looping GIF — to SEE the over-smoothing (generated rows
barely change across frames while real evolves). results/figs/ditst_filmstrip.png + ditst.gif"""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external/REPA"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.animation as anim
from models.sit import SiT_models
from scripts.gen_dit_st import sample, DEVICE
from src.data.ns import NSDataset


def load(ckpt):
    ck = torch.load(ckpt, map_location=DEVICE); a = ck["args"]; R, T = a["resolution"], a["frames"]; ch = 3 * T
    zdim = ck["ema"]["projectors.0.4.weight"].shape[0]
    extra = {"decoder_hidden_size": 384} if a["size"].split("/")[0].endswith("S") else {}
    m = SiT_models[a["size"]](input_size=R, in_channels=ch, num_classes=1, z_dims=[zdim], encoder_depth=4,
                              fused_attn=False, qk_norm=False, **extra).to(DEVICE)
    m.load_state_dict(ck["ema"]); m.eval()
    return m, R, T, ch


@torch.no_grad()
def gen_traj(ckpt, seed=3):
    m, R, T, ch = load(ckpt); torch.manual_seed(seed)
    g = sample(m, 1, ch, R)[0].view(T, 3, R, R).cpu().numpy()      # (T,3,R,R)
    return g, R, T


def main():
    gn, R, T = gen_traj("results/checkpoints/g1/ditst_none_s0.pt")
    gf, _, _ = gen_traj("results/checkpoints/g1/ditst_fae_s0.pt")
    va = NSDataset("valid", side=R, mode="clip", clip_len=T, frame_stride=4, n_traj=4)
    real = va[len(va) // 2][0].numpy().transpose(1, 0, 2, 3)        # (T,3,R,R)
    rows = [("real", real), ("none (pixel)", gn), ("fae-dyn", gf)]

    # filmstrip
    fig, ax = plt.subplots(3, T, figsize=(2.1 * T, 6.3))
    for r, (nm, seq) in enumerate(rows):
        v = np.abs(seq[:, 0]).max() + 1e-6
        for t in range(T):
            ax[r, t].imshow(seq[t, 0], cmap="RdBu_r", vmin=-v, vmax=v); ax[r, t].set_xticks([]); ax[r, t].set_yticks([])
            if r == 0: ax[r, t].set_title(f"frame {t+1}", fontsize=9)
        ax[r, 0].set_ylabel(nm, fontsize=11)
    fig.suptitle("NS smoke trajectory — real evolves; generated over-smoothed (barely changes)", fontsize=11)
    fig.tight_layout(); fig.savefig("results/figs/ditst_filmstrip.png", dpi=110); plt.close(fig)
    print("saved results/figs/ditst_filmstrip.png", flush=True)

    # gif
    fig, ax = plt.subplots(1, 3, figsize=(9, 3.3)); ims = []
    for a, (nm, _) in zip(ax, rows): a.set_xticks([]); a.set_yticks([]); a.set_title(nm, fontsize=10)
    vmax = [np.abs(s[:, 0]).max() + 1e-6 for _, s in rows]
    ims = [a.imshow(rows[i][1][0, 0], cmap="RdBu_r", vmin=-vmax[i], vmax=vmax[i], animated=True) for i, a in enumerate(ax)]

    def upd(t):
        for i, im in enumerate(ims): im.set_array(rows[i][1][t % T, 0])
        fig.suptitle(f"NS smoke — frame {t % T + 1}/{T}", fontsize=11); return ims
    fig.tight_layout()
    anim.FuncAnimation(fig, upd, frames=T * 3, interval=400).save("results/figs/ditst.gif", writer=anim.PillowWriter(fps=2))
    print("saved results/figs/ditst.gif", flush=True)


if __name__ == "__main__":
    main()
