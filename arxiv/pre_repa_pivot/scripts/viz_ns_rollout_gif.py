"""NS rollout GIF: autoregressive multi-step prediction from saved rollout models.
Panels: GT | time-only | ours(rep). Animates the smoke field (u) over a held-out trajectory ->
shows whether conditioning on our representation keeps the forecast on track longer.
results/figs/ns_rollout.gif"""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.animation as anim
from scripts.rollout_ns import build_unet, NSStep, compute_reps, RESERVE, DEVICE
import torch.nn as nn

STEPS = 14
FAE = "results/checkpoints/g1/faep_twoview_fae_ns_tw.pt"


def load(mode):
    ck = torch.load(f"results/checkpoints/g1/rollout_ns_{mode}_s0.pt", map_location=DEVICE)
    net = build_unet(); net.load_state_dict(ck["net"]); net.eval()
    rm = None
    if "rep_mlp" in ck:
        rm = nn.Sequential(nn.Linear(640, 1), nn.GELU(), nn.Linear(1, 256)).to(DEVICE)
        rm.load_state_dict(ck["rep_mlp"]); rm.eval()
    return net, rm, ck["stats"]


@torch.no_grad()
def rollout(net, x0, t0, emb_add):
    x, frames = x0.clone(), []
    for k in range(STEPS):
        x = net(x, torch.tensor([float(t0 + k)], device=DEVICE), None, emb_add)
        frames.append(x[0, 0, 0].cpu().numpy())     # smoke channel
    return frames


@torch.no_grad()
def main():
    va = NSStep("valid", n_traj_per_file=4)
    si = 0; traj = va.traj[si]                       # (T,3,H,W) fp16 normalized
    t0 = RESERVE
    x0 = torch.from_numpy(traj[t0].astype(np.float32)).to(DEVICE)[None, None]   # (1,1,3,H,W)
    gt = [traj[t0 + 1 + k].astype(np.float32)[0] for k in range(STEPS)]

    net_t, _, _ = load("time")
    net_r, rm, _ = load("rep")
    rep = compute_reps([va.head[si]], FAE)           # (1,640)
    pred_t = rollout(net_t, x0, t0, None)
    pred_r = rollout(net_r, x0, t0, rm(rep))

    v = np.abs(gt[0]).max() + 1e-6
    fig, ax = plt.subplots(1, 3, figsize=(9, 3.2))
    titles = ["ground truth", "time-only", "ours (FAE rep)"]
    ims = []
    for a, t in zip(ax, titles):
        a.set_xticks([]); a.set_yticks([]); a.set_title(t, fontsize=10)
        ims.append(a.imshow(gt[0], cmap="RdBu_r", vmin=-v, vmax=v, animated=True))

    def upd(k):
        for im, seq in zip(ims, [gt, pred_t, pred_r]):
            im.set_array(seq[k])
        fig.suptitle(f"NS smoke rollout — step {k+1}/{STEPS}", fontsize=11)
        return ims

    fig.tight_layout()
    a = anim.FuncAnimation(fig, upd, frames=STEPS, interval=250, blit=False)
    out = "results/figs/ns_rollout.gif"; os.makedirs(os.path.dirname(out), exist_ok=True)
    a.save(out, writer=anim.PillowWriter(fps=4)); print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
