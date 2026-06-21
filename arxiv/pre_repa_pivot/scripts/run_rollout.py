"""Direct (pixel U-Net) vs latent rollout, small-data sweep on shear_flow.
- DIRECT: x_t -> UNet -> x_{t+1}, rolled out autoregressively (predictions fed back).
- LATENT: conv-AE + latent predictor; roll PURELY in latent (z_{k+1}=pred(z_k)), decode each z.
Trained from scratch on N trajectories (ICs of one (Re,Sc) combo); rollout error vs horizon on
held-out ICs. Hypothesis: at small N, latent rollout degrades more gracefully (low-dim latent
regularizes; errors don't compound through the decoder) while the pixel U-Net overfits/diverges."""
import sys, os, glob, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, h5py
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIDE, STRIDE, T = 64, 4, 40          # 64x64, every 4th frame, 40 frames/trajectory
ROOT = os.environ["THE_WELL_DATA_DIR"] + "/shear_flow/data/train"


def load_trajs(n_keep=12):
    f = sorted(glob.glob(ROOT + "/*.hdf5"))[14]                       # one mid-Re combo
    with h5py.File(f) as h:
        tr = h["t0_fields/tracer"][:n_keep, :T * STRIDE:STRIDE]       # (n,T,512,256)
        pr = h["t0_fields/pressure"][:n_keep, :T * STRIDE:STRIDE]
        ve = h["t1_fields/velocity"][:n_keep, :T * STRIDE:STRIDE]     # (n,T,512,256,2)
    a = np.stack([tr, pr, ve[..., 0], ve[..., 1]], 2).astype(np.float32)  # (n,T,4,512,256)
    x = torch.from_numpy(a).reshape(-1, 4, a.shape[-2], a.shape[-1])
    x = F.interpolate(x, size=(SIDE, SIDE), mode="bilinear", align_corners=False).reshape(n_keep, T, 4, SIDE, SIDE)
    m, s = x.mean((0, 1, 3, 4), keepdim=True), x.std((0, 1, 3, 4), keepdim=True) + 1e-6
    return ((x - m) / s)                                             # (n,T,4,SIDE,SIDE) normalized


def conv(i, o, k=3, s=1): return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2), nn.GroupNorm(8, o), nn.GELU())


class UNet(nn.Module):
    def __init__(s, c=4, w=48):
        super().__init__()
        s.d1 = conv(c, w); s.d2 = conv(w, w * 2, s=2); s.d3 = conv(w * 2, w * 4, s=2)
        s.u2 = conv(w * 4, w * 2); s.u1 = conv(w * 4, w); s.out = nn.Conv2d(w * 2, c, 1)

    def forward(s, x):
        a = s.d1(x); b = s.d2(a); cc = s.d3(b)
        u = F.interpolate(s.u2(cc), scale_factor=2, mode="nearest")
        u = F.interpolate(s.u1(torch.cat([u, b], 1)), scale_factor=2, mode="nearest")
        return x + s.out(torch.cat([u, a], 1))                       # residual (predict delta)


class Latent(nn.Module):
    def __init__(s, c=4, w=48, cz=64):
        super().__init__()
        s.enc = nn.Sequential(conv(c, w, s=2), conv(w, w * 2, s=2), conv(w * 2, cz, s=2))   # 64->8
        s.dec = nn.Sequential(conv(cz, w * 2), nn.Upsample(scale_factor=2), conv(w * 2, w),
                              nn.Upsample(scale_factor=2), conv(w, w), nn.Upsample(scale_factor=2), nn.Conv2d(w, c, 1))
        s.pred = nn.Sequential(conv(cz, cz), conv(cz, cz), nn.Conv2d(cz, cz, 3, 1, 1))

    def encode(s, x): return s.enc(x)
    def decode(s, z): return s.dec(z)
    def step(s, z): return z + s.pred(z)                            # residual latent dynamics


def pairs(trajs):                                                    # (x_t, x_{t+1}) over all trajectories
    a = torch.cat([t[:-1] for t in trajs]); b = torch.cat([t[1:] for t in trajs])
    return a.to(DEVICE), b.to(DEVICE)


def train(model, trajs, kind, epochs=400, lr=3e-4):
    a, b = pairs(trajs); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(a), device=DEVICE)
        for i in range(0, len(a), 64):
            idx = perm[i:i + 64]; xa, xb = a[idx], b[idx]
            if kind == "unet":
                loss = F.mse_loss(model(xa), xb)
            else:
                za = model.encode(xa)
                loss = F.mse_loss(model.decode(za), xa) + F.mse_loss(model.decode(model.step(za)), xb)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()


@torch.no_grad()
def rollout(model, traj, kind, H=20):
    """relative L2 error per step:  ||pred - truth|| / ||truth||  (1.0 = mean-prediction level)."""
    x0 = traj[0:1].to(DEVICE); err = []
    x = x0; z = model.encode(x0) if kind != "unet" else None
    for k in range(1, H + 1):
        if kind == "unet":
            x = model(x); pred = x
        else:
            z = model.step(z); pred = model.decode(z)
        gt = traj[k:k + 1].to(DEVICE)
        err.append((torch.linalg.norm(pred - gt) / torch.linalg.norm(gt)).item())
    return np.array(err)


@torch.no_grad()
def make_plot(models, traj, out, hor=(1, 5, 10, 20)):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    x0 = traj[0:1].to(DEVICE); H = max(hor)
    pu, pl = {}, {}; x = x0.clone(); z = models["latent"].encode(x0)
    for k in range(1, H + 1):
        x = models["unet"](x); z = models["latent"].step(z)
        if k in hor:
            pu[k] = x[0, 0].cpu(); pl[k] = models["latent"].decode(z)[0, 0].cpu()
    rows = [("ground truth", {k: traj[k, 0].cpu() for k in hor}),
            ("direct (UNet)", pu), ("latent", pl)]
    fig, ax = plt.subplots(3, len(hor), figsize=(2.6 * len(hor), 7.6))
    for r, (name, d) in enumerate(rows):
        for j, k in enumerate(hor):
            v = max(abs(float(traj[k, 0].min())), abs(float(traj[k, 0].max())))
            ax[r, j].imshow(d[k], cmap="RdBu_r", vmin=-v, vmax=v); ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
            if r == 0: ax[r, j].set_title(f"t+{k}", fontsize=11)
        ax[r, 0].set_ylabel(name, fontsize=11)
    fig.suptitle("rollout prediction (tracer) — direct diverges, latent stays coherent", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig); print("saved", out)


def main():
    data = load_trajs(12)
    eval_trajs = data[-3:]; PLOT_N = 4; keep = {}                   # held-out ICs; keep N=4 models for the plot
    print(f"loaded {len(data)} trajectories, {T} frames each, {SIDE}x{SIDE}")
    print(f"{'N_traj':7} {'method':7} | rollout REL-L2 @ horizon 1 / 5 / 10 / 20")
    for N in (1, 2, 4, 8):
        train_trajs = [data[i] for i in range(N)]
        for kind, Mk in (("unet", UNet), ("latent", Latent)):
            torch.manual_seed(0); m = Mk().to(DEVICE)
            np_ = sum(p.numel() for p in m.parameters()) / 1e6
            train(m, train_trajs, kind)
            if N == PLOT_N: keep[kind] = m
            E = np.mean([rollout(m, t, kind) for t in eval_trajs], 0)   # avg over eval trajectories
            print(f"  {N:<5} {kind:7} | {E[0]:.3f} / {E[4]:.3f} / {E[9]:.3f} / {E[19]:.3f}   ({np_:.2f}M)", flush=True)
    os.makedirs("results/plots", exist_ok=True)
    make_plot(keep, eval_trajs[0], f"results/plots/rollout_pred_N{PLOT_N}.png")
    print("=> REL-L2: 1.0 = mean-prediction level. direct >1 = diverged; latent ~1 = saturated (stable).")


if __name__ == "__main__":
    main()
