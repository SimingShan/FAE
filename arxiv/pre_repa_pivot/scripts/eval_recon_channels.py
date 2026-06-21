"""Reconstruction across ALL 4 shear_flow channels [tracer,pressure,vx,vy], MAE vs ours,
for SEVERAL different clips. Rows = {GT, MAE recon (green squares = given patches),
ours recon (green dots = sensors)}; cols = channels. Physical units; normalized per-channel MSE.
Saves results/plots/recon_all_channels_c{i}.png."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.models import FAE
from benchmarks.mae.mae import mae_physics
from src.data.well2d import ShearFlowClipDataset, make_coords_2d, fields_to_tokens
DEVICE = "cpu"; os.makedirs("results/plots", exist_ok=True)
CH = ["tracer", "pressure", "vel_x", "vel_y"]
N_CLIPS, FRAME, N_SENS = 4, 15, 512


def t4(s):
    m, sd = s
    return (torch.tensor(np.asarray(m), dtype=torch.float32).view(1, 4, 1, 1),
            torch.tensor(np.asarray(sd), dtype=torch.float32).view(1, 4, 1, 1))


def mse(a, b): return float(((a - b) ** 2).mean())


def render(fae, mae, fm, fs, mm, ms, coords, R, xt, title, out):
    NPIX = R * R; phys = xt * fs + fm
    gg = torch.Generator().manual_seed(0)
    with torch.no_grad():
        iA = torch.randperm(NPIX, generator=gg)[:N_SENS]
        tok = fae.encode_tokens(fields_to_tokens(xt, iA), coords[iA])
        rec_o = (fae.decoder(tok, coords)[0].permute(1, 0).reshape(4, R, R).unsqueeze(0)) * fs + fm
        xt_mae = (phys - mm) / ms; torch.manual_seed(0)
        _, pred, _ = mae(xt_mae, mask_ratio=0.0)           # FULL field in -> decoder reconstruction
        rec_m = mae.unpatchify(pred) * ms + mm
    rec_o_n = (rec_o - fm) / fs; rec_m_n = (rec_m - mm) / ms; xt_mae = (phys - mm) / ms
    srow = (iA // R).numpy(); scol = (iA % R).numpy()
    rows = [("ground truth", phys, None),
            ("MAE recon (full field in)", rec_m, mse(rec_m_n, xt_mae)),
            ("ours recon (%d sensors)" % N_SENS, rec_o, mse(rec_o_n, xt))]
    fig, ax = plt.subplots(3, 4, figsize=(13, 9.2))
    for r, (name, field, e) in enumerate(rows):
        for c in range(4):
            gt = phys[0, c]
            v = max(abs(float(gt.min())), abs(float(gt.max())), 1.5 * float(fs[0, c, 0, 0]))  # clamp >=1.5x global std -> flat fields render flat
            ax[r, c].imshow(field[0, c], cmap="RdBu_r", vmin=-v, vmax=v); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0: ax[r, c].set_title(CH[c], fontsize=11)
            if r == 2:                                      # ours: green dots = the 512 sensors it sees
                ax[r, c].scatter(scol, srow, s=1.5, c="lime", marker=".", linewidths=0)
        lbl = name + (f"\n(MSE={e:.3f})" if e is not None else "")
        ax[r, 0].set_ylabel(lbl, fontsize=10)
    pc_m = [mse(rec_m_n[0, c], xt_mae[0, c]) for c in range(4)]
    pc_o = [mse(rec_o_n[0, c], xt[0, c]) for c in range(4)]
    for c in range(4):
        ax[1, c].set_xlabel(f"nMSE={pc_m[c]:.3f}", fontsize=8)
        ax[2, c].set_xlabel(f"nMSE={pc_o[c]:.3f}", fontsize=8)
    fig.suptitle(title, fontsize=12); fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return pc_m[0], pc_o[0]


def main():
    cko = torch.load("results/checkpoints/g1/faep_twoview_tvb_s0.pt", map_location="cpu")
    R = cko["train_args"]["resolution"]
    fae = FAE(emb_dim=320, num_iter=4, depth_per_iter=4, num_latents=128, num_cross_heads=4,
              num_self_heads=8, n_freq=32, max_freq=32, coord_dim=2, in_chans=4)
    fae.load_state_dict(cko["model"]); fae.eval(); fm, fs = t4(cko["stats"])
    ckm = torch.load("results/checkpoints/g1/mae_shear_mae_s1.pt", map_location="cpu")
    mae = mae_physics(img_size=R); mae.load_state_dict(ckm["model"]); mae.eval(); mm, ms = t4(ckm["stats"])
    va = ShearFlowClipDataset("valid", n_seed=4, frame_stride=4, clip_len=16, side=R, stats=cko["stats"])
    coords = make_coords_2d(R, DEVICE)
    logRe = getattr(va, "logRe", None); Sc = getattr(va, "Sc", None)
    idxs = [int(len(va) * k / N_CLIPS) for k in range(N_CLIPS)]   # spread across params/ICs
    print(f"valid {len(va)} clips; rendering {idxs}")
    for n, i in enumerate(idxs):
        xt = va[i][0][:, FRAME].unsqueeze(0)
        tag = ""
        if logRe is not None: tag = f"  Re=e^{logRe[i]:.1f}={np.exp(logRe[i]):.0f}  Sc={Sc[i]:.2f}"
        out = f"results/plots/recon_all_channels_c{n}.png"
        tm, to = render(fae, mae, fm, fs, mm, ms, coords, R, xt, f"shear_flow recon — clip {i}{tag}", out)
        print(f"  c{n} (clip {i}): tracer nMSE  MAE={tm:.3f}  ours={to:.3f}  -> {out}")


if __name__ == "__main__":
    main()
