"""t-SNE of frozen embeddings, MAE vs ours, colored by the physical coefficients (logRe, Sc).
A good representation should organize smoothly by parameter. 2x2: rows={ours, MAE}, cols={Re, Sc}."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from src.models import FAE
from benchmarks.mae.mae import mae_physics
from src.data.well2d import ShearFlowSnapshotDataset, make_coords_2d, fields_to_tokens
DEVICE = "cpu"; N_MAX = 800


def t4(s):
    m, sd = s
    return (torch.tensor(np.asarray(m), dtype=torch.float32).view(1, 4, 1, 1),
            torch.tensor(np.asarray(sd), dtype=torch.float32).view(1, 4, 1, 1))


def main():
    cko = torch.load("results/checkpoints/g1/faep_twoview_tvb_s0.pt", map_location="cpu", weights_only=False)
    R = cko["train_args"]["resolution"]
    fae = FAE(emb_dim=320, num_iter=4, depth_per_iter=4, num_latents=128, num_cross_heads=4,
              num_self_heads=8, n_freq=32, max_freq=32, coord_dim=2, in_chans=4)
    fae.load_state_dict(cko["model"]); fae.eval(); fm, fs = t4(cko["stats"])
    ckm = torch.load("results/checkpoints/g1/mae_shear_mae_s1.pt", map_location="cpu", weights_only=False)
    mae = mae_physics(img_size=R); mae.load_state_dict(ckm["model"]); mae.eval(); mm, ms = t4(ckm["stats"])

    snap_stats = (np.asarray(cko["stats"][0]).reshape(1, 4, 1, 1), np.asarray(cko["stats"][1]).reshape(1, 4, 1, 1))
    va = ShearFlowSnapshotDataset("valid", n_seed=4, frame_stride=16, side=R, stats=snap_stats)
    idx = np.random.default_rng(0).permutation(len(va))[:N_MAX]
    coords = make_coords_2d(R, DEVICE); NPIX = R * R
    g = torch.Generator().manual_seed(0); iA = torch.randperm(NPIX, generator=g)[:512]
    Zf, Zm, Y = [], [], []
    loader = DataLoader(torch.utils.data.Subset(va, idx.tolist()), batch_size=64)
    with torch.no_grad():
        for x, y in loader:
            tok = fae.encode_tokens(fields_to_tokens(x, iA), coords[iA])
            Zf.append(torch.cat([tok.mean(1), tok.std(1)], -1).numpy())
            xm = ((x * fs + fm) - mm) / ms                     # renormalize to MAE stats
            Zm.append(mae.encode(xm).numpy()); Y.append(y.numpy())
    Zf = np.concatenate(Zf); Zm = np.concatenate(Zm); Y = np.concatenate(Y)
    print(f"embedded {len(Y)} frames; FAE dim {Zf.shape[1]}, MAE dim {Zm.shape[1]}")
    # --- group-average: one point per (Re,Sc) combination (removes within-combo spread) ---
    keys = [tuple(np.round(y, 4)) for y in Y]
    uniq = sorted(set(keys))
    sel = lambda Z, u: Z[[i for i, k in enumerate(keys) if k == u]].mean(0)
    gf = np.stack([sel(Zf, u) for u in uniq]); gm = np.stack([sel(Zm, u) for u in uniq])
    gY = np.array([list(u) for u in uniq])
    print(f"{len(uniq)} (Re,Sc) groups; Re vals {sorted(set(gY[:,0].round(2)))}, Sc vals {sorted(set(gY[:,1].round(2)))}")
    perp = max(3, min(12, (len(uniq) - 1) // 3))
    tf = TSNE(n_components=2, perplexity=perp, init="pca", random_state=0).fit_transform(gf)
    tm = TSNE(n_components=2, perplexity=perp, init="pca", random_state=0).fit_transform(gm)

    names = ["logRe", "Sc"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 10))
    for r, (emb, mlbl) in enumerate([(tf, "ours (twoview)"), (tm, "MAE")]):
        for c in range(2):
            sca = ax[r, c].scatter(emb[:, 0], emb[:, 1], c=gY[:, c], cmap="viridis", s=130, edgecolors="k", linewidths=0.5)
            ax[r, c].set_title(f"{mlbl} — colored by {names[c]}", fontsize=11)
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            fig.colorbar(sca, ax=ax[r, c], fraction=0.046)
    fig.suptitle(f"t-SNE of per-(Re,Sc) mean embeddings ({len(uniq)} combos), shear_flow", fontsize=12)
    fig.tight_layout(); fig.savefig("results/plots/tsne_coeff_grouped.png", dpi=110)
    print("saved results/plots/tsne_coeff_grouped.png")


if __name__ == "__main__":
    main()
