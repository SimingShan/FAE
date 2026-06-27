"""Observation-invariance headline: SAME field, K different partial VIEWS at a MATCHED 25% budget.
Does the latent encode the FIELD or the arbitrary sampling? FAE is trained for view-invariance
(twoview); MAE/JEPA are not. Each method uses its NATIVE partial-view encoder:
  FAE : encode a random 25%-of-pixels sensor subset      (4096 sensors)
  MAE : forward_encoder(x, 0.75) -> 25% visible patches  (native masking, per-sample random)
  JEPA: context encoder on a random 25%-of-patches subset (its context branch, in-distribution)
Metrics (per method, 3 seeds): cosine same-field vs diff-field, and the variance "invariance ratio"
  ICC = between-field var / (between + within-field-across-views var)  in [0,1], 1 = fully invariant.
"""
import os, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = "/gpfs/radev/scratch/lu_lu/ss5235/WFAE"; sys.path.insert(0, ROOT)
from src.data.ns import NSDataset
from src.data.well2d import make_coords_2d, fields_to_tokens
from src.models.fae import FAE
from scripts.train_baseline import build_model

DEV = "cuda" if torch.cuda.is_available() else "cpu"
from src.config import ckpt_file
OUT = os.path.join(ROOT, "results/figures")
os.makedirs(OUT, exist_ok=True)
SIDE = 128; NPIX = SIDE * SIDE; IN_CH = 3
N, K = 96, 8                      # fields, views per field
FRAC = 0.25                       # matched observed fraction
N_SENS = int(FRAC * NPIX)         # 4096 FAE sensors
SEEDS = [0, 1, 2]
COL = {"FAE": "#d62728", "MAE": "#1f77b4", "JEPA": "#2ca02c"}

ds = NSDataset("test", side=SIDE, mode="clip", clip_len=2, frame_stride=4, n_traj=16)
X = torch.stack([ds[i][0][:, 0] for i in range(min(N, len(ds)))]).to(DEV)   # (N,C,H,W) frame-0
N = X.shape[0]
coords = make_coords_2d(n_side=SIDE, device=DEV)


@torch.no_grad()
def latents_over_views(method, ck):
    """Return Z (K, N, D): latent of each field under K random partial views."""
    a = ck["train_args"]
    Z = []
    if method == "fae":
        m = FAE(emb_dim=a["emb_dim"], num_iter=a.get("num_iter", 4), num_latents=a["num_latents"],
                in_chans=IN_CH, coord_dim=2).to(DEV); m.load_state_dict(ck["model"]); m.eval()
        for _ in range(K):
            idx = torch.randperm(NPIX, device=DEV)[:N_SENS]
            z = m.encode_tokens(fields_to_tokens(X, idx), coords[idx]).mean(1)
            Z.append(z)
    elif method == "mae":
        m = build_model("mae", resolution=SIDE, in_chans=IN_CH, embed_dim=a["embed_dim"],
                        depth=a["depth"], patch_size=a["patch_size"]).to(DEV); m.load_state_dict(ck["model"]); m.eval()
        for _ in range(K):
            z = m.forward_encoder(X, 1 - FRAC)[0][:, 1:].mean(1)   # per-sample random mask -> 25% visible
            Z.append(z)
    else:  # jepa context encoder on a random patch subset
        m = build_model("ijepa", resolution=SIDE, in_chans=IN_CH, embed_dim=a["embed_dim"],
                        depth=a["depth"], patch_size=a["patch_size"]).to(DEV); m.load_state_dict(ck["model"]); m.eval()
        P = m.num_patches; keep = max(1, int(FRAC * P))
        for _ in range(K):
            ki = torch.randperm(P, device=DEV)[:keep].unsqueeze(0).expand(N, -1)
            z = m.encoder(X, keep_idx=ki).mean(1)
            Z.append(z)
    return torch.stack(Z)            # (K, N, D)


def stats(Z):
    """cosine same-field / diff-field, and ICC variance ratio."""
    Zn = torch.nn.functional.normalize(Z, dim=-1)                 # (K,N,D)
    # same-field: mean cosine over view-pairs within each field
    S = torch.einsum("knd,jnd->knj", Zn, Zn)                      # (K,N,K) cos(view k, view j) per field
    iu = torch.triu_indices(K, K, 1)
    same = S[iu[0], :, iu[1]].mean().item()
    # diff-field: cosine between field means (different fields)
    fm = torch.nn.functional.normalize(Z.mean(0), dim=-1)         # (N,D)
    C = fm @ fm.T; off = ~torch.eye(N, dtype=torch.bool, device=Z.device)
    diff = C[off].mean().item()
    # ICC: between-field var vs within-field(view) var
    between = Z.mean(0).var(0, unbiased=False).sum().item()
    within = Z.var(0, unbiased=False).mean(0).sum().item()
    icc = between / (between + within + 1e-12)
    return same, diff, icc


res = {m: {"same": [], "diff": [], "icc": []} for m in ["FAE", "MAE", "JEPA"]}
for s in SEEDS:
    cks = {"FAE": ckpt_file("fae","fae_ns128",s), "MAE": ckpt_file("mae","mae_ns128",s), "JEPA": ckpt_file("jepa","jepa_ns128",s)}
    meth = {"FAE": "fae", "MAE": "mae", "JEPA": "jepa"}
    for name in ["FAE", "MAE", "JEPA"]:
        ck = torch.load(cks[name], map_location=DEV)
        same, diff, icc = stats(latents_over_views(meth[name], ck))
        res[name]["same"].append(same); res[name]["diff"].append(diff); res[name]["icc"].append(icc)
        print(f"seed{s} {name:4s}  same={same:.3f}  diff={diff:.3f}  gap={same-diff:+.3f}  ICC={icc:.3f}")

names = ["FAE", "MAE", "JEPA"]
fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.5, 4.6))
x = np.arange(len(names))
sm = [np.mean(res[n]["same"]) for n in names]; ss = [np.std(res[n]["same"]) for n in names]
dm = [np.mean(res[n]["diff"]) for n in names]; dd = [np.std(res[n]["diff"]) for n in names]
a0.bar(x - 0.2, sm, 0.4, yerr=ss, capsize=3, color=[COL[n] for n in names], label="same field (diff views)")
a0.bar(x + 0.2, dm, 0.4, yerr=dd, capsize=3, color=[COL[n] for n in names], alpha=0.45, label="different fields")
for i, n in enumerate(names):
    a0.text(i, max(sm[i], dm[i]) + 0.03, f"gap\n{sm[i]-dm[i]:+.2f}", ha="center", fontsize=9, fontweight="bold")
a0.set_xticks(x); a0.set_xticklabels(names); a0.set_ylabel("cosine similarity")
a0.set_title(f"View-invariance: same field across {K} views @ {int(FRAC*100)}% budget"); a0.legend(fontsize=8.5); a0.grid(alpha=.3, axis="y")
im = [np.mean(res[n]["icc"]) for n in names]; istd = [np.std(res[n]["icc"]) for n in names]
a1.bar(x, im, 0.55, yerr=istd, capsize=4, color=[COL[n] for n in names])
for i in range(len(names)):
    a1.text(i, im[i] + istd[i] + 0.01, f"{im[i]:.3f}", ha="center", fontsize=10, fontweight="bold")
a1.set_xticks(x); a1.set_xticklabels(names); a1.set_ylim(0, 1.02)
a1.set_ylabel("invariance ratio  ICC = between / (between+within)")
a1.set_title("Fraction of latent variance that is FIELD (not view)"); a1.grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "view_invariance.png"), dpi=150)
print("wrote view_invariance.png")
for n in names:
    print(f"{n:4s}  same {np.mean(res[n]['same']):.3f}  diff {np.mean(res[n]['diff']):.3f}  "
          f"gap {np.mean(res[n]['same'])-np.mean(res[n]['diff']):+.3f}  ICC {np.mean(res[n]['icc']):.3f}±{np.std(res[n]['icc']):.3f}")
