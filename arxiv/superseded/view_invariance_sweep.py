"""Invariance-vs-budget: FAE ICC across sensor budgets 64 -> 4096 (the load-bearing version).
Same field, K partial views, measure ICC = between-field var / (between+within-view var) in [0,1].
At a GENEROUS budget every encoder is near-saturated (invariance is easy); the real claim is that
FAE STAYS view-invariant as the budget COLLAPSES. FAE swept across the full range; MAE/JEPA shown only
at budgets where their native masking is still defensible (>=1024 ~ 6%; below that they are OOD).
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
SIDE = 128; NPIX = SIDE * SIDE; IN_CH = 3
N, K = 96, 8; SEEDS = [0, 1, 2]
BUDGETS = [64, 128, 256, 512, 1024, 2048, 4096]      # FAE sensor budgets
VIT_MIN = 1024                                        # MAE/JEPA only at/above this (else OOD)
COL = {"FAE": "#d62728", "MAE": "#1f77b4", "JEPA": "#2ca02c"}

ds = NSDataset("test", side=SIDE, mode="clip", clip_len=2, frame_stride=4, n_traj=16)
X = torch.stack([ds[i][0][:, 0] for i in range(min(N, len(ds)))]).to(DEV)
N = X.shape[0]; coords = make_coords_2d(n_side=SIDE, device=DEV)


def load(method, s):
    fn = {"FAE": ckpt_file("fae","fae_ns128",s), "MAE": ckpt_file("mae","mae_ns128",s), "JEPA": ckpt_file("jepa","jepa_ns128",s)}[method]
    ck = torch.load(fn, map_location=DEV); a = ck["train_args"]
    if method == "FAE":
        m = FAE(emb_dim=a["emb_dim"], num_iter=a.get("num_iter", 4), num_latents=a["num_latents"],
                in_chans=IN_CH, coord_dim=2)
    else:
        m = build_model("mae" if method == "MAE" else "ijepa", resolution=SIDE, in_chans=IN_CH,
                        embed_dim=a["embed_dim"], depth=a["depth"], patch_size=a["patch_size"])
    m.to(DEV); m.load_state_dict(ck["model"]); m.eval(); return m


@torch.no_grad()
def views(m, method, budget):
    Z = []
    for _ in range(K):
        if method == "FAE":
            idx = torch.randperm(NPIX, device=DEV)[:budget]
            Z.append(m.encode_tokens(fields_to_tokens(X, idx), coords[idx]).mean(1))
        elif method == "MAE":
            Z.append(m.forward_encoder(X, 1 - budget / NPIX)[0][:, 1:].mean(1))
        else:
            P = m.num_patches; keep = max(1, round(budget / NPIX * P))
            ki = torch.randperm(P, device=DEV)[:keep].unsqueeze(0).expand(N, -1)
            Z.append(m.encoder(X, keep_idx=ki).mean(1))
    return torch.stack(Z)                              # (K,N,D)


def icc_same(Z):
    between = Z.mean(0).var(0, unbiased=False).sum().item()
    within = Z.var(0, unbiased=False).mean(0).sum().item()
    Zn = torch.nn.functional.normalize(Z, dim=-1)
    iu = torch.triu_indices(K, K, 1)
    same = torch.einsum("knd,jnd->knj", Zn, Zn)[iu[0], :, iu[1]].mean().item()
    return between / (between + within + 1e-12), same


curve = {m: {b: {"icc": [], "same": []} for b in BUDGETS} for m in ["FAE", "MAE", "JEPA"]}
for s in SEEDS:
    for name in ["FAE", "MAE", "JEPA"]:
        m = load(name, s)
        for b in BUDGETS:
            if name != "FAE" and b < VIT_MIN:
                continue
            icc, same = icc_same(views(m, name, b))
            curve[name][b]["icc"].append(icc); curve[name][b]["same"].append(same)
            print(f"seed{s} {name:4s} budget={b:5d}  ICC={icc:.3f}  same={same:.3f}", flush=True)

fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.8))
for name in ["FAE", "MAE", "JEPA"]:
    bs = [b for b in BUDGETS if curve[name][b]["icc"]]
    icm = [np.mean(curve[name][b]["icc"]) for b in bs]; ics = [np.std(curve[name][b]["icc"]) for b in bs]
    sam = [np.mean(curve[name][b]["same"]) for b in bs]
    style = dict(color=COL[name], lw=2, capsize=3)
    a0.errorbar(bs, icm, yerr=ics, fmt="-o" if name == "FAE" else "--s", label=name, **style)
    a1.plot(bs, sam, "-o" if name == "FAE" else "--s", color=COL[name], lw=2, label=name)
for a, ttl, yl in [(a0, "Invariance vs budget — does FAE stay invariant as sensors collapse?", "ICC = between/(between+within)"),
                   (a1, "Same-field cosine vs budget (views diverge at low budget)", "same-field cosine")]:
    a.set_xscale("log", base=2); a.set_xticks(BUDGETS); a.set_xticklabels([str(b) for b in BUDGETS])
    a.set_xlabel("# sensors / equivalent observed pixels"); a.set_ylabel(yl); a.set_title(ttl, fontsize=10.5)
    a.grid(alpha=.3, which="both"); a.legend(fontsize=9)
a0.set_ylim(0, 1.03)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "view_invariance_sweep.png"), dpi=150)
print("wrote view_invariance_sweep.png")
print("\n=== FAE ICC across budgets ===")
for b in BUDGETS:
    print(f"  {b:5d}: ICC {np.mean(curve['FAE'][b]['icc']):.3f}  same-cos {np.mean(curve['FAE'][b]['same']):.3f}")
