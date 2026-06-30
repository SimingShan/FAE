"""Collapse / over-invariance check (answers the 'diff-field cosine ~0.95 = collapse?' review).
Same field x K views @ 25% budget. Report RAW vs MEAN-CENTERED same/diff cosine + ICC, per method.
RAW cosine is inflated by the shared latent offset; CENTERED cosine reveals true field separation.
A collapsed encoder stays high even centered AND would fail the probe (R^2~floor).
"""
import os, sys
import numpy as np, torch
ROOT = "/gpfs/radev/scratch/lu_lu/ss5235/WFAE"; sys.path.insert(0, ROOT)
from src.data.ns import NSDataset
from src.data.well2d import make_coords_2d, fields_to_tokens
from src.models.fae import FAE
from scripts.train_baseline import build_model
from src.config import ckpt_file

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SIDE = 128; NPIX = SIDE * SIDE; IN_CH = 3; N, K = 96, 8; FRAC = 0.25; NS = int(FRAC * NPIX)
SEEDS = [0, 1, 2]
ds = NSDataset("test", side=SIDE, mode="clip", clip_len=2, frame_stride=4, n_traj=16)
X = torch.stack([ds[i][0][:, 0] for i in range(min(N, len(ds)))]).to(DEV); N = X.shape[0]
coords = make_coords_2d(n_side=SIDE, device=DEV)


@torch.no_grad()
def latents(method, ck):
    a = ck["train_args"]
    if method == "fae":
        m = FAE(emb_dim=a["emb_dim"], num_iter=a.get("num_iter", 4), num_latents=a["num_latents"], in_chans=IN_CH, coord_dim=2).to(DEV)
        m.load_state_dict(ck["model"]); m.eval()
        return torch.stack([m.encode_tokens(fields_to_tokens(X, i), coords[i]).mean(1)
                            for i in (torch.randperm(NPIX, device=DEV)[:NS] for _ in range(K))])
    m = build_model("mae" if method == "mae" else "ijepa", resolution=SIDE, in_chans=IN_CH,
                    embed_dim=a["embed_dim"], depth=a["depth"], patch_size=a["patch_size"]).to(DEV)
    m.load_state_dict(ck["model"]); m.eval()
    if method == "mae":
        return torch.stack([m.forward_encoder(X, 1 - FRAC)[0][:, 1:].mean(1) for _ in range(K)])
    P = m.num_patches; keep = max(1, int(FRAC * P))
    return torch.stack([m.encoder(X, keep_idx=torch.randperm(P, device=DEV)[:keep][None].expand(N, -1)).mean(1) for _ in range(K)])


def cos_stats(Z):                                       # Z (K,N,D)
    Zn = torch.nn.functional.normalize(Z, dim=-1)
    iu = torch.triu_indices(K, K, 1)
    same = torch.einsum("knd,jnd->knj", Zn, Zn)[iu[0], :, iu[1]].mean().item()
    fm = torch.nn.functional.normalize(Z.mean(0), dim=-1)
    C = fm @ fm.T; off = ~torch.eye(N, dtype=torch.bool, device=Z.device)
    diff = C[off].mean().item()
    return same, diff


for name, meth in [("FAE", "fae"), ("MAE", "mae"), ("JEPA", "jepa")]:
    tag = {"fae": "fae_ns128", "mae": "mae_ns128", "jepa": "jepa_ns128"}[meth]
    R = {"raw_same": [], "raw_diff": [], "cen_same": [], "cen_diff": [], "icc": []}
    for s in SEEDS:
        Z = latents(meth, torch.load(ckpt_file(meth, tag, s), map_location=DEV))
        rs, rd = cos_stats(Z)
        Zc = Z - Z.mean((0, 1), keepdim=True)            # strip the shared offset
        cs, cd = cos_stats(Zc)
        between = Z.mean(0).var(0, unbiased=False).sum().item(); within = Z.var(0, unbiased=False).mean(0).sum().item()
        R["raw_same"].append(rs); R["raw_diff"].append(rd); R["cen_same"].append(cs); R["cen_diff"].append(cd)
        R["icc"].append(between / (between + within + 1e-12))
    mn = {k: np.mean(v) for k, v in R.items()}
    print(f"{name:5s}  RAW  same={mn['raw_same']:.3f} diff={mn['raw_diff']:.3f} (gap {mn['raw_same']-mn['raw_diff']:+.3f}) | "
          f"CENTERED same={mn['cen_same']:.3f} diff={mn['cen_diff']:.3f} (gap {mn['cen_same']-mn['cen_diff']:+.3f}) | ICC={mn['icc']:.3f}", flush=True)
