"""Resolve cosine-artifact vs genuine time-invariance of the functional encoder.

tpersist=cos(La,Lb)=1.0 could mean (a) the latent set truly does not change over Δt
(encoder is time-blind -> need an architectural fix), or (b) it changes but a large
shared DC offset saturates cosine (-> just a bad prediction target). This decides which.

Decomposition on the POOLED rep R[b,k] (sim b, frame k):
  sample_var   = Var_b( mean_k R )   — across-sim variation (carries the labels Re/Sc)
  temporal_var = mean_b( Var_k R )   — within-sim temporal variation (the dynamics)
If temporal_var/sample_var << 1 AND relative-L2 token change is tiny -> genuinely
time-blind. If relative-L2 is sizable while raw-cos~1 -> DC artifact, tokens DO move.
"""
from __future__ import annotations
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from src.models.fjepa import FunctionalJEPA
from src.data.well2d import ShearFlowClipDataset, make_coords_2d, fields_to_tokens

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--nclip", type=int, default=256)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location=DEVICE)
    ta = ck["train_args"]; R = ta["resolution"]; NPIX = R * R
    print(f"ckpt={os.path.basename(args.ckpt)}  res={R}  clip_len={max(ta['clip_len'], ta['dt_max']+1)}")

    model = FunctionalJEPA(coord_dim=2, in_chans=4, pred_depth=ta["pred_depth"],
                           pred_type=ta["pred_type"]).to(DEVICE).eval()
    model.encoder.load_state_dict(ck["encoder"])
    clip_len = max(ta["clip_len"], ta["dt_max"] + 1)
    va = ShearFlowClipDataset("valid", n_seed=8, frame_stride=ta["frame_stride"],
                              clip_len=clip_len, side=R, stats=ck["stats"])
    coords = make_coords_2d(n_side=R, device=DEVICE)
    g = torch.Generator(device=DEVICE).manual_seed(0)
    idx = torch.randperm(NPIX, generator=g, device=DEVICE)[:1024]

    reps, toks = [], []                       # R[b,k] pooled ; first-clip tokens for L2/cos
    nseen = 0
    for clip, _y in DataLoader(va, batch_size=64):
        clip = clip.to(DEVICE); B, _, K = clip.shape[:3]
        rk = []
        for k in range(K):
            t = model.encoder(fields_to_tokens(clip[:, :, k], idx), coords[idx])  # (B,M,D)
            rk.append(model.represent(t).cpu())
            if k in (0, 1, 2, 4) and nseen == 0:
                toks.append(t.cpu())
        reps.append(torch.stack(rk, 1))       # (B,K,D)
        nseen += B
        if nseen >= args.nclip:
            break
    Rbk = torch.cat(reps, 0).numpy()          # (Nb, K, D)
    Nb, K, D = Rbk.shape

    sample_var = Rbk.mean(1).var(0).sum()                 # Var_b(mean_k)
    temporal_var = Rbk.var(1).mean(0).sum()               # mean_b(Var_k)
    print(f"\nPOOLED rep variance decomposition (D={D}, Nb={Nb}, K={K}):")
    print(f"  sample_var (across-sim, label) = {sample_var:.5f}")
    print(f"  temporal_var (within-sim, dyn) = {temporal_var:.5f}")
    print(f"  temporal/sample ratio          = {temporal_var/ (sample_var+1e-12):.4f}")

    L0, L1, L2c, L4 = (t.to(DEVICE) for t in toks)        # tokens at frames 0,1,2,4
    for nm, Lk in [("Δ1", L1), ("Δ2", L2c), ("Δ4", L4)]:  # token-level change vs frame 0
        rel = (Lk - L0).norm(dim=-1) / (L0.norm(dim=-1) + 1e-8)        # (B,M)
        raw = F.cosine_similarity(L0, Lk, dim=-1)                       # (B,M)
        mu = L0.mean((0, 1), keepdim=True)                             # global DC token
        cen = F.cosine_similarity(L0 - mu, Lk - mu, dim=-1)            # DC-removed cosine
        print(f"  {nm}: relL2={rel.mean():.4f}  raw-cos={raw.mean():.4f}  "
              f"centered-cos={cen.mean():.4f}")


if __name__ == "__main__":
    main()
