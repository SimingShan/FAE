"""Residual diagnostic: is the latent FROZEN in time, or does it move along its own
direction (which cosine=1 would hide)? Answers the 'why not a residual predictor?' Q.

Per held-out (t, t+Δ) pair, on the frozen encoder:
  TOKEN/POOL cos       : direction agreement (≈1 was our 'frozen' read — direction only)
  resnorm  = ‖Lb-La‖/‖La‖        : RELATIVE size of the temporal step (magnitude-aware)
  magratio = ‖Lb‖/‖La‖           : pure-scaling check (cos=1 + magratio≠1 => scaling dynamics)
DECISION-RELEVANT (pooled):
  d_time   = mean ‖r(t+Δ)-r(t)‖   : how far the rep moves in TIME (same sim)
  d_sim    = mean ‖r_i - r_j‖     : spread ACROSS sims (what the probe rides on)
  ratio    = d_time / d_sim       : temporal motion as a fraction of the useful signal.
                                    <<1 => effectively static for the task (residual won't help).
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from src.models.fjepa import FunctionalJEPA
from src.data.well2d import ShearFlowClipDataset, make_coords_2d, fields_to_tokens

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def run(path, dt, nb=6):
    ck = torch.load(path, map_location="cpu"); ta = ck["train_args"]
    R = ta["resolution"]; NPIX = R * R
    model = FunctionalJEPA(coord_dim=2, in_chans=4,
                           pred_depth=ta["pred_depth"], pred_type=ta["pred_type"]).to(DEVICE)
    model.encoder.load_state_dict(ck["encoder"]); model.eval()
    clip_len = max(ta["clip_len"], dt + 1)
    va = ShearFlowClipDataset("valid", n_seed=8, frame_stride=ta["frame_stride"],
                              clip_len=clip_len, side=R, stats=ck["stats"])
    coords = make_coords_2d(n_side=R, device=DEVICE)
    g0 = torch.Generator(device=DEVICE).manual_seed(0)
    idx = torch.randperm(NPIX, generator=g0, device=DEVICE)[:1024]
    cT = rT = mT = cP = rP = 0.0; n = 0; dtime = 0.0; Rt0 = []
    for bi, (clip, _y) in enumerate(DataLoader(va, batch_size=128)):
        if bi >= nb: break
        clip = clip.to(DEVICE); B = clip.size(0); K = clip.size(2)
        bidx = torch.arange(B, device=DEVICE)
        delta = torch.full((B,), dt, device=DEVICE)
        ts = (torch.rand(B, device=DEVICE) * (K - delta).float()).long()
        fa = clip[bidx, :, ts]; fb = clip[bidx, :, ts + delta]
        La = model.encode(fields_to_tokens(fa, idx), coords[idx])
        Lb = model.encode(fields_to_tokens(fb, idx), coords[idx])
        cT += F.cosine_similarity(La, Lb, dim=-1).mean(1).sum().item()
        rT += ((Lb - La).norm(dim=-1) / (La.norm(dim=-1) + 1e-8)).mean(1).sum().item()
        mT += (Lb.norm(dim=-1) / (La.norm(dim=-1) + 1e-8)).mean(1).sum().item()
        ra, rb = model.represent(La), model.represent(Lb)
        cP += F.cosine_similarity(ra, rb, dim=-1).sum().item()
        rP += ((rb - ra).norm(dim=-1) / (ra.norm(dim=-1) + 1e-8)).sum().item()
        dtime += (rb - ra).norm(dim=-1).sum().item()
        Rt0.append(ra.cpu()); n += B
    Rt0 = torch.cat(Rt0)                                  # pooled rep at t, across sims
    pd = torch.pdist(Rt0); d_sim = pd.mean().item()       # between-sim spread
    d_time = dtime / n
    print(f"{os.path.basename(path):42s} dt={dt}  "
          f"TOKEN[cos={cT/n:.4f} resnorm={rT/n:.4f} mag={mT/n:.4f}]  "
          f"POOL[cos={cP/n:.4f} resnorm={rP/n:.4f}]  "
          f"d_time={d_time:.3f} d_sim={d_sim:.3f} ratio={d_time/max(d_sim,1e-8):.3f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--dt", type=int, default=4)
    args = ap.parse_args()
    for p in args.ckpt:
        run(p, args.dt)
