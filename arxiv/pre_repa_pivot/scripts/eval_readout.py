"""Readout ablation on a FROZEN fjepa encoder — does mean-pool throw away the
second-order (variance / cross-token) structure where Re/Sc actually live?
No retraining: load encoder, embed frame 0, probe (logRe, Sc) under several
readouts of the 128 latent tokens. Baseline = mean (what we've been using)."""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from torch.utils.data import DataLoader
from src.models.fjepa import FunctionalJEPA
from src.data.well2d import ShearFlowClipDataset, make_coords_2d, fields_to_tokens
from src.metrics import lin_probe

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def tokens_of(model, ds, coords, idx, batch=128):
    model.eval(); T, Y = [], []
    for clip, y in DataLoader(ds, batch_size=batch):
        fa = clip[:, :, 0].to(DEVICE)                    # frame 0
        tok = model.encode(fields_to_tokens(fa, idx), coords[idx])   # B,M,D
        T.append(tok.cpu()); Y.append(y.numpy())
    return torch.cat(T), np.concatenate(Y)


def readouts(T):
    """T: (N, M, D) -> dict of (N, F) feature matrices."""
    mean = T.mean(1)                                     # N,D
    std = T.std(1)                                        # N,D
    tc = T - T.mean(1, keepdim=True)
    D = T.shape[-1]
    cov = torch.einsum("nmd,nme->nde", tc, tc) / T.shape[1]   # N,D,D second moment
    iu = torch.triu_indices(D, D)
    triu = cov[:, iu[0], iu[1]]                           # N, D(D+1)/2
    return {
        "mean (baseline)": mean.numpy(),
        "std": std.numpy(),
        "mean+std": torch.cat([mean, std], 1).numpy(),
        "2nd-moment (triu)": triu.numpy(),
        "mean+std+2nd": torch.cat([mean, std, triu], 1).numpy(),
    }


def probe(Ztr, Ytr, Zva, Yva):
    out = []
    for j, nm in enumerate(["logRe", "Sc"]):
        yt, yv = Ytr[:, j], Yva[:, j]; m, s = yt.mean(), yt.std() + 1e-8
        out.append(lin_probe(Ztr, (yt - m) / s, Zva, (yv - m) / s))
    return out


def run(path):
    ck = torch.load(path, map_location="cpu"); ta = ck["train_args"]
    R = ta["resolution"]; NPIX = R * R
    model = FunctionalJEPA(coord_dim=2, in_chans=4,
                           pred_depth=ta["pred_depth"], pred_type=ta["pred_type"]).to(DEVICE)
    model.encoder.load_state_dict(ck["encoder"])
    clip_len = max(ta["clip_len"], 2)
    tr = ShearFlowClipDataset("train", n_seed=ta["n_seed"], frame_stride=ta["frame_stride"],
                              clip_len=clip_len, side=R, stats=ck["stats"])
    va = ShearFlowClipDataset("valid", n_seed=8, frame_stride=ta["frame_stride"],
                              clip_len=clip_len, side=R, stats=ck["stats"])
    coords = make_coords_2d(n_side=R, device=DEVICE)
    g0 = torch.Generator(device=DEVICE).manual_seed(0)
    idx = torch.randperm(NPIX, generator=g0, device=DEVICE)[:1024]
    Ttr, Ytr = tokens_of(model, tr, coords, idx); Tva, Yva = tokens_of(model, va, coords, idx)
    rtr, rva = readouts(Ttr), readouts(Tva)
    print(f"\n=== {os.path.basename(path)} ===", flush=True)
    for nm in rtr:
        re, sc = probe(rtr[nm], Ytr, rva[nm], Yva)
        print(f"  {nm:20s} dim={rtr[nm].shape[1]:6d}  logRe={re:+.3f}  Sc={sc:+.3f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", nargs="+", required=True)
    for p in ap.parse_args().ckpt:
        run(p)
