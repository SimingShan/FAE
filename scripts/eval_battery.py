"""Battery probe (generality, anti-Goodhart): frozen encoder -> linear-probe a panel of
instantaneous physical quantities computed from the DENSE frame, plus the (logRe, Sc)
params. A general representation should linearly expose many; if only Re/Sc move we'd
have a detector. Channels: [tracer, pressure, vx, vy]."""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from torch.utils.data import DataLoader
from src.models import FAE
from src.data.well2d import ShearFlowClipDataset, make_coords_2d, fields_to_tokens
from src.metrics import lin_probe
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NAMES = ["logRe", "Sc", "kin_energy", "enstrophy", "scalar_var", "scalar_grad", "press_var"]


def quantities(f):                                   # f: (B,4,H,W) dense
    tr, pr, vx, vy = f[:, 0], f[:, 1], f[:, 2], f[:, 3]
    vx_y, vx_x = torch.gradient(vx, dim=(1, 2)); vy_y, vy_x = torch.gradient(vy, dim=(1, 2))
    tr_y, tr_x = torch.gradient(tr, dim=(1, 2))
    vort = vx_y - vy_x
    return torch.stack([
        0.5 * (vx ** 2 + vy ** 2).mean((1, 2)),       # kinetic energy
        (vort ** 2).mean((1, 2)),                      # enstrophy
        tr.var((1, 2)),                                # scalar variance
        torch.sqrt(tr_x ** 2 + tr_y ** 2).mean((1, 2)),# scalar gradient (mixing)
        pr.var((1, 2)),                                # pressure variance
    ], dim=1)


@torch.no_grad()
def embed(m, ds, coords, idx, batch=64):
    m.eval(); Z, Q, Y = [], [], []
    for clip, y in DataLoader(ds, batch_size=batch):
        f = clip[:, :, 0].to(DEVICE)
        Z.append(m.represent(m.encode_tokens(fields_to_tokens(f, idx), coords[idx])).cpu().numpy())
        Q.append(quantities(f).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(Z), np.concatenate(Q), np.concatenate(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/checkpoints/g1/faep_twoview_tvb_s0.pt")
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu"); R = ck["train_args"]["resolution"]; NPIX = R * R
    m = FAE(emb_dim=320, num_iter=4, depth_per_iter=4, num_latents=128, num_cross_heads=4,
            num_self_heads=8, n_freq=32, max_freq=32, coord_dim=2, in_chans=4).to(DEVICE)
    m.load_state_dict(ck["model"]); m.eval()
    tr = ShearFlowClipDataset("train", n_seed=12, frame_stride=12, clip_len=2, side=R)
    va = ShearFlowClipDataset("valid", n_seed=8, frame_stride=12, clip_len=2, side=R, stats=ck["stats"])
    coords = make_coords_2d(R, DEVICE)
    idx = torch.randperm(NPIX, generator=torch.Generator(device=DEVICE).manual_seed(0), device=DEVICE)[:1024]
    Ztr, Qtr, Ytr = embed(m, tr, coords, idx); Zva, Qva, Yva = embed(m, va, coords, idx)
    Ttr = np.concatenate([Ytr, Qtr], 1); Tva = np.concatenate([Yva, Qva], 1)
    print(f"=== battery probe [{os.path.basename(args.ckpt)}] (frozen mean-pool, 1024 sparse pts) ===")
    for j, nm in enumerate(NAMES):
        yt, yv = Ttr[:, j], Tva[:, j]; mn, s = yt.mean(), yt.std() + 1e-8
        r2 = lin_probe(Ztr, (yt - mn) / s, Zva, (yv - mn) / s)
        print(f"  {nm:12s}  R2={r2:+.3f}", flush=True)


if __name__ == "__main__":
    main()
