"""Probe-density sweep: frozen encoder, vary the number of sensor points fed at
EVAL time (64..4096), linear-probe (logRe, Sc). Tests inference-time observation-
invariance — does probe quality hold as sampling density changes?"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from torch.utils.data import DataLoader
from src.models import FAE
from src.data.well2d import ShearFlowClipDataset, make_coords_2d, fields_to_tokens
from src.metrics import lin_probe
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def embed(m, ds, coords, idx, batch=64):
    m.eval(); Zm, Zms, Y = [], [], []
    for clip, y in DataLoader(ds, batch_size=batch):
        fa = clip[:, :, 0].to(DEVICE)
        t = m.encode_tokens(fields_to_tokens(fa, idx), coords[idx])
        Zm.append(t.mean(1).cpu().numpy())
        Zms.append(torch.cat([t.mean(1), t.std(1)], -1).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(Zm), np.concatenate(Zms), np.concatenate(Y)


def probe(Ztr, Ytr, Zva, Yva):
    return [lin_probe(Ztr, (Ytr[:, j] - Ytr[:, j].mean()) / (Ytr[:, j].std() + 1e-8),
                      Zva, (Yva[:, j] - Ytr[:, j].mean()) / (Ytr[:, j].std() + 1e-8)) for j in range(2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/checkpoints/g1/faep_twoview_tvb_s0.pt")
    ap.add_argument("--counts", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048])
    ap.add_argument("--n_seed", type=int, default=8, help="probe train trajectories (small for RAM)")
    ap.add_argument("--stride", type=int, default=16, help="larger stride => fewer clips => less RAM")
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu"); R = ck["train_args"]["resolution"]; NPIX = R * R
    m = FAE(emb_dim=320, num_iter=4, depth_per_iter=4, num_latents=128, num_cross_heads=4,
            num_self_heads=8, n_freq=32, max_freq=32, coord_dim=2, in_chans=4).to(DEVICE)
    m.load_state_dict(ck["model"]); m.eval()
    tr = ShearFlowClipDataset("train", n_seed=args.n_seed, frame_stride=args.stride, clip_len=2, side=R)
    va = ShearFlowClipDataset("valid", n_seed=8, frame_stride=args.stride, clip_len=2, side=R, stats=ck["stats"])
    coords = make_coords_2d(R, DEVICE)
    print(f"=== probe-density sweep [{os.path.basename(args.ckpt)}] ===")
    print(f"{'#sensors':>9} | {'mean logRe':>10} {'Sc':>7} | {'mean+std logRe':>14} {'Sc':>7}")
    for n in args.counts:
        idx = torch.randperm(NPIX, generator=torch.Generator(device=DEVICE).manual_seed(0), device=DEVICE)[:n]
        Ztr_m, Ztr_ms, Ytr = embed(m, tr, coords, idx); Zva_m, Zva_ms, Yva = embed(m, va, coords, idx)
        rm = probe(Ztr_m, Ytr, Zva_m, Yva); rms = probe(Ztr_ms, Ytr, Zva_ms, Yva)
        print(f"{n:>9} | {rm[0]:>10.3f} {rm[1]:>7.3f} | {rms[0]:>14.3f} {rms[1]:>7.3f}", flush=True)


if __name__ == "__main__":
    main()
