"""Intrinsic-rank (POD/PCA) of shear_flow frames — the reconstruction TARGET difficulty.
If a frame needs >>128 modes for 90-99% energy, a 128-latent bottleneck is undersized;
if <<128, the bottleneck is NOT the recon limit. Compares laminar (early) vs developed (late)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from src.data.well2d import ShearFlowClipDataset
CL, STR, T_LAM, T_DEV = 16, 4, 2, 15


def rank_of(X):                                   # X: (N, D) -> #modes for energy thresholds
    X = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False) ** 2
    c = np.cumsum(s) / s.sum()
    return {p: int(np.searchsorted(c, p) + 1) for p in (0.5, 0.9, 0.95, 0.99)}, len(s)


def main():
    va = ShearFlowClipDataset("valid", n_seed=8, frame_stride=STR, clip_len=CL, side=224)
    lam, dev = [], []
    for i in range(len(va)):
        c = va[i][0]
        lam.append(c[:, T_LAM].reshape(-1).numpy()); dev.append(c[:, T_DEV].reshape(-1).numpy())
    lam, dev = np.stack(lam), np.stack(dev)
    print(f"frames: {len(lam)}  dim={lam.shape[1]}  (max rank = #frames = {len(lam)})")
    for nm, X in [(f"laminar (frame {T_LAM})", lam), (f"developed (frame {T_DEV})", dev)]:
        r, n = rank_of(X)
        print(f"  {nm:24s}: modes for 50%={r[0.5]:3d}  90%={r[0.9]:3d}  95%={r[0.95]:3d}  99%={r[0.99]:3d}  / {n}")
    print("=> if 90-99% needs >>128, the 128-latent bottleneck is plausibly the recon limit (test it).")


if __name__ == "__main__":
    main()
