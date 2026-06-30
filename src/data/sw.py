"""Spherical shallow-water (Galewsky barotropic jet on a rotating sphere; L-DeepONet generator).
Invert probe: recover the two perturbation parameters (alpha, beta) from the EVOLVED vorticity field.
Stored equirectangular (phi x theta), so the poles are oversampled — a 2D encoder sees that distortion.
  outputs (N, 72, 256, 256) vorticity ; params (N, 2) = (alpha, beta) ; one (alpha,beta) per trajectory.
mode="single" -> (1, side, side) frames (MAE/JEPA) ; mode="clip" -> (1, clip_len, side, side) (FAE temporal).
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

SW_PATH = os.path.expanduser("~/scratch/sw_data/shallow_water_full.npz")


def _rs(fr, side):
    fr = torch.from_numpy(np.asarray(fr, dtype=np.float32))[None, None]
    return F.interpolate(fr, size=(side, side), mode="bilinear", align_corners=False)[0]      # (1, side, side)


class SWDataset(Dataset):
    PARAMS = ["alpha", "beta"]

    def __init__(self, split, side=128, mode="single", clip_len=2, frame_stride=8, n_train=120, stats=None):
        d = np.load(SW_PATH, mmap_mode="r")
        out, params = d["outputs"], np.asarray(d["params"], dtype=np.float32)
        N, T = out.shape[0], out.shape[1]
        ids = range(0, n_train) if split == "train" else range(n_train, N)        # trajectory-disjoint split
        X, Y = [], []
        for t in ids:
            if mode == "clip":
                for f in range(0, T - clip_len + 1, frame_stride):
                    X.append(torch.stack([_rs(out[t, f + k], side) for k in range(clip_len)], 1))   # (1, clip_len, side, side)
                    Y.append(params[t])
            else:
                for f in range(0, T, frame_stride):
                    X.append(_rs(out[t, f], side)); Y.append(params[t])           # (1, side, side)
        self.X = torch.stack(X)
        self.Y = torch.tensor(np.array(Y))
        self.stats = stats if stats is not None else (float(self.X.mean()), float(self.X.std()) + 1e-8)
        self.X = (self.X - self.stats[0]) / self.stats[1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]
