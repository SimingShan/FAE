"""Well MHD_64 — 3D compressible MHD turbulence (64^3). Inverse probe: recover (Ma, Ms)
[sonic & Alfvenic Mach] from the evolved 3D field. 7 channels = density(1) + velocity(3) + magnetic(3).
Each file = one (Ma, Ms) combo, 5 trajectories x 100 timesteps; train/valid share the 10-combo grid
(disjoint trajectories). Lazy HDF5 (full set ~36GB) — per-item reads, handles cached per worker.
mode="single" -> (7, D, H, W) ; mode="clip" -> (7, clip_len, D, H, W) (FAE temporal).
"""
import os, glob, re
import numpy as np, torch, yaml, h5py
import torch.nn.functional as F
from torch.utils.data import Dataset

MHD_ROOT = os.path.expanduser("~/scratch/the_well/datasets/MHD_64")


def _norm_stats():                                              # per-channel (7,) mean/std from the Well stats.yaml
    s = yaml.safe_load(open(os.path.join(MHD_ROOT, "stats.yaml")))
    mean = [s["mean"]["density"]] + list(s["mean"]["velocity"]) + list(s["mean"]["magnetic_field"])
    std = [s["std"]["density"]] + list(s["std"]["velocity"]) + list(s["std"]["magnetic_field"])
    return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)


class MHDDataset(Dataset):
    PARAMS = ["logMa", "logMs"]

    def __init__(self, split, side=64, mode="clip", clip_len=2, frame_stride=5, n_traj=5, stats=None):
        self.files = sorted(glob.glob(os.path.join(MHD_ROOT, "data", split, "*.hdf5")))
        self.mode, self.clip_len, self.side = mode, clip_len, side
        self.stats = stats if stats is not None else _norm_stats()
        self.mean = self.stats[0].view(1, -1, 1, 1, 1); self.std = self.stats[1].view(1, -1, 1, 1, 1)
        self._cache = {}                                        # h5 handles, populated lazily per worker (fork-safe)
        self.index, labels = [], []
        for fp in self.files:
            Ma, Ms = map(float, re.search(r"Ma_([\d.]+)_Ms_([\d.]+)\.hdf5$", fp).groups())
            with h5py.File(fp, "r") as h:
                ntr, T = h["t0_fields/density"].shape[:2]
            y = np.array([np.log(Ma), np.log(Ms)], dtype=np.float32)
            last = (T - clip_len + 1) if mode == "clip" else T
            for tr in range(min(n_traj, ntr)):
                for f in range(0, last, frame_stride):
                    self.index.append((fp, tr, f)); labels.append(y)
        self.Y = torch.tensor(np.array(labels))

    def _h5(self, fp):
        h = self._cache.get(fp)
        if h is None: h = self._cache[fp] = h5py.File(fp, "r")
        return h

    def _read(self, fp, tr, f0, n):                             # -> (n, 7, D, H, W) raw, channel order [rho, v(3), B(3)]
        h = self._h5(fp)
        d = h["t0_fields/density"][tr, f0:f0 + n][..., None]    # (n, D, H, W, 1)
        v = h["t1_fields/velocity"][tr, f0:f0 + n]              # (n, D, H, W, 3)
        b = h["t1_fields/magnetic_field"][tr, f0:f0 + n]        # (n, D, H, W, 3)
        x = np.concatenate([d, v, b], axis=-1)                  # (n, D, H, W, 7)
        return torch.from_numpy(x).permute(0, 4, 1, 2, 3).float()

    def _norm(self, x):                                         # (n,7,D,H,W) -> resize cube to side, standardize per channel
        if self.side != x.shape[-1]:
            x = F.interpolate(x, size=(self.side,) * 3, mode="trilinear", align_corners=False)
        return (x - self.mean) / self.std

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        fp, tr, f0 = self.index[i]
        if self.mode == "clip":
            x = self._norm(self._read(fp, tr, f0, self.clip_len))      # (clip_len, 7, D,H,W)
            return x.permute(1, 0, 2, 3, 4), self.Y[i]                 # (7, clip_len, D,H,W)
        return self._norm(self._read(fp, tr, f0, 1))[0], self.Y[i]     # (7, D,H,W)
