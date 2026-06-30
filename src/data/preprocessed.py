"""ONE uniform loader over the preprocessed arrays in data/<ds>/ (built by scripts/data/preprocess.py).
Reads mmap'd RAW fields + meta, normalizes with TRAIN stats, serves single frames or clips — identical
interface for ns/shear/sw. Replaces the per-dataset adapters at train/probe time (fewer moving parts).

  mode="single" -> (C, H, W)              [MAE/JEPA, probe]
  mode="clip"   -> (C, clip_len, H, W)    [FAE temporal; trainer samples (t0, t0+Δ) inside]
Labels: (L,) per frame (constant within a trajectory). meta.dt_max is in SAVED-frame units.
"""
import os, json
import numpy as np, torch
from torch.utils.data import Dataset

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")


class PDEDataset(Dataset):
    def __init__(self, dataset, split, mode="clip", clip_len=2, start_stride=1, root=DATA_ROOT):
        d = os.path.join(root, dataset)
        self.meta = json.load(open(os.path.join(d, "meta.json")))
        self.fields = np.load(os.path.join(d, f"{split}_fields.npy"), mmap_mode="r")   # (N,T,C,H,W) RAW
        self.labels = np.load(os.path.join(d, f"{split}_labels.npy")).astype(np.float32)
        C = self.meta["C"]
        self.mean = np.asarray(self.meta["mean"], np.float32).reshape(C, 1, 1)
        self.std = np.asarray(self.meta["std"], np.float32).reshape(C, 1, 1)
        self.mode, self.clip_len = mode, clip_len
        self.label_names = self.meta["label_names"]
        self.dt_max = self.meta["dt_max"]
        self.stats = (self.mean, self.std)                              # adapter-compat
        N, T = self.fields.shape[:2]
        last = (T - clip_len + 1) if mode == "clip" else T
        self.index = [(n, t0) for n in range(N) for t0 in range(0, max(1, last), start_stride)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        n, t0 = self.index[i]
        y = torch.from_numpy(self.labels[n])
        if self.mode == "clip":
            x = np.asarray(self.fields[n, t0:t0 + self.clip_len], np.float32)          # (clip_len,C,H,W)
            x = (x - self.mean) / self.std
            return torch.from_numpy(x).permute(1, 0, 2, 3).contiguous(), y             # (C,clip_len,H,W)
        x = np.asarray(self.fields[n, t0], np.float32)                                 # (C,H,W)
        return torch.from_numpy((x - self.mean) / self.std), y
