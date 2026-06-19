"""PDE-Arena NavierStokes-2D-conditioned (the SSLForPDEs benchmark) for our FAE.
Fields u (smoke), vx, vy -> 3 channels, native 128x128. Label = buoyancy (constant per file,
in filename). Clips of clip_len frames at frame_stride. Reads trajectories/frames SELECTIVELY
from each h5 (no full 704MB load). Mirrors well2d / flowbench datasets so the trainers swap."""
import os, glob, re, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import Dataset
NS = os.environ.get("NS_DATA_ROOT", os.path.expanduser("~/scratch/ns_data"))


def buoyancy(fn):
    return float(re.search(r'_(?:train|valid|test)_\d+_([0-9]+\.[0-9]+)', os.path.basename(fn)).group(1))


class NSDataset(Dataset):
    def __init__(self, split, side=128, mode="clip", clip_len=2, frame_stride=4,
                 n_traj=8, stats=None, n_frames=None):
        import h5py
        self.side, self.mode, self.clip_len, self.fstride = side, mode, max(clip_len, 2), frame_stride
        grp = split  # "train" | "valid" | "test" -> matches both filename token and h5 group
        files = sorted(glob.glob(f"{NS}/*_{grp}_*.h5"))
        clips, labels = [], []
        for f in files:
            buo = buoyancy(f)
            with h5py.File(f, "r") as h:
                g = h[grp]
                ntr = min(n_traj, g["u"].shape[0])
                for tj in range(ntr):
                    u = g["u"][tj, ::frame_stride]; vx = g["vx"][tj, ::frame_stride]; vy = g["vy"][tj, ::frame_stride]
                    x = np.stack([u, vx, vy], 1).astype(np.float32)        # (T',3,H,W)
                    if x.shape[-1] != side:
                        x = F.interpolate(torch.from_numpy(x), size=(side, side), mode="bilinear", align_corners=False).numpy()
                    clips.append(x); labels.append(buo)
        self.clips = clips; self.labels = np.array(labels, dtype=np.float32)
        self.lengths = [c.shape[0] for c in clips]
        if stats is None:
            cat = np.concatenate([c for c in clips]).reshape(-1, 3, side, side)
            self.stats = (cat.mean((0, 2, 3)), cat.std((0, 2, 3)) + 1e-6)
        else:
            self.stats = stats
        m, s = self.stats
        for c in self.clips:
            c -= m[None, :, None, None]; c /= s[None, :, None, None]
        self.idx = []
        for si, L in enumerate(self.lengths):
            span = self.fstride * 0 + (self.clip_len - 1)
            for t0 in range(0, max(1, L - (self.clip_len - 1))):
                self.idx.append((si, t0))
        self.logRe = np.log10(np.clip(self.labels, 1e-6, None))  # alias so probe2 'logRe' slot = log-buoyancy
        self.Sc = self.labels                                    # alias 'Sc' slot = raw buoyancy
        self.buoyancy = self.labels

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        si, t0 = self.idx[i]
        if self.mode == "clip":
            fr = [self.clips[si][min(t0 + k, self.lengths[si] - 1)] for k in range(self.clip_len)]
            x = torch.from_numpy(np.stack(fr, 1))                # (3, clip_len, H, W)
        else:
            x = torch.from_numpy(self.clips[si][t0])             # (3,H,W)
        return x, np.array([self.labels[si]], dtype=np.float32)
