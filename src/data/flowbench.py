"""FlowBench FPO (flow-past-object) — complex geometry, time-dependent. Crop [60:316] ->
256x256 (object + near-wake, native res, NO aspect distortion) -> resize `side`. 3 channels
(u,v,p). Probe target = log10 shedding frequency (Strouhal-proportional), computed by FFT of
the wake v-signal (a global flow property; cropping the input doesn't change it).

Mirrors src/data/well2d datasets (yields (clip,(label,)) for clips, (frame,(label,)) snapshots)
so train_fae_predict / train_baseline can swap datasets with minimal change.
Split by GEOMETRY case (generalize to unseen shapes): case_id % 5 == 0 -> valid, else train.
"""
import os, glob, re, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import Dataset

FB = os.environ.get("FLOWBENCH_DIR", os.path.expanduser("~/scratch/flowbench")) + "/FPO_NS_2D_1024x256"
C0, C1 = 60, 316                                  # 256-wide crop (object cols ~100-158)


def shedding_freq(v, solid):                       # v:(T,H,W) cropped, solid:(H,W)
    rows, cols = np.where(solid); rc, cc = int(rows.mean()), int(cols.mean())
    W = solid.shape[1]; T = v.shape[0]; freqs = np.fft.rfftfreq(T)
    best, bf = -1.0, 0.0
    for dc in (40, 80, 120):
        c = min(cc + dc, W - 1)
        s = v[:, max(rc - 12, 0):rc + 12, c].mean(1)
        ps = np.abs(np.fft.rfft(s - s.mean())) ** 2; ps[0] = 0
        k = ps[1:].argmax() + 1
        if ps[k] > best: best, bf = ps[k], freqs[k]
    return bf


class FlowBenchFPO(Dataset):
    def __init__(self, split, family="harmonics", side=224, mode="clip", clip_len=2,
                 frame_stride=8, stats=None, n_frames=None):
        self.side, self.mode, self.clip_len = side, mode, max(clip_len, 2)
        self.fstride = frame_stride
        sims = sorted(glob.glob(f"{FB}/{family}/*/Re_*.npz"))
        keep = [f for f in sims if (int(os.path.basename(os.path.dirname(f))) % 5 == 0) == (split == "valid")]
        clips, labels = [], []
        for f in keep:
            d = np.load(f)['data'].astype(np.float32)[:, :, C0:C1, :]          # (T,256,256,3)
            solid = (~np.load(os.path.dirname(f) + "/input_geometry.npz")['mask'].astype(bool))[:, C0:C1]
            sf = shedding_freq(d[..., 1], solid)
            if sf <= 0: continue
            x = torch.from_numpy(d).permute(0, 3, 1, 2)                        # (T,3,256,256)
            x = F.interpolate(x, size=(side, side), mode="bilinear", align_corners=False)
            x[:, :, F.interpolate(torch.from_numpy(solid)[None, None].float(), (side, side))[0, 0] > 0.5] = 0.0
            clips.append(x.numpy()); labels.append(np.log10(sf))
        self.clips = clips; self.labels = np.array(labels, dtype=np.float32)   # (N,)
        self.lengths = [c.shape[0] for c in clips]
        # per-channel z-norm (shared stats)
        if stats is None:
            allf = np.concatenate([c.reshape(-1, side, side) for c in clips])
            cat = np.concatenate([c for c in clips]).reshape(-1, 3, side, side)
            self.stats = (cat.mean((0, 2, 3)), cat.std((0, 2, 3)) + 1e-6)
        else:
            self.stats = stats
        m, s = self.stats
        for c in self.clips: c -= m[None, :, None, None]; c /= s[None, :, None, None]
        # index: (sim, start) windows
        self.idx = []
        for si, L in enumerate(self.lengths):
            span = self.fstride * (self.clip_len - 1)
            for t0 in range(0, max(1, L - span), self.fstride):
                self.idx.append((si, t0))
        self.Strouhal = self.labels  # alias for probe naming

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        si, t0 = self.idx[i]
        if self.mode == "clip":
            fr = [self.clips[si][min(t0 + k * self.fstride, self.lengths[si] - 1)] for k in range(self.clip_len)]
            x = torch.from_numpy(np.stack(fr, 1))            # (3, clip_len, H, W)
        else:
            x = torch.from_numpy(self.clips[si][t0])         # (3,H,W)
        return x, np.array([self.labels[si]], dtype=np.float32)
