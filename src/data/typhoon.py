"""Digital Typhoon (WP basin) for the FAE harness — single-channel IR brightness-temperature fields
+ best-track labels (wind / pressure). Frames are HOURLY; we stride to 6h (the native best-track
cadence) to drop redundancy and use UNINTERPOLATED labels. Split BY TYPHOON ID (no frame leakage).
512 -> `side` downsample. Mirrors src/data/ns.py so the trainers/probe swap in. Label = wind (default).

Prep (the data is a 54GB split tar extracted separately):
  1) build_manifest()  scan WP/metadata/<id>.csv -> per-frame (typhoon_id, file, wind, pressure, ...) strided
  2) build_cache()     load+downsample the strided frames -> per-typhoon arrays in CACHE (loaded in RAM)
Then `TyphoonDataset(split, target="wind")` just loads the cache.
"""
import os, glob, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import Dataset

IMG = os.environ.get("TYPHOON_IMG_ROOT", os.path.expanduser("~/scratch/digital_typhoon_ext/WP/image"))
META = os.environ.get("TYPHOON_META_ROOT", os.path.expanduser("~/scratch/typhoon_meta/WP/metadata"))
CACHE = os.environ.get("TYPHOON_CACHE", os.path.expanduser("~/scratch/typhoon_cache"))


def _find_img_col(cols):
    for c in ("file_1", "file", "image", "filename"):
        if c in cols:
            return c
    raise KeyError(f"no image-path column in metadata CSV (cols={list(cols)})")


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def build_manifest(meta_dir=META, stride_h=6):
    """Scan per-typhoon metadata CSVs (stdlib csv, no pandas) -> list of dicts
    {typhoon_id, file, wind, pressure, lat, lng, grade}. Keeps rows with finite wind OR pressure;
    strides to every `stride_h` hours when an 'hour' column exists."""
    import csv as _csv
    out = []
    for csvf in sorted(glob.glob(os.path.join(meta_dir, "*.csv"))):
        tid = os.path.splitext(os.path.basename(csvf))[0]
        with open(csvf, newline="") as fh:
            rd = _csv.DictReader(fh)
            ic = _find_img_col(rd.fieldnames)
            has_intp = "intp" in (rd.fieldnames or [])
            for r in rd:
                fn = str(r.get(ic) or "").strip()
                if not fn:                                   # no satellite image at this best-track time
                    continue
                if has_intp:                                 # native cadence = UNINTERPOLATED fixes (intp==0, ~6h)
                    try:
                        if int(float(r.get("intp") or 0)) != 0:
                            continue
                    except ValueError:
                        pass
                elif stride_h > 1 and r.get("hour") not in (None, ""):
                    try:
                        if int(round(float(r["hour"]))) % stride_h != 0:
                            continue
                    except ValueError:
                        pass
                w, p = _f(r.get("wind")), _f(r.get("pressure"))
                if not (np.isfinite(w) or np.isfinite(p)):
                    continue
                out.append(dict(typhoon_id=tid, file=os.path.basename(fn), wind=w, pressure=p,
                                lat=_f(r.get("lat")), lng=_f(r.get("lng")), grade=_f(r.get("grade"))))
    return out


def build_cache(side=128, stride_h=6, meta_dir=META, img_root=IMG, cache=CACHE, limit_typhoons=None):
    """Load+downsample every strided frame -> one .npz per typhoon (x:(T,1,side,side), wind, pressure)
    under CACHE, so TyphoonDataset can load in RAM (like NS)."""
    import h5py
    man = build_manifest(meta_dir, stride_h)
    os.makedirs(cache, exist_ok=True)
    by_tid = {}
    for r in man:
        by_tid.setdefault(r["typhoon_id"], []).append(r)
    tids = sorted(by_tid)
    if limit_typhoons:
        tids = tids[:limit_typhoons]
    kept = 0
    for tid in tids:
        xs, w, p = [], [], []
        for r in by_tid[tid]:
            if not r["file"]:
                continue
            fp = os.path.join(img_root, tid, r["file"])
            if not os.path.isfile(fp):
                continue
            with h5py.File(fp, "r") as h:
                a = np.array(h["Infrared"], dtype=np.float32)              # (512,512) K
            if a.shape[-1] != side:
                a = F.interpolate(torch.from_numpy(a)[None, None], size=(side, side), mode="area")[0, 0].numpy()
            xs.append(a[None]); w.append(r["wind"]); p.append(r["pressure"])
        if not xs:
            continue
        np.savez(os.path.join(cache, f"{tid}.npz"), x=np.stack(xs).astype(np.float32),
                 wind=np.array(w, np.float32), pressure=np.array(p, np.float32))
        kept += 1
    print(f"cached {kept} typhoons -> {cache}", flush=True)
    return kept


class TyphoonDataset(Dataset):
    def __init__(self, split, side=128, target="wind", mode="single", clip_len=2,
                 cache=CACHE, stats=None, splits=(0.7, 0.15, 0.15), seed=0, max_typhoons=None):
        self.side, self.mode, self.clip_len, self.target = side, mode, max(clip_len, 2), target
        tids = sorted(t[:-4] for t in os.listdir(cache) if t.endswith(".npz"))
        rng = np.random.default_rng(seed); rng.shuffle(tids)
        if max_typhoons:
            tids = tids[:max_typhoons]
        n = len(tids); a, b = int(splits[0] * n), int((splits[0] + splits[1]) * n)
        keep = {"train": tids[:a], "valid": tids[a:b], "test": tids[b:]}[split]
        self.seqs, self.labs = [], []                                     # per-typhoon (T,1,H,W), (T,2)=[wind,pressure]
        for tid in keep:
            d = np.load(os.path.join(cache, f"{tid}.npz"))
            w = d["wind"].astype(np.float32); p = d["pressure"].astype(np.float32)
            m = np.isfinite(w) & np.isfinite(p)                           # keep frames with BOTH labels
            if m.sum() == 0:
                continue
            self.seqs.append(d["x"][m]); self.labs.append(np.stack([w[m], p[m]], 1))   # (T,2) [wind,pressure]
        self.lengths = [s.shape[0] for s in self.seqs]
        if stats is None:                                                 # IR standardization (single channel)
            cat = np.concatenate([s.reshape(-1) for s in self.seqs])
            self.stats = (float(cat.mean()), float(cat.std()) + 1e-6)
        else:
            self.stats = stats
        m, s = self.stats
        for x in self.seqs:
            x -= m; x /= s
        self.idx = [(si, t0) for si, L in enumerate(self.lengths) for t0 in range(max(1, L - (self.clip_len - 1)))]

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        si, t0 = self.idx[i]; L = self.lengths[si]
        if self.mode == "clip":
            fr = [self.seqs[si][min(t0 + k, L - 1)] for k in range(self.clip_len)]
            x = torch.from_numpy(np.stack(fr, 1))                         # (1, clip_len, H, W)
        else:
            x = torch.from_numpy(self.seqs[si][t0])                       # (1, H, W)
        return x, self.labs[si][t0].astype(np.float32)                    # (2,) [wind, pressure]
