"""Preprocess ns / shear / sw ONCE into uniform, inspectable arrays under data/<ds>/.
Locked spec (2026-06-27):

  ds     spatial    stride  frames/traj  train traj   test traj   held-out test by
  ns     128x128    1       56           312 (104x3)  36 (12x3)   buoyancy value (interp)
  shear  128x256    4       50           252 (28x9)   28 (28x1)   seed (in-distribution)
  sw     128x128    1       72           135          15          (alpha,beta) value (interp)

Saves (per split):  {split}_fields.npy  (N, T, C, H, W) float32 RAW (mmap-able)
                    {split}_labels.npy  (N, L) float32
                    meta.json           label_names, stride, dt_max, H/W/C, train mean/std, split info
Normalization stats are computed on TRAIN only; the loader applies them (raw saved -> re-checkable).
Run ONCE on CPU:  python scripts/data/preprocess.py --dataset all
"""
import os, sys, glob, re, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, h5py, torch
import torch.nn.functional as F
from numpy.lib.format import open_memmap

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "data")
NS_ROOT = os.path.expanduser("~/scratch/ns_data")
SHEAR_ROOT = os.path.expanduser("~/scratch/the_well_data/shear_flow/data/train")
SW_PATH = os.path.expanduser("~/scratch/sw_data/shallow_water_full.npz")
RBC_ROOT = os.path.expanduser("~/scratch/the_well/datasets/rayleigh_benard/data/train")


def resize_to(x, H, W):                                   # (..., h, w) -> (..., H, W) bilinear
    if x.shape[-2:] == (H, W):
        return np.ascontiguousarray(x, np.float32)
    s = x.shape
    t = torch.from_numpy(np.ascontiguousarray(x, np.float32)).reshape(-1, 1, s[-2], s[-1])
    t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    return t.reshape(*s[:-2], H, W).numpy()


def interior_holdout(n, k):                               # k evenly-spaced INTERIOR indices (interpolation)
    return sorted(set(np.linspace(0.05 * n, 0.95 * n, k).round().astype(int).tolist()))


def chan_stats(mm):                                       # train memmap (N,T,C,H,W) -> per-channel mean,std
    C = mm.shape[2]; ssum = np.zeros(C); ssq = np.zeros(C); cnt = 0
    for i in range(mm.shape[0]):
        x = np.asarray(mm[i], np.float64)
        ssum += x.sum((0, 2, 3)); ssq += (x ** 2).sum((0, 2, 3)); cnt += x.shape[0] * x.shape[2] * x.shape[3]
    mean = ssum / cnt; std = np.sqrt(np.clip(ssq / cnt - mean ** 2, 0, None)) + 1e-6
    return mean, std


def finalize(ds, label_names, stride, dt_max, H, W, C, T, info):
    d = os.path.join(OUT, ds)
    mean, std = chan_stats(np.load(f"{d}/train_fields.npy", mmap_mode="r"))
    meta = dict(dataset=ds, label_names=label_names, stride=stride, dt_max=dt_max,
                H=H, W=W, C=C, frames_per_traj=T, mean=mean.tolist(), std=std.tolist(), **info)
    json.dump(meta, open(f"{d}/meta.json", "w"), indent=2)
    print(f"  [{ds}] mean={np.round(mean,3)} std={np.round(std,3)}  meta saved", flush=True)


# ----------------------------------------------------------------------------- NS
def prep_ns(H=128, W=128, stride=4, dt_max=1, n_traj=24):     # EXACT pre-shrink restore: train/test FILE split (matches old NSDataset), 78x24=1872 train; git 7f3358d FAE 0.929 > MAE 0.905
    splits = {"train": sorted(glob.glob(f"{NS_ROOT}/*_train_*.h5")),
              "test":  sorted(glob.glob(f"{NS_ROOT}/*_test_*.h5"))}                  # split by filename token (NOT pooled by buoyancy)
    buo = {}
    for flist in splits.values():
        for f in flist:
            with h5py.File(f, "r") as h:
                g = list(h.keys())[0]; buo[f] = float(np.asarray(h[g]["buo_y"][0]))
    T = 56 // stride
    os.makedirs(f"{OUT}/ns", exist_ok=True)
    for split, flist in splits.items():
        N = len(flist) * n_traj
        mm = open_memmap(f"{OUT}/ns/{split}_fields.npy", mode="w+", dtype=np.float32, shape=(N, T, 3, H, W))
        lab = np.zeros((N, 1), np.float32); j = 0
        for f in flist:
            with h5py.File(f, "r") as h:
                g = list(h.keys())[0]
                u = h[g]["u"][:n_traj, ::stride][:, :T]; vx = h[g]["vx"][:n_traj, ::stride][:, :T]; vy = h[g]["vy"][:n_traj, ::stride][:, :T]
            x = resize_to(np.stack([u, vx, vy], 2).astype(np.float32), H, W)        # (n_traj,T,3,H,W)
            for k in range(n_traj):
                mm[j] = x[k]; lab[j] = buo[f]; j += 1
        mm.flush(); np.save(f"{OUT}/ns/{split}_labels.npy", lab); print(f"  [ns/{split}] {mm.shape}", flush=True)
    finalize("ns", ["buoyancy"], stride, dt_max, H, W, 3, T,
             dict(held_out="train/test FILE split (matches old NSDataset)", n_train=len(splits["train"]) * n_traj, n_test=len(splits["test"]) * n_traj))


# -------------------------------------------------------------------------- SHEAR
def prep_shear(H=128, W=256, stride=4, dt_max=12, n_traj=32, n_test_seed=4):    # FULL-SCALE restore (all 32 seeds/cell; in-distribution seed split, 4 held out)
    files = sorted(glob.glob(f"{SHEAR_ROOT}/*.hdf5"))                                # 28 (Re,Sc) cells
    T = 200 // stride
    os.makedirs(f"{OUT}/shear", exist_ok=True)
    n_train_seed = n_traj - n_test_seed
    counts = {"train": len(files) * n_train_seed, "test": len(files) * n_test_seed}
    mm = {s: open_memmap(f"{OUT}/shear/{s}_fields.npy", mode="w+", dtype=np.float32, shape=(counts[s], T, 4, H, W)) for s in counts}
    lab = {s: np.zeros((counts[s], 2), np.float32) for s in counts}; j = {"train": 0, "test": 0}
    for f in files:
        with h5py.File(f, "r") as h:
            Re = float(h.attrs["Reynolds"]); Sc = float(h.attrs["Schmidt"])
            tr = h["t0_fields/tracer"][:n_traj, ::stride][:, :T]; pr = h["t0_fields/pressure"][:n_traj, ::stride][:, :T]
            ve = h["t1_fields/velocity"][:n_traj, ::stride][:, :T]
        x = resize_to(np.stack([tr, pr, ve[..., 0], ve[..., 1]], 2).astype(np.float32), H, W)   # (n_traj,T,4,H,W)
        y = np.array([np.log10(Re), Sc], np.float32)
        for k in range(n_traj):
            s = "test" if k >= n_train_seed else "train"                             # last seed(s) -> test (in-distribution)
            mm[s][j[s]] = x[k]; lab[s][j[s]] = y; j[s] += 1
    for s in counts:
        mm[s].flush(); np.save(f"{OUT}/shear/{s}_labels.npy", lab[s]); print(f"  [shear/{s}] {mm[s].shape}", flush=True)
    finalize("shear", ["logRe", "Sc"], stride, dt_max, H, W, 4, T,
             dict(held_out="seed (in-distribution; same 28 Re x Sc cells)", n_train=counts["train"], n_test=counts["test"]))


# ----------------------------------------------------------------------------- SW
def prep_sw(H=128, W=128, stride=1, dt_max=10, n_test=15):
    d = np.load(SW_PATH, mmap_mode="r"); out = d["outputs"]; params = np.asarray(d["params"], np.float32)
    N, Tn = out.shape[0], out.shape[1]; T = Tn // stride
    order = np.argsort(params[:, 0])                                                 # sort by alpha -> interior holdout
    test_traj = set(int(order[p]) for p in interior_holdout(N, n_test))
    splits = {"train": [t for t in range(N) if t not in test_traj],
              "test":  [t for t in range(N) if t in test_traj]}
    os.makedirs(f"{OUT}/sw", exist_ok=True)
    for split, tlist in splits.items():
        mm = open_memmap(f"{OUT}/sw/{split}_fields.npy", mode="w+", dtype=np.float32, shape=(len(tlist), T, 1, H, W))
        lab = np.zeros((len(tlist), 2), np.float32)
        for i, t in enumerate(tlist):
            x = resize_to(np.asarray(out[t, ::stride][:T])[:, None], H, W)            # (T,1,H,W)
            mm[i] = x; lab[i] = params[t]
        mm.flush(); np.save(f"{OUT}/sw/{split}_labels.npy", lab); print(f"  [sw/{split}] {mm.shape}", flush=True)
    finalize("sw", ["alpha", "beta"], stride, dt_max, H, W, 1, T,
             dict(held_out="(alpha,beta) value (interpolation)", n_train=len(splits["train"]), n_test=len(splits["test"])))


# ----------------------------------------------------------------------- RAYLEIGH-BENARD
def prep_rbc(H=128, W=256, stride=4, dt_max=8, n_traj=28, n_test_seed=4, crop_x=256):    # The Well rayleigh_benard: 5 Ra x 7 Pr cells, 28/40 traj (24 train+4 test/cell ~18GB, shear-scale); buoyancy-convection (NS scaled up)
    # VERIFIED against rayleigh_benard_Rayleigh_1e9_Prandtl_5.hdf5: t0=buoyancy/pressure, t1=velocity; native (x=512,y=128), y wall-bounded -> swap to (y=128, x=512).
    # x is PERIODIC/homogeneous -> CENTER-CROP x to crop_x (representative + full-res, no blur); y is wall-bounded -> KEEP FULL. crop_x=None -> native 512 (resolution-showcase variant).
    files = sorted(glob.glob(f"{RBC_ROOT}/*.hdf5"))                                  # one file per (Ra,Pr) cell
    assert files, f"no RBC files at {RBC_ROOT} — download The Well rayleigh_benard first"
    T = 200 // stride
    os.makedirs(f"{OUT}/rbc", exist_ok=True)
    n_train_seed = n_traj - n_test_seed
    counts = {"train": len(files) * n_train_seed, "test": len(files) * n_test_seed}
    mm = {s: open_memmap(f"{OUT}/rbc/{s}_fields.npy", mode="w+", dtype=np.float32, shape=(counts[s], T, 4, H, W)) for s in counts}
    lab = {s: np.zeros((counts[s], 2), np.float32) for s in counts}; j = {"train": 0, "test": 0}
    for f in files:
        with h5py.File(f, "r") as h:
            Ra = float(h.attrs["Rayleigh"]); Pr = float(h.attrs["Prandtl"])
            bu = h["t0_fields/buoyancy"][:n_traj, ::stride][:, :T]; pres = h["t0_fields/pressure"][:n_traj, ::stride][:, :T]
            ve = h["t1_fields/velocity"][:n_traj, ::stride][:, :T]
        bu, pres = bu.swapaxes(-1, -2), pres.swapaxes(-1, -2); ve = ve.swapaxes(-3, -2)   # (...,x,y)->(...,y,x) so vertical=H
        if crop_x:                                                                       # center-crop periodic x (full y kept), preserves resolution
            x0 = (bu.shape[-1] - crop_x) // 2
            bu, pres = bu[..., x0:x0 + crop_x], pres[..., x0:x0 + crop_x]; ve = ve[..., x0:x0 + crop_x, :]
        x = resize_to(np.stack([bu, pres, ve[..., 0], ve[..., 1]], 2).astype(np.float32), H, W)   # (n_traj,T,4,H,W)
        y = np.array([np.log10(Ra), np.log10(Pr)], np.float32)                       # logRa in [6,10], logPr in [-1,1]
        for k in range(n_traj):
            s = "test" if k >= n_train_seed else "train"
            mm[s][j[s]] = x[k]; lab[s][j[s]] = y; j[s] += 1
        for s in counts: mm[s].flush()                       # per-cell flush -> release dirty mmap pages (avoid 33GB GPFS SIGBUS)
        print(f"  [rbc] cell {os.path.basename(f)} done (train j={j['train']})", flush=True)
    for s in counts:
        mm[s].flush(); np.save(f"{OUT}/rbc/{s}_labels.npy", lab[s]); print(f"  [rbc/{s}] {mm[s].shape}", flush=True)
    finalize("rbc", ["logRa", "logPr"], stride, dt_max, H, W, 4, T,
             dict(held_out="seed (in-distribution; 5 Ra x 7 Pr cells; use held-out-CELL probe)", n_train=counts["train"], n_test=counts["test"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", choices=["ns", "shear", "sw", "rbc", "all"], default="all")
    a = ap.parse_args()
    for ds in (["ns", "shear", "sw"] if a.dataset == "all" else [a.dataset]):
        print(f"=== preprocessing {ds} ===", flush=True)
        {"ns": prep_ns, "shear": prep_shear, "sw": prep_sw, "rbc": prep_rbc}[ds]()
    print("done.", flush=True)
