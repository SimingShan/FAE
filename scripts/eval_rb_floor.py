"""Trivial-floor vet for The Well rayleigh_benard, by STREAMING (hf://) a few frames per
(Ra,Pr) combo — no full download. Targets: log Rayleigh, log Prandtl. Train/test split is
by COMBO (file), so the floor measures generalization to UNSEEN parameter values (honest).
Channels: buoyancy, pressure, vel_x, vel_y at a developed frame."""
import re, numpy as np, h5py
from huggingface_hub import HfFileSystem
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
BASE = "datasets/polymathic-ai/rayleigh_benard/data/train"
FRAME, ICS = 150, [0, 12, 24]


def parse(fn):
    ra = float(re.search(r'Rayleigh_(.+?)_Prandtl', fn).group(1))
    pr = float(re.search(r'Prandtl_(.+?)\.hdf5', fn).group(1))
    return ra, pr


def main():
    fs = HfFileSystem()
    files = sorted(f for f in fs.ls(BASE, detail=False) if f.endswith(".hdf5"))
    print(f"{len(files)} combos")
    Xcm, Xcms, Xrp, lRa, lPr, combo = [], [], [], [], [], []
    rng = np.random.default_rng(0); P = None
    for ci, fp in enumerate(files):
        ra, pr = parse(fp.split("/")[-1])
        with fs.open(fp) as fo, h5py.File(fo) as h:
            for ic in ICS:
                b = h["t0_fields/buoyancy"][ic, FRAME]
                p = h["t0_fields/pressure"][ic, FRAME]
                v = h["t1_fields/velocity"][ic, FRAME]
                snap = np.stack([b, p, v[..., 0], v[..., 1]], -1).astype(np.float32)   # (512,128,4)
                cm = snap.mean((0, 1)); cs = snap.std((0, 1))
                flat = snap[::8, ::8].reshape(-1)
                if P is None: P = rng.standard_normal((flat.size, 128)).astype(np.float32)
                Xcm.append(cm); Xcms.append(np.concatenate([cm, cs])); Xrp.append(flat @ P)
                lRa.append(np.log10(ra)); lPr.append(np.log10(pr)); combo.append(ci)
    Xcm, Xcms, Xrp, lRa, lPr, combo = map(np.array, (Xcm, Xcms, Xrp, lRa, lPr, combo))
    print(f"{len(lRa)} samples | Ra vals {sorted(set(np.round(10**np.unique(lRa)).astype(int)))[:6]}... | "
          f"Pr vals {sorted(set(np.round(10**lPr, 2)))}")
    cperm = rng.permutation(len(files)); test_c = set(cperm[:max(7, len(files)//5)].tolist())
    te = np.array([c in test_c for c in combo]); tr = ~te
    def floor(X, y, nm):
        m, s = X[tr].mean(0), X[tr].std(0) + 1e-8; Xs = (X - m) / s
        print(f"  TRIVIAL {nm:16s} R2={r2_score(y[te], Ridge(1.0).fit(Xs[tr], y[tr]).predict(Xs[te])):+.3f}")
    print("=== Rayleigh (log) floor — combo-split (unseen Ra,Pr) ===")
    for X, nm in [(Xcm, "channel-mean"), (Xcms, "channel mean+std"), (Xrp, "random-proj")]: floor(X, lRa, nm)
    print("=== Prandtl (log) floor ===")
    for X, nm in [(Xcm, "channel-mean"), (Xcms, "channel mean+std"), (Xrp, "random-proj")]: floor(X, lPr, nm)
    print("=> low floor (esp. mean+std) => HARD target. Prandtl (diffusivity ratio) expected harder than Rayleigh.")


if __name__ == "__main__":
    main()
