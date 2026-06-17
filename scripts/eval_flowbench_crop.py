"""Fast trivial-floor vet for an arbitrary crop window. On first run, caches (frame-120
full-res, Cd label, family, Re) for all sims -> subsequent crop experiments are instant
(no 270GB reload). Usage: eval_flowbench_crop.py R0 R1 C0 C1   (default 224x448 = 16 240 60 508)."""
import glob, re, os, sys, numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
FBR = "/gpfs/radev/home/ss5235/scratch/flowbench/FPO_NS_2D_1024x256"
CACHE = "/gpfs/radev/home/ss5235/scratch/flowbench/cd_cache.npz"


def cd_of(d, solid):
    u, p = d[..., 0], d[..., 2]; W = u.shape[2]
    cols = np.where(solid.any(0))[0]; rows = np.where(solid.any(1))[0]
    x1 = max(cols.min() - 20, 1); x2 = min(cols.max() + 200, W - 1)
    D = max(rows.max() - rows.min(), 1)
    flux = lambda x: (u[:, :, x] ** 2 + p[:, :, x]).sum(1)
    U = u[:, :, x1].mean(1)
    return float(np.median((flux(x1) - flux(x2)) / (0.5 * U ** 2 * D + 1e-9)))


def build_cache():
    files = sorted(glob.glob(FBR + "/*/*/Re_*.npz"))
    print(f"building cache from {len(files)} sims (one-time ~20min load)...", flush=True)
    fr, cd, fam, rr = [], [], [], []
    for f in files:
        d = np.load(f)['data'].astype(np.float32)
        solid = ~np.load(os.path.dirname(f) + "/input_geometry.npz")['mask'].astype(bool)
        c = cd_of(d, solid)
        if not np.isfinite(c): continue
        fr.append(d[120].copy()); cd.append(c)   # .copy() so the full 1.5GB sim can be freed
        fam.append(f.split("/FPO_NS_2D_1024x256/")[1].split("/")[0])
        rr.append(int(re.search(r'Re_(\d+)', f).group(1))); del d
    np.savez(CACHE, frames=np.stack(fr).astype(np.float32), cd=np.array(cd),
             fam=np.array(fam), re=np.array(rr))
    print("cached ->", CACHE, flush=True)


def main():
    if not os.path.exists(CACHE): build_cache()
    z = np.load(CACHE, allow_pickle=True)
    frames, cd, fam = z['frames'], z['cd'], z['fam']
    R0, R1, C0, C1 = (int(x) for x in (sys.argv[1:5] if len(sys.argv) > 4 else (16, 240, 60, 508)))
    print(f"=== crop rows[{R0}:{R1}] cols[{C0}:{C1}] = {R1-R0}x{C1-C0}  ({len(cd)} sims) ===")
    rng = np.random.default_rng(0); P = None
    Xcm, Xcms, Xrp = [], [], []
    for f in frames:
        snap = f[R0:R1, C0:C1]
        cm = snap.mean((0, 1)); cs = snap.std((0, 1))
        Xcm.append(cm); Xcms.append(np.concatenate([cm, cs]))
        flat = snap[::8, ::8].reshape(-1)
        if P is None: P = rng.standard_normal((flat.size, 128)).astype(np.float32)
        Xrp.append(flat @ P)
    Xcm, Xcms, Xrp = map(np.array, (Xcm, Xcms, Xrp)); Y = cd
    n = len(Y); idx = rng.permutation(n); tr, te = idx[:int(.8 * n)], idx[int(.8 * n):]
    def floor(X, nm):
        m, s = X[tr].mean(0), X[tr].std(0) + 1e-8; Xs = (X - m) / s
        print(f"  TRIVIAL {nm:16s} R2(Cd)={r2_score(Y[te], Ridge(1.0).fit(Xs[tr], Y[tr]).predict(Xs[te])):+.3f}")
    floor(Xcm, "channel-mean"); floor(Xcms, "channel mean+std"); floor(Xrp, "random-proj(128)")


if __name__ == "__main__":
    main()
