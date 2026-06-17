"""Surface (form) drag coefficient via the object-boundary PRESSURE integral, then vet its
trivial floor. Form drag F_x = -∮ p n_x dS over the body = Σ_domain p·(∂χ/∂x) with χ the solid
indicator (∂χ/∂x is the surface delta * normal). This depends on the front/back pressure
DISTRIBUTION on the body, NOT a bulk field integral — so unlike the momentum-flux Cd it should
NOT be recoverable from gross field stats. Time-averaged (median) over the shedding.
Cd computed on the FULL domain; trivial features from the 256x256 crop (what the model sees)."""
import glob, re, os, numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
FBR = "/gpfs/radev/home/ss5235/scratch/flowbench/FPO_NS_2D_1024x256"
C0, C1 = 60, 316


def surface_cd(d, solid):
    chi = solid.astype(np.float32)
    dchidx = np.gradient(chi, axis=1)                  # ∂χ/∂x  (x = streamwise = axis 1 of (H,W))
    p = d[..., 2].copy()                               # (T,H,W) pressure
    p[:, solid] = 0.0                                  # zero pressure inside the solid
    Fx = (p * dchidx[None]).sum((1, 2))                # (T,) form drag (front high-p dominates -> >0)
    rows = np.where(solid.any(1))[0]; D = max(rows.max() - rows.min(), 1)
    cols = np.where(solid.any(0))[0]; x1 = max(cols.min() - 20, 1)
    U = d[:, :, x1, 0].mean(1)                          # freestream u upstream
    return float(np.median(Fx / (0.5 * U ** 2 * D + 1e-9)))


def main():
    files = sorted(glob.glob(FBR + "/*/*/Re_*.npz"))
    Cd, Xcm, Xcms, Xrp, Re, Fam = [], [], [], [], [], []
    rng = np.random.default_rng(0); P = None
    for f in files:
        d = np.load(f)['data'].astype(np.float32)
        solid = ~np.load(os.path.dirname(f) + "/input_geometry.npz")['mask'].astype(bool)
        c = surface_cd(d, solid)
        if not np.isfinite(c):
            del d; continue
        Cd.append(c); Re.append(int(re.search(r'Re_(\d+)', f).group(1)))
        Fam.append(f.split("/FPO_NS_2D_1024x256/")[1].split("/")[0])
        snap = d[120, :, C0:C1].astype(np.float32)
        cm = snap.mean((0, 1)); cs = snap.std((0, 1))
        Xcm.append(cm); Xcms.append(np.concatenate([cm, cs]))
        flat = snap[::8, ::8].reshape(-1)
        if P is None: P = rng.standard_normal((flat.size, 128)).astype(np.float32)
        Xrp.append(flat @ P); del d
    Cd = np.array(Cd); Xcm, Xcms, Xrp, Re, Fam = map(np.array, (Xcm, Xcms, Xrp, Re, Fam))
    print(f"=== SURFACE (form) Cd — {len(Cd)} sims across {sorted(set(Fam))} ===")
    for fm in sorted(set(Fam)):
        cc = Cd[Fam == fm]; print(f"  {fm:10s}: {len(cc)} sims, Cd {cc.min():.2f}..{cc.max():.2f} (median {np.median(cc):.2f})")
    print(f"  corr(surfCd, logRe)={np.corrcoef(Cd, np.log10(Re))[0,1]:+.2f}")
    n = len(Cd); idx = rng.permutation(n); tr, te = idx[:int(.8 * n)], idx[int(.8 * n):]
    def floor(X, nm):
        m, s = X[tr].mean(0), X[tr].std(0) + 1e-8; Xs = (X - m) / s
        print(f"  TRIVIAL {nm:16s} R2(surfCd)={r2_score(Cd[te], Ridge(1.0).fit(Xs[tr], Cd[tr]).predict(Xs[te])):+.3f}")
    floor(Xcm, "channel-mean"); floor(Xcms, "channel mean+std"); floor(Xrp, "random-proj(128)")
    print("=> floor ~0 => surface-Cd is a HARD target (valid linear-probe benchmark).")


if __name__ == "__main__":
    main()
