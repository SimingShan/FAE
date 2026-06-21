"""Compute a drag coefficient Cd per FPO sim via a control-volume momentum+pressure
balance (x-momentum flux deficit between an upstream and a downstream plane, normalized
by 0.5*rho*U^2*D, time-averaged over the shedding) — then VET its trivial-baseline floor.
Cd is a SURFACE-integrated force => should NOT be recoverable from gross field stats => floor ~0.

Cd computed on the FULL 256x1024 domain (accurate wake); trivial features taken from the
256x256 CROP (what the model sees), so the vet matches the planned experiment."""
import glob, re, os, numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
FBR = "/gpfs/radev/home/ss5235/scratch/flowbench/FPO_NS_2D_1024x256"
C0, C1 = 60, 316


def cd_of(d, solid):                                  # d:(T,256,1024,3), solid:(256,1024)
    u, v, p = d[..., 0], d[..., 1], d[..., 2]
    W = u.shape[2]
    cols = np.where(solid.any(0))[0]; rows = np.where(solid.any(1))[0]
    x1 = max(cols.min() - 20, 1)                       # upstream of object
    x2 = min(cols.max() + 200, W - 1)                  # downstream in the wake
    D = max(rows.max() - rows.min(), 1)                # object frontal height (cells)
    flux = lambda x: (u[:, :, x] ** 2 + p[:, :, x]).sum(1)     # rho=1, integrate over y
    Fx = flux(x1) - flux(x2)                            # (T,) drag force per unit span
    U = u[:, :, x1].mean(1)                             # freestream
    Cd = Fx / (0.5 * U ** 2 * D + 1e-9)
    return float(np.median(Cd))                        # robust time-average


def main():
    files = sorted(glob.glob(FBR + "/*/*/Re_*.npz"))     # all geometry families
    Cd, Xcm, Xcms, Xrp, Re, Fam = [], [], [], [], [], []
    rng = np.random.default_rng(0); P = None
    for f in files:
        d = np.load(f)['data'].astype(np.float32)
        solid = ~np.load(os.path.dirname(f) + "/input_geometry.npz")['mask'].astype(bool)
        c = cd_of(d, solid)
        if not np.isfinite(c): continue
        Cd.append(c); Re.append(int(re.search(r'Re_(\d+)', f).group(1)))
        Fam.append(f.split("/FPO_NS_2D_1024x256/")[1].split("/")[0])
        snap = d[120, :, C0:C1].astype(np.float32)     # crop frame for trivial features
        cm = snap.mean((0, 1)); cs = snap.std((0, 1))
        Xcm.append(cm); Xcms.append(np.concatenate([cm, cs]))
        flat = snap[::8, ::8].reshape(-1)
        if P is None: P = rng.standard_normal((flat.size, 128)).astype(np.float32)  # 128 (<<n) to avoid overfit
        Xrp.append(flat @ P); del d
    Cd = np.array(Cd); Xcm, Xcms, Xrp, Re = map(np.array, (Xcm, Xcms, Xrp, Re)); Fam = np.array(Fam)
    print(f"=== Cd — {len(Cd)} sims across families {sorted(set(Fam))} ===")
    for fm in sorted(set(Fam)):
        cc = Cd[Fam == fm]; print(f"  {fm:10s}: {len(cc)} sims, Cd {cc.min():.2f}..{cc.max():.2f} (median {np.median(cc):.2f})")
    print(f"  corr(Cd, logRe)={np.corrcoef(Cd, np.log10(Re))[0,1]:+.2f}  "
          f"(low corr + cross-family Cd spread at fixed Re => shape-driven => hard)")
    Y = Cd
    n = len(Y); idx = rng.permutation(n); tr, te = idx[:int(.8 * n)], idx[int(.8 * n):]
    def floor(X, nm):
        m, s = X[tr].mean(0), X[tr].std(0) + 1e-8; Xs = (X - m) / s
        r = Ridge(alpha=1.0).fit(Xs[tr], Y[tr]); print(f"  TRIVIAL {nm:16s} R2(Cd)={r2_score(Y[te], r.predict(Xs[te])):+.3f}")
    floor(Xcm, "channel-mean"); floor(Xcms, "channel mean+std"); floor(Xrp, "random-proj(128)")
    print("--- per-family Cd floor (use ONLY that geometry family) ---")
    for fm in sorted(set(Fam)):
        msk = Fam == fm; Xf, Xrf, Yf = Xcms[msk], Xrp[msk], Cd[msk]
        nf = len(Yf); ix = rng.permutation(nf); t, e = ix[:int(.8 * nf)], ix[int(.8 * nf):]
        def fl(X):
            m, s = X[t].mean(0), X[t].std(0) + 1e-8; Xs = (X - m) / s
            return r2_score(Yf[e], Ridge(1.0).fit(Xs[t], Yf[t]).predict(Xs[e]))
        print(f"  {fm:10s} ({nf:3d} sims, Cd {Yf.min():.2f}..{Yf.max():.2f}): "
              f"mean+std R2={fl(Xf):+.3f}  random-proj R2={fl(Xrf):+.3f}")
    print("=> floor ~0 => Cd is a HARD target (valid benchmark).")


if __name__ == "__main__":
    main()
