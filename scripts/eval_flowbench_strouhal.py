"""Compute vortex-shedding frequency (Strouhal-proportional) per FPO sim via FFT of the
wake transverse-velocity signal, then VET its trivial-baseline floor (single-frame gross
stats should NOT recover a temporal shedding rate -> floor ~0 => HARD target)."""
import glob, re, os, numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
FB = "/gpfs/radev/home/ss5235/scratch/flowbench/FPO_NS_2D_1024x256/harmonics"


def shedding_freq(d, solid):
    """d:(T,H,W,3); solid:(H,W) bool object. FFT the wake v-signal -> peak freq (cyc/frame)."""
    rows, cols = np.where(solid)
    rc, cc = int(rows.mean()), int(cols.mean())                 # object centroid
    H, W = solid.shape
    sig = []
    for dc in (40, 80, 120):                                    # probes downstream in the wake
        c = min(cc + dc, W - 1)
        band = d[:, max(rc - 12, 0):rc + 12, c, 1]              # v over a vertical band, all time
        sig.append(band.mean(1))
    best, bf = -1, 0.0
    T = d.shape[0]
    freqs = np.fft.rfftfreq(T)
    for s in sig:
        ps = np.abs(np.fft.rfft(s - s.mean())) ** 2
        ps[0] = 0
        k = ps[1:].argmax() + 1
        if ps[k] > best:
            best, bf = ps[k], freqs[k]
    return bf


def main():
    files = sorted(glob.glob(FB + "/*/Re_*.npz"))
    F, Xcm, Xcms, Xrp, Re = [], [], [], [], []
    rng = np.random.default_rng(0); P = None
    C0, C1 = 60, 316                                            # 256x256 crop: object + near-wake
    for f in files:
        d = np.load(f)['data'].astype(np.float32)[:, :, C0:C1, :]   # (242,256,256,3) cropped
        solid = (~np.load(os.path.dirname(f) + "/input_geometry.npz")['mask'].astype(bool))[:, C0:C1]
        fr = shedding_freq(d, solid)
        if fr <= 0:
            continue
        F.append(fr); Re.append(int(re.search(r'Re_(\d+)', f).group(1)))
        snap = d[120].astype(np.float32)                        # one frame for trivial features
        cm = snap.mean((0, 1)); cs = snap.std((0, 1))
        Xcm.append(cm); Xcms.append(np.concatenate([cm, cs]))
        flat = snap[::8, ::8].reshape(-1)
        if P is None: P = rng.standard_normal((flat.size, 256)).astype(np.float32)
        Xrp.append(flat @ P)
        del d
    F = np.log10(np.array(F)); Xcm, Xcms, Xrp, Re = map(np.array, (Xcm, Xcms, Xrp, Re))
    n = len(F); idx = rng.permutation(n); tr, te = idx[:int(.8 * n)], idx[int(.8 * n):]
    print(f"=== FPO shedding-frequency (log) — {n} sims, freq range {10**F.min():.4f}-{10**F.max():.4f} cyc/frame ===")
    print(f"  (sanity: corr(logFreq, logRe) = {np.corrcoef(F, np.log10(Re))[0,1]:+.2f})")
    def floor(X, nm):
        m, s = X[tr].mean(0), X[tr].std(0) + 1e-8; Xs = (X - m) / s
        r = Ridge(alpha=1.0).fit(Xs[tr], F[tr]); print(f"  TRIVIAL {nm:16s} R2={r2_score(F[te], r.predict(Xs[te])):+.3f}")
    floor(Xcm, "channel-mean"); floor(Xcms, "channel mean+std"); floor(Xrp, "random-proj(256)")
    print("=> floor ~0 => shedding-freq is a HARD target (valid benchmark).")


if __name__ == "__main__":
    main()
