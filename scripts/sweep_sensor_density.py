"""Information-ceiling sweep: how much (logRe, Sc) signal survives at sensor
density N?

For each N, sample N random spatial locations (the same access an FAE encoder
gets), read their 4 channel values, random-project to 320d (matching FAE's
representation dim AND the random-projection trivial floor), and ridge-probe
logRe/Sc. N = full reproduces the 0.25 / 0.16 floor exactly. The curve says
where probe R^2 plateaus — i.e. the minimum density that captures the physics,
and therefore whether 256-512 sensors is starving the encoder.

  python scripts/sweep_sensor_density.py            # default densities
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from src.data.well2d import ShearFlowSnapshotDataset
from src.metrics import lin_probe

PARAMS = ["logRe", "Sc"]


def probe_at(feat_tr, feat_va, Ytr, Yva, proj, rng):
    """feat_*: (N, 4*Ns) sensor features -> random-project -> ridge R^2 per param."""
    W = rng.standard_normal((feat_tr.shape[1], proj)).astype(np.float32) / np.sqrt(feat_tr.shape[1])
    Ztr, Zva = feat_tr @ W, feat_va @ W
    out = []
    for j in range(2):
        ytr, yva = Ytr[:, j], Yva[:, j]
        ym, ys = ytr.mean(), ytr.std() + 1e-8
        out.append(lin_probe(Ztr, (ytr - ym) / ys, Zva, (yva - ym) / ys))
    return out                                   # [r2_logRe, r2_Sc]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seed", type=int, default=24)
    ap.add_argument("--densities", type=int, nargs="+",
                    default=[128, 256, 512, 1024, 2048, 4096])
    ap.add_argument("--proj", type=int, default=320)
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    tr = ShearFlowSnapshotDataset("train", n_seed=args.n_seed, frame_stride=12, side=128)
    va = ShearFlowSnapshotDataset("valid", n_seed=8, frame_stride=12, side=128, stats=tr.stats)
    NPIX = tr.fields.shape[-1] * tr.fields.shape[-2]                 # 128*128
    Ftr = tr.fields.reshape(len(tr), 4, NPIX)                        # (N,4,16384)
    Fva = va.fields.reshape(len(va), 4, NPIX)
    Ytr = np.stack([tr.logRe, tr.Sc], 1)
    Yva = np.stack([va.logRe, va.Sc], 1)
    print(f"=== sensor-density information ceiling  train {len(tr)} / valid {len(va)}  "
          f"NPIX={NPIX}  proj={args.proj}d  trials={args.trials} ===", flush=True)
    print(f"{'density':>9} {'frac':>6}   {'logRe R2':>16}   {'Sc R2':>16}", flush=True)

    densities = [n for n in args.densities if n < NPIX] + [NPIX]
    for N in densities:
        res = []
        for t in range(args.trials):
            rng = np.random.default_rng(1000 + t)
            idx = (np.arange(NPIX) if N >= NPIX
                   else rng.choice(NPIX, N, replace=False))
            ftr = Ftr[:, :, idx].reshape(len(tr), -1)
            fva = Fva[:, :, idx].reshape(len(va), -1)
            res.append(probe_at(ftr, fva, Ytr, Yva, args.proj, rng))
        res = np.array(res)                       # (trials, 2)
        m, s = res.mean(0), res.std(0)
        tag = " (full=floor)" if N >= NPIX else ""
        print(f"{N:>9} {N/NPIX:>6.1%}   {m[0]:+.3f} ± {s[0]:.3f}   "
              f"{m[1]:+.3f} ± {s[1]:.3f}{tag}", flush=True)


if __name__ == "__main__":
    main()
