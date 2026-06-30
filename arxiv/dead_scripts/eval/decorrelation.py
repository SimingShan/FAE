"""Decorrelation vs Δ (native frames) for ns/shear/sw — to set dt_max physically, uniformly.
D(Δ) = mean Pearson corr(x_t, x_{t+Δ}) over t & trajectories (flattened channels+space, spatially
downsampled for speed; correlation is scale-robust). Pick dt_max where corr drops into ~0.5-0.7
(non-trivial prediction, resists the identity-collapse). Δ here is in NATIVE frames (stride=1).
  python scripts/eval/decorrelation.py
"""
import os, sys, glob, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def corr_curve(trajs, deltas):
    """trajs: list of (T, F) flattened-frame arrays. -> mean corr(x_t, x_{t+Δ}) per Δ."""
    out = []
    for d in deltas:
        cs = []
        for x in trajs:
            T = len(x)
            if T <= d: continue
            a, b = x[:T - d], x[d:]
            a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
            num = (a * b).sum(1); den = np.sqrt((a * a).sum(1) * (b * b).sum(1)) + 1e-12
            cs.append((num / den).mean())
        out.append(np.mean(cs) if cs else np.nan)
    return np.array(out)


def load_ns(n=8, sub=2):
    fs = sorted(glob.glob(os.path.expanduser("~/scratch/ns_data/*_train_*.h5")))[:n]
    out = []
    for f in fs:
        with h5py.File(f, "r") as h:
            g = list(h.keys())[0]
            u = np.stack([h[g][k][0, :, ::sub, ::sub] for k in ("u", "vx", "vy")], 1)  # (T,3,h,w)
        out.append(u.reshape(u.shape[0], -1).astype(np.float32))
    return out


def load_shear(n=6, sub=4):
    fs = sorted(glob.glob(os.path.expanduser("~/scratch/the_well_data/shear_flow/data/train/*.hdf5")))
    fs = fs[::len(fs) // n][:n]
    out = []
    for f in fs:
        with h5py.File(f, "r") as h:
            tr = h["t0_fields/tracer"][0, :, ::sub, ::sub]
            ve = h["t1_fields/velocity"][0, :, ::sub, ::sub]
            x = np.stack([tr, ve[..., 0], ve[..., 1]], 1)
        out.append(x.reshape(x.shape[0], -1).astype(np.float32))
    return out


def load_sw(n=8, sub=4):
    d = np.load(os.path.expanduser("~/scratch/sw_data/shallow_water_full.npz"), mmap_mode="r")["outputs"]
    return [np.asarray(d[i, :, ::sub, ::sub]).reshape(d.shape[1], -1).astype(np.float32) for i in range(n)]


DS = {"ns": (load_ns, 56), "shear": (load_shear, 200), "sw": (load_sw, 72)}
fig, ax = plt.subplots(figsize=(7, 4.5))
print(f"{'dataset':8s} {'Δ@corr0.7':>10s} {'Δ@corr0.5':>10s}   corr at Δ=1,2,4,8,16")
for name, (loader, T) in DS.items():
    trajs = loader()
    deltas = np.unique(np.clip(np.r_[1:8, np.round(2 ** np.arange(3, np.log2(T - 1), 0.5)).astype(int)], 1, T - 1))
    c = corr_curve(trajs, deltas)
    def at(thr):
        below = np.where(c < thr)[0]
        return int(deltas[below[0]]) if len(below) else f">{int(deltas[-1])}"
    sample = {dd: f"{c[list(deltas).index(dd)]:.2f}" for dd in [1, 2, 4, 8, 16] if dd in deltas}
    print(f"{name:8s} {str(at(0.7)):>10s} {str(at(0.5)):>10s}   {sample}")
    ax.plot(deltas, c, "o-", label=name, ms=4)
ax.axhline(0.7, ls="--", c="gray", lw=0.8); ax.axhline(0.5, ls=":", c="gray", lw=0.8)
ax.set_xscale("log", base=2); ax.set_xlabel("Δ (native frames)"); ax.set_ylabel("corr(x_t, x_{t+Δ})")
ax.set_title("Decorrelation vs Δ — set dt_max where corr ≈ 0.5–0.7"); ax.legend(); ax.grid(alpha=0.3)
os.makedirs("results/figs", exist_ok=True); fig.tight_layout(); fig.savefig("results/figs/decorrelation.png", dpi=140)
print("wrote results/figs/decorrelation.png")
