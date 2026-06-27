"""Prep Digital Typhoon: manifest (6h stride) -> label stats -> downsampled cache -> trivial floor.
Run AFTER the WP tar is extracted (TYPHOON_IMG_ROOT / TYPHOON_META_ROOT point at WP/image, WP/metadata).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.data.typhoon import build_manifest, build_cache, TyphoonDataset
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

STRIDE = int(os.environ.get("TYPHOON_STRIDE_H", 6))
SIDE = int(os.environ.get("TYPHOON_SIDE", 128))

man = build_manifest(stride_h=STRIDE)
ntid = len(set(r["typhoon_id"] for r in man))
print(f"=== manifest: {ntid} typhoons, {len(man)} frames (uninterpolated intp==0 fixes, ~{STRIDE}h) ===", flush=True)
for col in ("wind", "pressure"):
    v = np.array([r[col] for r in man if np.isfinite(r[col])])
    if len(v):
        print(f"  {col}: n={len(v)} min={v.min():.1f} max={v.max():.1f} mean={v.mean():.1f} std={v.std():.1f}", flush=True)

print("=== building 128-downsampled cache (this reads the strided frames) ===", flush=True)
build_cache(side=SIDE, stride_h=STRIDE)

# ---- trivial floor: predict wind from CRUDE field statistics (coldest cloud top etc.) ----
def feats(ds):
    X, Y = [], []
    for i in range(len(ds)):
        x, y = ds[i]; a = x.numpy().reshape(-1)
        X.append([a.mean(), a.min(), a.max(), a.std(), np.quantile(a, 0.01)]); Y.append(float(y[0]))
    return np.array(X), np.array(Y)

for tgt in ("wind", "pressure"):
    tr = TyphoonDataset("train", side=SIDE, target=tgt)
    te = TyphoonDataset("test", side=SIDE, target=tgt, stats=tr.stats)
    if len(tr) == 0 or len(te) == 0:
        print(f"  [{tgt}] empty split — skip"); continue
    Xtr, Ytr = feats(tr); Xte, Yte = feats(te)
    m, s = Ytr.mean(), Ytr.std() + 1e-8
    reg = RidgeCV(alphas=[1e-2, 1e-1, 1, 10, 100, 1e3]).fit(Xtr, (Ytr - m) / s)
    r2 = r2_score((Yte - m) / s, reg.predict(Xte))
    print(f"=== TRIVIAL FLOOR [{tgt}] from crude IR stats: R2={r2:.3f}  "
          f"(n_train={len(tr)} n_test={len(te)}; THE BAR TO BEAT) ===", flush=True)
