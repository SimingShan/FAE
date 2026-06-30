"""Merge existing shallow_water.npz (samples 0..28) + array chunks -> full shallow_water_full.npz."""
import numpy as np, glob, os
base = os.path.expanduser("~/scratch/sw_data")
parts = []
ex = f"{base}/shallow_water.npz"
if os.path.exists(ex):
    d = np.load(ex); parts.append((0, d["params"], d["inputs"], d["outputs"]))
for f in glob.glob(f"{base}/chunks/sw_chunk_*.npz"):
    d = np.load(f); start = int(os.path.basename(f).split("_")[-1].split(".")[0])
    parts.append((start, d["params"], d["inputs"], d["outputs"]))
parts.sort(key=lambda x: x[0])
P = np.concatenate([p[1] for p in parts]); I = np.concatenate([p[2] for p in parts]); O = np.concatenate([p[3] for p in parts])
out = f"{base}/shallow_water_full.npz"
np.savez(out, params=P, inputs=I, outputs=O)
print(f"merged {len(parts)} parts (starts {[p[0] for p in parts]}) -> {O.shape[0]} samples, outputs {O.shape} -> {out}", flush=True)
