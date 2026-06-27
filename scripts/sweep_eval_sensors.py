"""Eval-time sensor sweep on the EXISTING FAE checkpoints (no retrain).
Probe R^2 vs number of sensors fed to the frozen FAE encoder {64..1024, full-grid}, 3 seeds.
FAE was trained with mcnt_range [64,1024] random -> this whole sweep is in-distribution.
MAE/JEPA can only ingest the full grid -> shown as single reference points.
"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = "/gpfs/radev/scratch/lu_lu/ss5235/WFAE"; sys.path.insert(0, ROOT)
from src.config import load_config
from src.eval import probe

from src.config import ckpt_file
OUT = os.path.join(ROOT, "results/figures"); os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2]
COUNTS = [64, 128, 256, 512, 1024]            # sparse sensor counts
FULL = 128 * 128                              # full grid reference

cfg = load_config("configs/ns/fae/default.yaml")
res = {}                                      # count -> list of r2 over seeds
floor = None
for n in COUNTS + ["full"]:
    cfg.eval_fae_full_grid = (n == "full")
    if n != "full":
        cfg.eval_n_sensors = int(n)
    r2s = []
    for s in SEEDS:
        ck = ckpt_file("fae","fae_ns128",s)
        d = probe(cfg, ck); r2s.append(d["r2"]); floor = d["floor_r2"]
    res[str(n)] = r2s
    print(f"sensors={str(n):>5}  R2 = {np.mean(r2s):.3f} ± {np.std(r2s):.3f}   (seeds {[round(x,3) for x in r2s]})")
json.dump({"res": res, "floor": floor}, open(os.path.join(OUT, "sensor_sweep.json"), "w"), indent=2)

# ---- plot ----
xs = COUNTS
mean = [np.mean(res[str(n)]) for n in COUNTS]
std = [np.std(res[str(n)]) for n in COUNTS]
fm, fs = np.mean(res["full"]), np.std(res["full"])
fig, ax = plt.subplots(figsize=(7.2, 5))
ax.errorbar(xs, mean, yerr=std, fmt="-o", color="#d62728", capsize=3, lw=2, label="FAE (sparse sensors)")
ax.errorbar([FULL], [fm], yerr=[fs], fmt="D", color="#d62728", ms=9, capsize=3,
            label=f"FAE (full grid) {fm:.3f}")
# MAE/JEPA can ONLY use the full grid -> reference points at x=FULL
ax.scatter([FULL], [0.905], marker="s", s=80, color="#1f77b4", zorder=5, label="MAE (full grid only) 0.905")
ax.scatter([FULL], [0.656], marker="^", s=80, color="#2ca02c", zorder=5, label="JEPA (full grid only) 0.656")
ax.axhline(floor, color="gray", ls=":", lw=1, label=f"trivial floor {floor:.2f}")
ax.set_xscale("log", base=2)
ax.set_xticks(xs + [FULL]); ax.set_xticklabels([str(n) for n in xs] + ["full\n16384"])
ax.set_xlabel("# sensors fed to encoder (log scale)"); ax.set_ylabel("buoyancy probe $R^2$ (valid→test)")
ax.set_title("Eval-time sensor sweep — FAE degrades gracefully; ViTs need the full grid")
ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8.5, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "sensor_sweep.png"), dpi=150)
print("wrote sensor_sweep.png")
