"""Paper fig 3 — sparse-regime probe: test probe-R2 vs # sensors (log2 x-axis, 2^x), FAE / MAE / JEPA.
Consumes results/sweeps/sensor_sweep_<ds>.json (produced by scripts/figs/sweep_sensors.py). plotstyle.
  python scripts/figs/fig3_probe_sensors.py --dataset typhoon
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import matplotlib; matplotlib.use("Agg")
from src.plotstyle import apply, COLORS, panels
apply()

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["shear", "typhoon", "ns"], required=True)
args = ap.parse_args()
J = json.load(open(f"results/sweeps/sensor_sweep_{args.dataset}.json"))
sens = np.array(J["sensors"]); tgt = J.get("target", "R²")

fig, ax = panels(1, side=5.5)
for meth in ["FAE", "MAE", "JEPA"]:
    if meth in J["probe"]:
        ax.plot(sens, J["probe"][meth], "-o", color=COLORS[meth], label=meth)
if "floor" in J:
    ax.axhline(J["floor"], color=COLORS["floor"], ls="--", lw=1.6, label=f"floor {J['floor']:+.2f}")
ax.set_xscale("log", base=2); ax.set_xticks(sens); ax.set_xticklabels([str(s) for s in sens])
ax.set_xlabel("# sensors"); ax.set_ylabel(f"test probe R²  ({tgt})")
ax.set_title(f"{args.dataset}: probe vs sensors"); ax.legend()
out = f"results/figs/{args.dataset}/fig3_probe_sensors.png"; os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=200); print(f"wrote {out}", flush=True)
